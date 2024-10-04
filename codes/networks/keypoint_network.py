
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import cv2

from .segmentation_network import LWTLDecoder
from .netOps import geom_mapping_toolkit as geom_utils
from .transformer_network import TransPoseR
from .netOps.nonlinear_weighted_blind_pnp import NonlinearWeightedBlindPnP, NonlinearWeightedOrthBlindPnP


def perspective_to_orthographic_estimation(theta0, verts):
    """
    This function takes 3D vertices and perspective transformation parameters as input and estimates the
    orthographic projection parameters. It transforms the vertices using the rotation matrix and adjusts
    the translation and scaling based on the mean z-depth, which is necessary for orthographic projection.
    The output parameters include the rotation, scaling factor, and adjusted translation for orthographic projection.

    :param theta0: A tensor representing the perspective transformation parameters.
    :param verts:  A tensor of 3D vertices, representing the shape or object to be transformed.
    :return:
    """

    # Rotation vector
    rotvec = theta0[:, :3]
    # Translation vector
    transvec = theta0[:, 3:]
    # Gt 3D rotation matrix
    R = geom_utils.angle_axis_to_rotation_matrix(rotvec)
    # Transform vertices using rotation matrix
    verts_transformed = torch.einsum('brs,bms->bmr', (R, verts))
    #print(transvec[...,-1])

    # Estimate the scaling factor (s_est)
    mean_z = (verts_transformed[...,-1]+transvec[...,-1].unsqueeze(-1)).mean(dim=-1, keepdims=True)
    s_est = 1. / mean_z
    # Estimate the translation for the orthographic projection (t_est):
    t_est = transvec[...,:2] * s_est

    return torch.cat((rotvec, s_est, t_est), dim=-1)


def ransac_p3p(P, p2d, p3d, topK=16, reprojectionError=0.05):
    """
    This function, performs RANSAC (Random Sample Consensus) with the P3P (Perspective-3-Point) algorithm
    to estimate the pose (rotation and translation) of a camera from 2D-3D point correspondences.
    The RANSAC approach helps ignore outliers (incorrect correspondences), and P3P provides a way to solve
    the camera pose estimation from 3D points.

    :param P: A tensor containing probabilities or confidence scores for the correspondences between 2D and 3D points (e.g., based on feature matching).
    :param p2d: A tensor of size (B, N, 2) containing 2D image points (with 2 coordinates: x and y). B is the batch size, and N is the number of points.
    :param p3d: A tensor of size (B, N, 3) containing 3D points corresponding to the 2D points. Each 3D point has x, y, and z coordinates.
    :param topK: The number of top correspondences to use in the RANSAC algorithm (default is 16).
    :param reprojectionError: The maximum allowed reprojection error for a point to be considered an inlier (default is 0.05).

    :return:
            theta0: A tensor containing the estimated 6D pose (rotation (3D) and translation (3D)) for each batch element.
            P_topk_i: The indices of the top k correspondences selected from P.
            inliers_list: A list of inliers found by RANSAC for each batch.
    """

    # A. Select the top k correspondences based on the confidence values in P
    inds, P_topk_i = torch.topk(P, k=topK, dim=-1, largest=True, sorted=True)

    # B. Set Up Calibration Parameters
    # The intrinsic camera matrix (identity matrix here, assuming a simple normalized camera)
    K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    # Distortion coefficients (set to zero, assuming no lens distortion).
    dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
    # Initializes the pose estimates as a zero tensor.
    theta0 = p3d.new_zeros((p3d.size(0), 6))
    # The last element (corresponding to the z-translation) is initialized to 2.5
    theta0[..., -1] = 2.5

    # C. Loop Over the Batch and Run RANSAC:
    inliers_list = []
    for i in range(p3d.size(0)):

        p2d_np = p2d[i, P_topk_i[i], :].cpu().numpy()
        p3d_np = p3d[i, P_topk_i[i], :].cpu().numpy()

        # Run the P3P Algorithm with RANSAC
        # solvePnPRansac: Estimates the camera pose (rotation rvec and translation tvec) using
        # the P3P algorithm with RANSAC to eliminate outliers.
        # The *iterations* Count is set to 1000, meaning RANSAC will run up to 1000 iterations,
        # The *reprojectionError* is the threshold for classifying a point as an inlier.
        # RANSAC is used to robustly estimate the pose by ignoring correspondences with high reprojection errors.
        # cv2.solvePnPRansac from OpenCV: non-differentiable and not designed to propagate gradients
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(p3d_np, p2d_np, K, dist_coeff,
                                                         iterationsCount=1000,
                                                         reprojectionError=reprojectionError,
                                                         flags=cv2.SOLVEPNP_P3P)

        # Check and Store the Pose
        if rvec is not None and tvec is not None and retval:
            rvec = torch.as_tensor(rvec, dtype=p3d.dtype, device=p3d.device).squeeze(-1)
            tvec = torch.as_tensor(tvec, dtype=p3d.dtype, device=p3d.device).squeeze(-1)
            if torch.isfinite(rvec).all() and torch.isfinite(tvec).all():
                theta0[i, :3] = rvec
                theta0[i, 3:] = tvec
        else:
            print("ransac returned none")

        # Store Inliers
        inliers_list.append(torch.tensor(inliers_list).cuda())

    return theta0, P_topk_i, inliers_list


class PosePredictor(nn.Module):

    def __init__(self,
                 intrinsic_camera_matrix=None,
                 layer_channels=None,
                 transformer=True,
                 end2end=False,
                 robust=True,
                 faces=None,
                 verts=None,
                 is_training=False,
                 num_keypoints=32,
                 image_size=(256, 256),
                 max_num_2d_points=100,
                 ):
        super().__init__()

        keypoint_indices = geom_utils.generate_key_points(verts, num_keypoints)
        self.register_buffer('keypoint_indices', keypoint_indices)
        self.register_buffer('intrinsic_camera_matrix', torch.tensor(list(intrinsic_camera_matrix)))
        self.end2end = end2end
        self.robust = robust
        self.is_training = is_training

        decoder_input_layers = ("layer4", "layer3", "layer2", "layer1",)
        decoder_input_layers_channels = {L: layer_channels[L] for L in decoder_input_layers}

        self.transformer = transformer
        if not transformer:
            self.feature_extractor = LWTLDecoder(in_channels=1,
                                                 out_channels=64,
                                                 ft_channels=decoder_input_layers_channels,
                                                 use_bn=False,
                                                 num_classes=len(self.keypoint_indices))
        else:
            self.feature_extractor = TransPoseR(num_keypoints=len(self.keypoint_indices),
                                                d_model=64,
                                                dim_feedforward=128)

        self.activation = nn.Tanh()

        self.declarative_pnp = NonlinearWeightedOrthBlindPnP()
        self.faces = faces

        self.image_size = image_size
        grid_y, grid_x = torch.meshgrid(torch.arange(image_size[1]), torch.arange(image_size[0]))
        self.coordinates_2d = torch.stack((grid_x.float(), grid_y.float())).cuda()  # (2, image_size[0], image_size[1])
        self.max_num_2d_points = max_num_2d_points
        self.apply_P = False

    def extract_3d_keypoints(self, verts):
        keypoints = verts[:, self.keypoint_indices]
        return keypoints

    def extract_2d_points(self, mask):
        return self.coordinates_2d[:, mask[0] > 0]

    def sample_features(self, feat, points_2d, im_size):
        """

        :param feat: Segmentation mask (B, 1, H, W)
        :param points_2d: 2D points coordinates (B, 1, 2(x,y))
        :param im_size: Image size
        :return:
        """

        coords_x = 2. * points_2d[..., 0] / float(im_size[1]) - 1.0
        coords_y = 2. * points_2d[..., 1] / float(im_size[0]) - 1.0
        coords = torch.stack((coords_x, coords_y), dim=-1).unsqueeze(2)

        assert points_2d.shape[1] == feat.shape[1]

        f_samps = []
        for i in range(points_2d.shape[1]):
            f_samp = nn.functional.grid_sample(feat[:, i].unsqueeze(1), coords[:, i].unsqueeze(1))
            f_samps.append(f_samp)

        return torch.cat(f_samps, dim=1).squeeze(-1).squeeze(-1)

    def downsample(self, X, num_points):

        from torch_cluster import fps

        ratio = float(num_points)/float(X.shape[0])
        if ratio < 1.0:
            index = fps(X, ratio=ratio, random_start=True)
        else:
            index = torch.arange(X.shape[0])

        return index

    def _mask(self, feats):
        """
        This function create the segmentation mask that projects image features to binary mask
        :param feats: Image Features B x F x H x W
        :return: segs: segmentation mask B x 1 x H x W,
        """

        # Create the segmentation mask that projects image features to binary mask B x 1 x H x W
        if self.transformer:
            segs = self.feature_extractor(feats["layer2"])
        else:
            segs = self.feature_extractor(self.image_size, feats)

        if torch.isinf(segs).any():
            print('ERROR: segmentations were inf!!!')
        if torch.isnan(segs).any():
            print('ERROR: segmentations were nan!!!')

        return segs

    def get_confidence_scores(self, coordinates_2d, segs, segs_n):
        """
        Training Mode: It computes a weighted sum of pixel coordinates, applies a Gaussian function to generate heatmaps,
        and computes a confidence score based on the heatmap and segmentation mask.

        Evaluation Mode: It finds the most confident pixel in the segmentation mask and samples features around
        that point for confidence estimation.

        :param coordinates_2d: 2d coordinates (2, image_size[0], image_size[1])
        :param segs: segmentation mask B x 1 x H x W
        :param segs_n: normalized segmentation mask B x 1 x H x W
        :return:
        """

        if self.is_training:

            # (2, H, W) => (H, W, 2) => (1, 1, H, W, 2)
            permuted_coords = coordinates_2d.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)

            # Expand the segmentation mask to match the dimensions of the coordinates
            weighted_coords = permuted_coords * segs_n.unsqueeze(-1)  # Shape: (B, 1, H, W, 2)

            # Sum the weighted coordinates along the spatial dimensions (H, W) to get the aggregated points
            points_2d = weighted_coords.sum(dim=(-2, -3))  # Shape: (B, 1, 2)

            # x (1,1,1, segs.shape[-1]) and y (1,1,segs.shape[-2],1)
            x = torch.arange(0, segs.shape[-1], 1).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda().float()
            y = torch.arange(0, segs.shape[-2], 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).cuda().float()

            # heatmaps have a shape of (B, 1, H, W).
            sigma = 13.5
            heatmaps = torch.exp(- ((x - points_2d[:, :, 0:1, None]) ** 2 + (y - points_2d[:, :, 1:, None]) ** 2) / (2. * sigma ** 2))
            # Probability, Confidence values (B, 1)
            P = (heatmaps * segs).sum(dim=(-1, -2))/(sigma ** 2)

            if torch.isinf(P).any():
                print('ERROR: P was inf!!!')
            if torch.isnan(P).any():
                print('ERROR: P was nan!!!')

            return P, points_2d

        # (B, 1, H, W) => (B, 1, H*W).
        segs_flat = segs.view(*segs.shape[:2], -1)
        # max indices of batch samples: (B, 1)
        maxind = segs_flat.argmax(dim=-1)

        width = segs.shape[-1]
        # column index (or the x-coordinate) in the original 2D (B,1)
        c = maxind.remainder(width)
        # row index (or the y-coordinate) in the 2D (B,1)
        r = maxind // width
        # (B,1) || (B,1) => (B,1,2) = (x,y) coordinate corresponding to maximum segmentation score prediction of batch.
        points_2d = torch.stack((c, r), dim=-1).float()

        # P (B, 1)
        P = self.sample_features(segs_n, points_2d, self.image_size)

        if torch.isinf(P).any():
            print('ERROR: P was inf!!!')
        if torch.isnan(P).any():
            print('ERROR: P was nan!!!')

        return P, points_2d

    def scale_trans_quat(self, camera):
        """
        Convert camera which is 3D-rotation and 3D-translation format to:

        cams_bx7: 3D-translation, 4D quaternion
        cams_bx7_: 1D-scale, 2D-rotation, 4D quaternions

        :param camera: (B,6)
        :return: cams_bx7 (B,7), cams_bx7_ (B,7)
        """

        ang = camera[..., :3].norm(dim=-1)
        quat_pred = geom_utils.axisangle2quat(camera[..., :3], ang)
        cams_bx7 = torch.cat((camera[..., 3:], quat_pred), dim=-1)

        trans_2d = camera[..., 3:-1]
        scale_1d = torch.norm(camera[..., 3:], dim=-1, keepdim=True)
        cams_bx7_ = torch.cat((scale_1d, trans_2d, quat_pred), dim=-1)

        return cams_bx7, cams_bx7_

    def train_heatmap(self, feats):
        """
        Create the segmentation mask (direct output of keypoint pose prediction network)
        that projects image features to binary mask.

        :param feats: Image deep feature from convolution encoder.
        :return: Dictionary of Heatmap prediction from prediction network (self.feature_extractor). (B, 1, H, W)
        """

        segs = self._mask(feats)

        return {"heatmaps": segs}

    def inference(self, feats, verts):
        """
        This function works at inference mode. The ransac_p3p() can not be integrated into a differentiable pipeline
        since ransac is not differentiable. RANSAC itself is not differentiable, and so it cannot be used directly in
        gradient-based optimization or backpropagation.

        :param feats: Image deep feature from convolution encoder.
        :param verts: Shape vertices 3D (B, V, 3)
        :return: A dictionary inclding

        "cameras": the model's estimated cameras based on the predicted heatmap (B,7) = (Batch, [Translation (3D), Quaternions (4D)])
        "cameras0": The robust ransac estimated cameras based on the predicted heatmap (B,7) = (Batch, [Translation (3D), Quaternions (4D)])
        "Ps": P: Confidence scores (B,1)
        "points3d": 3D points coordinates (B, V, 3),
        "points2d": 2D points coordinates.
        "heatmaps": The model's predicted heatmap (B, 1, H, W)

        """

        # Extract 3D keypoint from 3D shape vertices
        points3d = self.extract_3d_keypoints(verts.detach())

        # Create the segmentation mask (direct output of keypoint pose prediction network)
        # that projects image features to binary mask segs (B, 1, H, W)
        segs = self._mask(feats)
        segs = torch.relu(segs)
        segs_n = segs / (segs.sum(dim=(-1, -2), keepdim=True) + 0.000001)

        # Get confidence scores for batch samples and check if it is Inf and NaN
        P, points_2d = self.get_confidence_scores(segs, segs_n, self.coordinates_2d)

        if not self.apply_P:
            # This makes P as a normal distribution with all confidence equal to 1
            P = P / P.sum(dim=-1, keepdims=True)

        points_2d_bearings = 2. * points_2d / self.intrinsic_camera_matrix[:2] - 1.0

        if self.robust:
            # The ransac_p3p() is not differentiable.
            theta0, P_top_i, inliers_list = ransac_p3p(P.detach(),
                                                       points_2d_bearings.detach(),
                                                       points3d,
                                                       topK=P.shape[-1] // 2,
                                                       reprojectionError=0.08)

            # The perspective_to_orthographic_estimation() is differentiable and can be part of a gradient flow
            theta0 = perspective_to_orthographic_estimation(theta0, points3d)

            P_filt = torch.zeros_like(P)
            for i in range(len(inliers_list)):
                inliers = inliers_list[i]
                if len(inliers) > 6:
                    P_filt[i, P_top_i[:, inliers]] = P[:, P_top_i[:, inliers]]
                else:
                    P_filt[i] = P[i]
            P = P_filt
        else:
            theta0 = points3d.new_zeros((points3d.size(0), 6))
            theta0[..., 3] = 1.0

        # The declarative_pnp() is differentiable and can be part of a gradient flow
        # declarative_pnp(): it returns theta where the first 3 elements are rotation and the last 3 are translation.
        # camera = (B: Batch, 6D: [rotation (3D), translation (3D)])
        camera = self.declarative_pnp.forward(P, points_2d_bearings, points3d,
                                              theta0=theta0)  # return rotator vector (angle=norm(x))

        cams_bx7 = self.scale_trans_quat(camera)

        cams_bx7_0 = self.scale_trans_quat(theta0)

        return {"cameras": cams_bx7,
                "cameras0": cams_bx7_0,
                "Ps": P,
                "points3d": points3d,
                "points2d": points_2d,
                "heatmaps": segs
                }

    def forward(self, feats, verts):

        if not self.end2end and self.is_training:

            return self.train_heatmap(feats)

        return self.inference(feats, verts)





