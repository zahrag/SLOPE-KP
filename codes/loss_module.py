
import numpy as np
import torch

from networks.netOps import geom_mapping_toolkit as geom_utils
from networks.netOps import loss_metrics as loss_utils


class Loss(torch.nn.Module):
    def __init__(self, verts, faces, img_size, perspective,
                 softras_gamma, softras_sigma, uv_sampler_shape, configs_loss):
        super(Loss, self).__init__()  # Call to the parent class constructor

        # Conditional boolean parameters to control loss computation; facilitates various processing and save resources.
        self.apply_texture_loss = configs_loss.apply_texture_loss
        self.apply_mask_loss = configs_loss.apply_mask_loss
        self.apply_shape_loss = configs_loss.apply_shape_loss
        self.apply_pose_loss = configs_loss.apply_pose_loss
        self.apply_heatmap_loss = configs_loss.apply_heatmap_loss
        self.apply_quat_scores = configs_loss.apply_quat_scores
        self.apply_multiplex_loss = configs_loss.apply_multiplex_loss
        self.apply_balanced_multiplex_loss = configs_loss.apply_balanced_multiplex_loss
        self.apply_multiplex_quat_error = configs_loss.apply_multiplex_quat_error
        self.apply_network_quat_error = configs_loss.apply_network_quat_error

        # Initialize the criterion/metric function
        # For mask
        self.rend_mask_loss_fn = loss_utils.mask_l2_dt_loss
        self.rend_mask_iou = loss_utils.maskiou

        # For shape
        self.deform_loss_fn = loss_utils.deform_l2reg
        self.laplacian_loss_fn = loss_utils.LaplacianLoss(faces.long(), verts.detach())
        self.meanV_laplacian_loss_fn = loss_utils.LaplacianLoss(faces.long(), verts.detach())
        self.deltaV_laplacian_loss_fn = loss_utils.LaplacianLoss(faces.long(), verts.detach())
        self.graphlap_loss_fn = loss_utils.GraphLaplacianLoss(faces.long(), verts.shape[0])
        self.edge_loss_fn = loss_utils.EdgeLoss(verts.detach(), faces.long())

        # For camera
        self.camera_loss = loss_utils.camera_loss

        # For quaternion error
        self.quat_inverse = geom_utils.quat_inverse
        self.hamilton_product = geom_utils.hamilton_product
        self.quat2axisangle = geom_utils.quat2axisangle

        # For texture
        self.add_module('texture_loss_fn', loss_utils.PerceptualTextureLoss(use_gpu=torch.cuda.is_available()))

        # For Keypoints
        self.heatmap_loss = loss_utils.HeatmapLoss(faces, img_size,
                                                   perspective, softras_gamma, softras_sigma, uv_sampler_shape)

        # Move the entire model (including submodules) to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def multiplex_quat_error(self, cam_poses, gt_camera_pose, input_b, num_multipose):
        """
        This function computes the Quaternions error in two cases:

        Camera Quaternion Error of the multiplex (cam_pred):
        This is insightful only if we optimize camera multiplex parameters during this phase.


        :param cam_poses: Cameras optimized tensor for multiplex. (B, P, 7)
        :param gt_camera_pose: Ground truth camera pose (B, 7)
        :param input_b: Input batch size (B)
        :param num_multipose: Multiple poses number (P)

        :return: quat_error: Quaternion error tensor of the multiplex (cam_pred). (B, P, 1)

        """
        if not self.apply_multiplex_quat_error:
            return torch.zeros(input_b, num_multipose, device=cam_poses.device, dtype=torch.float)

        cam_pred = cam_poses.view((input_b * num_multipose,) + cam_poses.shape[2:])
        quat = cam_pred[:, 3:7].view(input_b, num_multipose, 4)
        quat_gt = gt_camera_pose[:, None, 3:7]
        quat_error_mp = self.compute_quat_error(quat, quat_gt)

        return quat_error_mp

    def network_quat_error(self, pred_cam, gt_camera_pose, input_b):
        """
        This function computes the Quaternions error in two cases:

        Camera Quaternion Error of the model (pred_cam):
        This is possible only if we predict camera by model in this phase.

        :param pred_cam: Camera predicted tensor by NN model. (B, 7)
        :param gt_camera_pose: Ground truth camera pose (B, 7)
        :param input_b: Input batch size (B)

        :return:  network_quat_error: Quaternion error tensor of the model (pred_cam). (B, 1)
        """

        if not self.apply_network_quat_error or pred_cam is None:
            return torch.zeros(input_b, 1, device=gt_camera_pose.device, dtype=torch.float)

        quat = pred_cam[:, 3:7]
        quat_gt = gt_camera_pose[:, 3:7]
        network_quat_error = self.compute_quat_error(quat, quat_gt)

        return network_quat_error

    def compute_quat_error(self, quat, quat_gt):
        """
        This function calculates the camera quaternion error.
        :param quat_gt: Ground truth camera quaternions. (B, P, 4)
        :param quat: Camera quaternion. (B, 1, 4)
        :return: Quaternion error. (B, 1)
        """

        # Calculate quaternion error
        quat_gt_inv = self.quat_inverse(quat_gt)
        quat_error = self.hamilton_product(quat, quat_gt_inv)
        _, quat_error = self.quat2axisangle(quat_error)
        quat_error = torch.min(quat_error, 2 * np.pi - quat_error)
        quat_error_deg = quat_error * 180 / np.pi

        return quat_error_deg

    def get_texture_loss(self, image_batch, rend_mask, rend_texture, rend_texture_flip, input_b, num_multi_pose, n_bp,
                         mask, device, texture_flipCam=False):

        """
        B: Batch size, P: Number of multi pose, C: Channel (e.g., 3), H: Height, W: Width
        F: Faces of mean shape, T: Texture resolution.

        :param image_batch: Original image batch tensor. (B, P, C, H, W)
        :param rend_mask: Rendered mask tensor. (B*P, H, W)
        :param rend_texture: Rendered texture tensor. (B*P, F, T, T, T, C)
        :param rend_texture_flip: Rendered texture flipped tensor. (B*P, F, T, T, T, C)
        :param input_b: Input batch size (B)
        :param num_multi_pose: Multiple poses number (P)
        :param n_bp: Number of batches x Number of poses. (B*P)
        :param mask: Ground-truth mask tensor. (B*P, H, W)
        :param device: Device
        :param texture_flipCam: If flip camera. Boolean.

        :return: Texture loss tensor. (B, P)
        """

        # Return zeros if texture loss is not applied or necessary inputs are missing
        if not self.apply_texture_loss or rend_mask is None or rend_texture is None:
            return torch.zeros((input_b, num_multi_pose), dtype=torch.float32, device=device)

        image_batch = image_batch.view((n_bp,) + image_batch.shape[2:])
        texture_loss_mp = self.texture_loss_fn(rend_texture, image_batch, rend_mask, mask)

        if texture_flipCam:
            imgs_batch_flip = torch.flip(image_batch, [-1])
            mask_flip = torch.flip(mask, [-1])
            texture_flip_loss_mp = self.texture_loss_fn(rend_texture_flip, imgs_batch_flip, None, mask_flip)
            texture_loss_mp = (self.texture_loss_mp + texture_flip_loss_mp) / 2  # TODO: FIX

        # Texture loss for multiplex we get a loss per camera pose
        texture_loss_mp = texture_loss_mp.view(input_b, num_multi_pose)

        return texture_loss_mp

    def get_mask_loss(self, rend_mask, mask, mask_dt, input_b, num_multi_pose):
        """
        :param rend_mask: Rendered mask tensor. (B*P, H, W)
        :param mask: Ground-truth mask tensor. (B*P, H, W)
        :param mask_dt: Ground-truth masks dt tensor. (B*P, H, W)
        :param input_b: Input batch size (B)
        :param num_multi_pose: Multiple poses number (P)
        :return: Mask loss tensor. (B, P)
        """

        if not self.apply_mask_loss or rend_mask is None:
            return (torch.zeros((input_b, num_multi_pose), dtype=torch.float32, device=input_b.device),
                    torch.zeros((input_b, num_multi_pose), dtype=torch.float32, device=input_b.device))

        mask_loss_mp = self.rend_mask_loss_fn(rend_mask,
                                              mask, reduction='none',
                                              mask2_dt=mask_dt).mean(dim=(-1, -2)).view(input_b, num_multi_pose)

        mask_iou_mp = self.rend_mask_iou(rend_mask, mask).view(input_b, num_multi_pose)

        return mask_loss_mp, mask_iou_mp

    def get_total_loss_mp(self, mask_loss_mp, texture_loss_mp, mask_loss_wt, texture_loss_wt, balanced=False):
        """
        :param mask_loss_mp:
        :param texture_loss_mp:
        :param mask_loss_wt:
        :param texture_loss_wt:
        :param balanced:
        :return: Total loss tensor of camera multiplex. (B, P)
        """

        if not self.apply_multiplex_loss:
            return torch.zeros_like(mask_loss_mp)

        if self.apply_balanced_multiplex_loss:
            return self.rescaled_weighte_loss(mask_loss_mp, texture_loss_mp)
        else:
            return self.weighted_sum_loss(mask_loss_mp, mask_loss_wt, texture_loss_mp, texture_loss_wt)

    def rescaled_weighte_loss(self, mask_loss_mp, texture_loss_mp):
        """
        This function computes camera multiplex loss for both masks and textures rendered for multiple camera poses.
        Total multiplex loss is weighted of the normalized losses that fit within the range [0, 1].
        This function balances losses more effectively when they have different scales.

        Use this loss on mask loss and texture loss if:

        1- The scales of the two losses are different, and you want them to contribute more equally.
        2- You want to ensure that neither loss dominates the other by rescaling them between 0 and 1.
        3- You want greater control over how losses are combined, with equal contributions regardless of their original magnitude.

        :param mask_loss_mp: masks loss tensor of the multiplex (cam_pred). (B, P)
        :param texture_loss_mp: textures loss tensor of the multiplex (cam_pred). (B, P)
        :return: Total multiplex loss (B, P)
        """

        mask_weight = 0.5
        texture_weight = 0.5

        # silhouette loss with scaling
        _mask_loss_min, _ = mask_loss_mp.min(dim=1, keepdim=True)
        _mask_loss_max, _ = mask_loss_mp.max(dim=1, keepdim=True)
        _mask_loss_rescaled = (mask_loss_mp - _mask_loss_min) / (_mask_loss_max - _mask_loss_min + 1e-4)

        # image reconstruction texture loss with scaling
        _texture_loss_min, _ = texture_loss_mp.min(dim=1, keepdim=True)
        _texture_loss_max, _ = texture_loss_mp.max(dim=1, keepdim=True)
        _texture_loss_rescaled = (texture_loss_mp - _texture_loss_min) / (
                _texture_loss_max - _texture_loss_min + 1e-4)

        # weighted & scaled silhouette loss + weighted & scaled image reconstruction loss
        total_loss_mp = mask_weight * _mask_loss_rescaled + texture_weight * _texture_loss_rescaled

        return total_loss_mp

    def weighted_sum_loss(self, mask_loss_mp, mask_loss_wt, texture_loss_mp, texture_loss_wt):
        """
        Both losses (mask_loss_mp and texture_loss_mp) are already on similar scales.

        Use this loss if:

        1- You are confident that the predefined weights (mask_loss_wt and texture_loss_wt)
        will appropriately balance the contributions from both losses.
        2- You prefer a simpler and computationally less expensive approach without additional rescaling.
        3- Simplicity and faster execution.

        :param mask_loss_mp: masks loss tensor of the multiplex (cam_pred). (B, P)
        :param mask_loss_wt: Rendered mask loss weights (set in configs). (1,)
        :param texture_loss_mp: textures loss tensor of the multiplex (cam_pred). (B, P)
        :param texture_loss_wt: Texture loss weights (set in configs). (1,)
        :return: Total multiplex loss (B, P)
        """
        camera_loss_mp = mask_loss_wt * mask_loss_mp.mean() + texture_loss_wt * texture_loss_mp.mean()

        return camera_loss_mp

    def get_camera_quat_scores(self, total_loss_mp, quatScorePeakiness):
        """
        This function computes camera quaternion scores using the multiplex total loss and
        a Softmin Function, which computes scores that emphasize lower values by assigning them higher probabilities.
        It’s the inverse of the softmax function, which highlights higher values.

        :param total_loss_mp: Total camera multiplex loss tensor. (B, P)
        :param quatScorePeakiness: The rescaled loss is multiplied by quatScorePeakiness, a hyperparameter
        that controls how "peaked" the distribution of the softmin output is.
        Higher values of quatScorePeakiness make the softmin output more focused on the smallest values (i.e., losses).
        :return: Quaternion scores tensor. (B, P)
        """

        if not self.apply_quat_scores:
            return torch.ones_like(total_loss_mp)

        cam_pose_loss = total_loss_mp.detach()
        loss_min, _ = cam_pose_loss.min(dim=1)
        loss_max, _ = cam_pose_loss.max(dim=1)
        # Normalized loss that fits within the range [0, 1]
        loss_rescaled = (cam_pose_loss - loss_min[:, None]) / (loss_max[:, None] - loss_min[:, None] + 1e-6)
        quat_score = torch.nn.functional.softmin(loss_rescaled * quatScorePeakiness, dim=1)

        return quat_score

    def pose_loss(self, pred_cam, cam_poses, quat_score, device):
        """
        This is the loss computed for the camera neural network prediction model, using
        the best camera pose from the multiplex as the supervision signal.

        This loss enables unsupervised learning by using the best camera pose from the multiplex, determined by
        quaternion scores, as a supervision signal. The camera multiplex itself is integrated into the
        training and optimization process.

        :param pred_cam: Predicted camera as neural network output. (B, 7)
        :param cam_poses: Camera poses from multiple. (B, P, 7)
        :param quat_score: Quaternion scores. (B, P)
        :param device:

        :return:
        """

        if not self.apply_pose_loss:
            return torch.tensor(0., device=device)

        _mm, _ii = quat_score.max(dim=1)
        _rr = torch.arange(_ii.shape[0], dtype=_ii.dtype, device=_ii.device)
        _gt_cam_bx7 = cam_poses[_rr, _ii, :]
        camera_loss = self.camera_loss(pred_cam, _gt_cam_bx7, 0)

        return camera_loss

    def shape_loss(self, delta_v, pred_v, mean_shape,
                   laplacianDeltaV, laplacian_loss_wt, meanV_laplacian_loss_wt, deltaV_laplacian_loss_wt,
                   graphlap_loss_wt, edge_loss_wt, deform_loss_wt, device):

        """
        This function computes shape loss.
        :param delta_v: Shape deformation tensor. (B*P, V, 3)
        :param pred_v: Vertices tensor. (B*P, V, 3)
        :param mean_shape: Category-specific mean shape tensor. (B*P, V, 3)
        :param laplacianDeltaV:
        :param laplacian_loss_wt:
        :param meanV_laplacian_loss_wt:
        :param deltaV_laplacian_loss_wt:
        :param graphlap_loss_wt:
        :param edge_loss_wt:
        :param deform_loss_wt:
        :param device: Device
        :return:
        """

        if not self.apply_shape_loss or delta_v is None or pred_v is None:
            return {
                "edge_loss": torch.tensor(0., device=device),
                "laplacian_loss": torch.tensor(0., device=device),
                "meanV_laplacian_loss": torch.tensor(0., device=device),
                "deltaV_laplacian_loss": torch.tensor(0., device=device),
                "graphlap_loss": torch.tensor(0., device=device),
                "deform_loss": torch.tensor(0., device=device),
                "total_shape_loss": torch.tensor(0., device=device),
            }

        _zero = torch.tensor(0., device=device)

        deform_loss = self.deform_loss_fn(delta_v)

        laplacian_loss = self.laplacian_loss_fn(delta_v if laplacianDeltaV else pred_v) if laplacian_loss_wt > 0 else _zero

        meanV_laplacian_loss = self.meanV_laplacian_loss_fn(mean_shape[None, :, :]) if meanV_laplacian_loss_wt > 0 else _zero

        deltaV_laplacian_loss = self.deltaV_laplacian_loss_fn(delta_v) if deltaV_laplacian_loss_wt > 0 else _zero

        graphlap_loss = self.graphlap_loss_fn(pred_v) if graphlap_loss_wt > 0 else _zero

        edge_loss = self.edge_loss_fn(pred_v) if edge_loss_wt > 0 else _zero

        # Shape Priors
        total_loss = 0
        total_loss += edge_loss_wt * edge_loss
        total_loss += laplacian_loss_wt * laplacian_loss
        total_loss += meanV_laplacian_loss_wt * meanV_laplacian_loss
        total_loss += deltaV_laplacian_loss_wt * deltaV_laplacian_loss
        total_loss += graphlap_loss_wt * graphlap_loss
        total_loss += deform_loss_wt * deform_loss

        return {
            "edge_loss": edge_loss,
            "laplacian_loss": laplacian_loss,
            "meanV_laplacian_loss": meanV_laplacian_loss,
            "deltaV_laplacian_loss": deltaV_laplacian_loss,
            "graphlap_loss": graphlap_loss,
            "deform_loss": deform_loss,
            "total_shape_loss": total_loss,
        }

    def keypoint_loss(self, pred_pose, pred_vertices, gt_camera_pose, keypoint_indices):

        """
        heatmap: (B, nk, W, H) = (128, 32, 256, 256)
        pred_v(predicted vertices): (B, V, XYZ) = (128, num verts = 555, 3)
        faces: (F, 3) = (num faces = 1106, 3): Faces are triangles connecting 3 vertices.

        :param pred_pose: Predicted output from Keypoints Prediction Network
        :param pred_vertices: Predicted vertices.
        :param gt_camera_pose: Ground truth camera pose (SfM, ot the best pose of camera multiplex obtained from Phase-II).
        :param keypoint_indices: List of keypoint indices.
        :return:
        """

        if not self.apply_heatmap_loss:
            return {"heatmap_loss": torch.tensor(0., device=pred_pose.device),
                    "label_texture": torch.zeros((1, 1, 1, 1, 1, 1)),
                    "heatmap_gt": torch.zeros((1, 1, 1, 1)),
                    }

        # Predicted heatmaps from the keypoints prediction network
        pred_heatmaps = pred_pose["heatmaps"]

        # Cameras predicted by the pose prediction network (end-to-end training)
        # cams_bx7 = pred_pose["cameras"].detach()

        # Calculate heatmap loss
        heatmap_loss, heatmap_gt, label_texture = self.heatmap_loss.forward(pred_heatmaps,
                                                                            pred_vertices,
                                                                            gt_camera_pose,
                                                                            keypoint_indices)

        return {"heatmap_loss": heatmap_loss,
                "label_texture": label_texture,
                "heatmap_gt": heatmap_gt
                }

    def update_loss(self, configs, num_multi_pose,
                    input_b, gt_camera_pose,
                    orig_img_batch, n_bp, mask, mask_dt,
                    pose_net_output, cam_poses, pred_cam, keypoint_indices,
                    rend_mask, rend_texture, rend_texture_flip,
                    delta_v, pred_v, mean_shape,
                    device=None):

        """
        :param configs: Configurations.
        :param num_multi_pose: Number of poses. (P)
        :param input_b: Input batch size. (B)
        :param gt_camera_pose: ground-truth camera poses. (B, 7)
        :param orig_img_batch: Original image batch. (B, P, C, H, W)
        :param n_bp: Batch size x number of poses.
        :param mask: Ground-truth mask. (B*p, H, W)
        :param mask_dt: Ground-truth mask dt. (B*p, H, W)

        :param pose_net_output: Pose network output; either a camera (=pred_cam) or a dict (if keypoint pose prediction)
        :param cam_poses: Multiplex camera poses from multiple. (B, P, 7)
        :param pred_cam: Pose network predicted camera as neural network output. (B, 7)
        :param keypoint_indices: List of keypoint indices.

        :param rend_mask: rendered mask. (B*p, H, W)
        :param rend_texture: Rendered texture. (B*p, F, T, T, T, C)
        :param rend_texture_flip: Flipped rendered texture. (B*p, F, T, T, T, C)
        :param delta_v: Deformation tensor. (B*P, V, 3)
        :param pred_v: Predicted shape tensor. (B*P, V, 3)
        :param mean_shape: Category-specific mean shape tensor. (B*P, V, 3)
        :param device

        :return: Loss dictionary.
        Note: _multiplex suffix of loss component refers to when the neural network prediction model's direct output
        is not given to the loss function, but instead it is given to rendering function with input camera pose to obtain
        the rendered mask or texture in the input camera perspective.
        """

        # QUATERION ERROR (multiplex (B,P))
        quat_error_mp = self.multiplex_quat_error(cam_poses, gt_camera_pose, input_b, num_multi_pose)

        # QUATERION ERROR (Pose network (B,))
        quat_error_net = self.network_quat_error(pred_cam, gt_camera_pose, input_b)

        # TEXTURE LOSS (multiplex) (B,P)
        texture_loss_mp = self.get_texture_loss(orig_img_batch, rend_mask, rend_texture, rend_texture_flip,
                                                input_b, num_multi_pose, n_bp, mask, device,
                                                texture_flipCam=configs.texture_flipCam)

        # MASK LOSS (multiplex): (B,P)
        mask_loss_mp, mask_iou_mp = self.get_mask_loss(rend_mask, mask, mask_dt, input_b, num_multi_pose)

        # TOTAL LOSS (multiplex): (B,P)
        total_loss_mp = self.get_total_loss_mp(mask_loss_mp, texture_loss_mp,
                                               configs.mask_loss_wt, configs.texture_loss_wt)

        # QUATERION SCORES (multiplex)
        quat_score_mp = self.get_camera_quat_scores(total_loss_mp, configs.quatScorePeakiness)

        # SHAPE LOSS
        shape_losses = self.shape_loss(delta_v, pred_v, mean_shape,
                                       configs.laplacianDeltaV,
                                       configs.laplacian_loss_wt,
                                       configs.meanV_laplacian_loss_wt,
                                       configs.deltaV_laplacian_loss_wt,
                                       configs.graphlap_loss_wt,
                                       configs.edge_loss_wt,
                                       configs.deform_loss_wt,
                                       device)

        # POSE LOSS
        camera_loss = self.pose_loss(pred_cam, cam_poses, quat_score_mp, device)

        # POSE LOSS (KEYPOINT LOSS) (HEATMAPS)
        kp_loss = self.keypoint_loss(pose_net_output, pred_v, gt_camera_pose, keypoint_indices)

        # TOTAL LOSS
        total_loss = shape_losses["total_shape_loss"]
        total_loss += configs.mask_loss_wt * (mask_loss_mp * quat_score_mp).sum(dim=1).mean()
        total_loss += configs.texture_loss_wt * (texture_loss_mp * quat_score_mp).sum(dim=1).mean()
        total_loss += configs.camera_loss_wt * camera_loss
        total_loss += kp_loss["heatmap_loss"]

        return {
            "camera_loss": camera_loss,
            "shape_losses": shape_losses,
            "quat_error_network": quat_error_net,
            "quat_error_multiplex": quat_error_mp,
            "quat_score_multiplex": quat_score_mp,
            "total_loss_multiplex": total_loss_mp,
            "mask_loss_multiplex": mask_loss_mp,
            "mask_iou_multiplex": mask_iou_mp,
            "texture_loss_multiplex": texture_loss_mp,
            "keypoint_loss": kp_loss,

            "total_loss": total_loss,

        }













