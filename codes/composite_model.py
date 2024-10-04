
import torch
import torchvision
import numpy as np

from networks.netOps import geom_mapping_toolkit as geom_utils
from networks.netOps import mesh

from networks.netOps.nmr_local import SoftRas as SoftRas

from networks.composite_network import CompositeNet

from multiplex import CameraMultiplex
from loss_module import Loss
from networks.netOps import image as image_utils


class CompositeModel(torch.nn.Module):

    def __init__(self, configs):
        # First call the parent's __init__ method
        super(CompositeModel, self).__init__()

        # Load configuration dictionary
        self.cfg = configs

        # Initialize batch statistics, which is updated at the end of every epoch
        self.batch_stats = {}

        # Initialize model predicted output of batch samples
        self.predicted_vertices = None                # (B,V,D:3)
        self.predicted_textures_3D = None             # (B,F,T,T,T,C:3)
        self.predicted_texture_uv_image = None        # (B,F,T,T,T,C:3)

        # Load category-specific mean shape
        mean_shape, verts, faces, verts_uv, faces_uv = self.load_mean_shape(self.cfg.shape.shape_path)

        # Load mean shape components into buffer to not be updated during training but remain part of the model state
        self.register_buffer('verts_uv', verts_uv.float())
        self.register_buffer('faces_uv', faces_uv.float())
        self.register_buffer('verts', verts.float())
        self.register_buffer('faces', faces.long())

        # Initialize UV sampler for 2D to 3D transformation/projection
        uv_texture, uv_sampler_nmr = self.init_uv_sampler(mean_shape,
                                                          self.cfg.texture.textureUnwrapUV,
                                                          self.cfg.texture.tex_size,
                                                          self.cfg.texture.texture_uvshift)
        self.uv_sampler = uv_texture  # .cuda()
        self.uv_sampler_nmr = uv_sampler_nmr  # .cuda()

        # Initialize pretrained model for RGB input images
        self.resnet_transform = torchvision.transforms.Normalize(mean=self.cfg.BGR_MEAN, std=self.cfg.BGR_STD)

        # Initialize rendering function
        self.renderer_mask = SoftRas(self.cfg.renderer.img_size,
                                     perspective=self.cfg.renderer.perspective,
                                     light_intensity_ambient=self.cfg.renderer.mask_light_intensity_ambient,
                                     light_intensity_directionals=self.cfg.renderer.mask_light_intensity_directionals)

        self.renderer_mask.ambient_light_only()
        self.renderer_mask.renderer.set_gamma(self.cfg.renderer.softras_gamma)
        self.renderer_mask.renderer.set_sigma(self.cfg.renderer.softras_sigma)

        # Initialize camera multiplex parameters for optimization.
        self.multiplex = CameraMultiplex(self.cfg.multiplex, self.cfg.dataset_size)

        # Initialize the neural network model used for prediction.
        self.net = CompositeNet(self.cfg.img_size, self.cfg.learning.is_training,
                                self.cfg.net,
                                shape_cfg=self.cfg.shape, texture_cfg=self.cfg.texture, pose_cfg=self.cfg.pose,
                                mean_shape=self.verts, faces=self.faces, verts=self.verts,
                                verts_uv=self.verts_uv, faces_uv=self.faces_uv)

        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()

        # Initialize loss function
        self.loss = Loss(self.verts, self.faces, self.cfg.img_size,
                         self.cfg.renderer.perspective, self.cfg.renderer.softras_gamma, self.cfg.renderer.softras_sigma,
                         self.uv_sampler.shape, self.cfg.loss,)

        if not self.cfg.learning.is_training:
            self.net.eval()

    def to_device(self, tensor):
        # Check if CUDA is available and get the appropriate device (CUDA or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If the tensor is on the CPU and the device is CUDA, apply pin_memory and non_blocking transfer
        if device.type == "cuda":
            # Move the tensor to the GPU with non_blocking=True
            return tensor.pin_memory().cuda(non_blocking=True)
        else:
            # Move the tensor to the CPU if CUDA is not available
            return tensor.to(device)

    def load_mean_shape(self, shape_path):
        """
        This function reload category-specific mean shape and its components
        :param shape_path: Path to the mean shape file.
        :return: None.
        """

        mean_shape = mesh.fetch_mean_shape(shape_path, mean_centre_vertices=True)
        verts = mean_shape['verts']
        faces = mean_shape['faces']
        verts_uv = mean_shape['verts_uv']
        faces_uv = mean_shape['faces_uv']

        verts_uv = torch.from_numpy(verts_uv).float()  # (V,2)
        verts = torch.from_numpy(verts).float()        # (V,3)
        faces = torch.from_numpy(faces).long()         # (F,3)
        faces_uv = torch.from_numpy(faces_uv).float()  # (F,3,2)

        assert (verts_uv.shape[0] == verts.shape[0])
        assert (verts_uv.shape[1] == 2)
        assert (verts.shape[1] == 3)
        assert (faces.shape[1] == 3)
        assert (faces_uv.shape == faces.shape + (2,))

        return mean_shape, verts, faces, verts_uv, faces_uv

    def init_uv_sampler(self, mean_shape, texture_unwrap_uv, texture_size, texture_uv_shift):
        """
        :param mean_shape:
        :param texture_unwrap_uv:
        :param texture_size:
        :param texture_uv_shift:
        :return:
        """

        # Store UV spherical texture map
        verts_sph = geom_utils.convert_uv_to_3d_coordinates(mean_shape['verts_uv'])
        if not texture_unwrap_uv:
            uv_sampler = mesh.compute_uvsampler_softras(verts_sph, mean_shape['faces'],
                                                        tex_size=texture_size,
                                                        shift_uv=texture_uv_shift)
        else:
            uv_sampler = mesh.compute_uvsampler_softras_unwrapUV(mean_shape['faces_uv'], mean_shape['faces'],
                                                                 tex_size=texture_size,
                                                                 shift_uv=texture_uv_shift)
        uv_texture = image_utils.uv2bgr(uv_sampler)  # F,T,T,3
        uv_texture = np.repeat(uv_texture[:, :, :, None, :], texture_size, axis=3)  # F,T,T,T,2
        uv_texture = torch.tensor(uv_texture).float() / 255.

        if not texture_unwrap_uv:
            uv_sampler_nmr = mesh.compute_uvsampler(verts_sph, mean_shape['faces'],
                                                    tex_size=texture_size,
                                                    shift_uv=texture_uv_shift)
        else:
            uv_sampler_nmr = mesh.compute_uvsampler_unwrapUV(mean_shape['faces_uv'], mean_shape['faces'],
                                                             tex_size=texture_size,
                                                             shift_uv=texture_uv_shift)
        uv_sampler_nmr = torch.tensor(uv_sampler_nmr).float()

        return uv_texture, uv_sampler_nmr

    def create_input(self, batch):
        """
        Create input dictionary including camera parameters of the batch samples.
        Batch contents used in this function:
        batch['ids']: index of the dataset sample (B, 1).
        batch['img']: RGB image (B, C, H, W).
        batch['mask']: mask image (B, H, W).
        batch['mask_dt']: mask dt image (B, H, W).
        batch['sfm_pose']: Ground-truth camera pose (SfM pose) (B, 7)

        :param batch: Batch.
        :return: Input dictionary.
        """

        frame_id, img, mask, mask_dt, gt_cam_pose = map(batch.get, ['ids', 'img', 'mask', 'mask_dt', 'sfm_pose'])
        gt_cam_pose = gt_cam_pose.float()

        # *********** Check shapes of batch items
        B, H, W = mask.shape
        assert (img.shape[0] == B)
        assert (img.shape[2:] == (H, W))
        input_B = B
        input_H = H
        input_W = W

        # ********** Deep features
        img = img.float()
        mask = mask.float()
        mask_dt = mask_dt.float()

        input_img = img.clone()
        for b in range(input_img.size(0)):
            input_img[b] = self.resnet_transform(input_img[b])

        frame_ids = frame_id
        img = self.to_device(img)                                 # (B, C, H, W)
        input_img = self.to_device(input_img)                     # (B, C, H, W)
        mask = self.to_device(mask)                               # (B, H, W)
        mask_dt = self.to_device(mask_dt)                         # (B, H, W)
        gt_camera_pose = self.to_device(gt_cam_pose)              # (B, 7)

        return {"batch_size": input_B, "image_height": input_H, "image_width": input_W,
                "frame_ids": frame_ids, "img": img, "input_img": input_img, "mask": mask,
                "mask_dt": mask_dt, "gt_camera_pose": gt_camera_pose}

    def expand_for_multipose(self, data_dict, P):
        """
        This function expands the input attributes to fit for the number of multiple poses to facilitate computations.
        :param data_dict: Input data dictionary.
        :param P: Number of multiple poses per instance (e.g., 5x8=40)
        :return:
        """

        B, H, W = data_dict["mask"].shape
        assert (data_dict["input_img"].shape[0] == B)
        assert (data_dict["input_img"].shape[2:] == (H, W))

        # Expand mask for multi_pose
        orig_img_batch = data_dict["img"][:, None].repeat(1, P, 1, 1, 1)    # (B, P, C, H, W)
        mask = data_dict["mask"][:, None, :, :].repeat(1, P, 1, 1)          # (B, P, H, W)
        mask = mask.view(B * P, H, W)                                       # (B*P, H, W)
        mask_dt = data_dict["mask_dt"][:, None, :, :].repeat(1, P, 1, 1)    # (B, P, H, W)
        mask_dt = mask_dt.view(B * P, H, W)                                 # (B*P, H, W)

        return {"num_multipose": P,
                "num_batch_pose": B * P,
                "orig_img_batch": orig_img_batch,
                "mask": mask,
                "mask_dt": mask_dt
                }

    def rendering(self, num_batch_pose, vertices=None, textures=None, camera=None, texture_flipCam=False):
        """
        This function render the mask and textures.
        It gives a "2D representations" of the model’s texture as viewed from the given "camera parameters".

        :param num_batch_pose: Batch_size x Number of poses.
        :param vertices: Vertices.
        :param textures: Textures.
        :param camera: Camera parameters.
        :param texture_flipCam: If predict textures using the flip camera parameters.
        :return: Rendered masks (B*P, H, W) and textures (B*P, F, T, T, T, C).
        """

        if vertices is None:
            return None, None, None

        # (B, P, 7) => (B*P, 7)
        camera = camera.view((num_batch_pose,) + camera.shape[2:])
        assert (torch.isfinite(camera).all())

        faces_batch = self.faces[None, :, :].expand(vertices.shape[0], -1, -1).to(vertices.device)
        rend_texture, rend_mask = self.renderer_mask.forward(vertices, faces_batch.int(), camera, textures=textures)

        if textures is None:
            return rend_mask, None, None

        if not texture_flipCam:
            return rend_mask, rend_texture, None

        camera_reflected = geom_utils.reflect_cam_pose(camera)
        rend_texture_flip, _ = self.renderer_mask.forward(vertices, faces_batch.int(), camera_reflected,
                                                          textures=textures)
        return rend_mask, rend_texture, rend_texture_flip

    def get_params(self):
        """Retrieve model parameters from the underlying network."""
        # Access the underlying model if self.net is a DataParallel instance
        if isinstance(self.net, torch.nn.DataParallel):
            return self.net.module.get_params()
        return self.net.get_params()

    def forward(self, batch):

        # Get input data from batch
        data_dict = self.create_input(batch)

        # Get the number of camera poses sampled for each batch instance (frame)
        P = self.cfg.multiplex.num_multipose

        # Expand for multiple poses and get number of batch x number of poses
        data_expanded = self.expand_for_multipose(data_dict, P)

        # ************************** (A) PREDICTION ***************************** #
        # The neural networks model predicts
        # 1- deformation (delta_v) (B: Batch, V: Num Vertices, D: 3D),
        # 2- textures, (B: Batch, F: Num Faces, T: Texture Size, T: Texture Size, C: Channel)
        # 3- Pose [(7D: scale (1D), translation (2D), quaternions (4D))] or a dict if keypoint used.
        shape_output, texture_output, pose_output = self.net(data_dict["input_img"])

        # *********** Pose Prediction: Prediction of camera pose or heatmaps.
        pred_cam = pose_output['cameras'] if self.cfg.pose.pred_pose_keypoint and not self.cfg.learning.is_training else pose_output
        keypoint_indices = self.net.poseNet.posePred.keypoint_indices if self.cfg.pose.pred_pose_keypoint else None

        # *********** Shape Prediction: Prediction of deformation (delta_v).
        delta_v_expanded, pred_v_expanded, mean_shape_learned = None, None, None
        if self.cfg.net.pred_shape:
            delta_v = shape_output["shape"]    # (B,V,3)
            delta_v_mp = delta_v[:, None, ...].repeat(1, P, 1, 1)  # (B,P,V,3)
            delta_v_expanded = delta_v_mp.view((data_expanded["num_batch_pose"],) + delta_v_mp.shape[2:])  # (B*P,V,3)

            # Apply predicted deformation (delta_v) to the learned mean shape to gain final shape from vertices (pred_v)
            mean_shape_learned = shape_output["mean_shape_learned"]
            pred_v_expanded = mean_shape_learned[None, :, :] + delta_v_expanded   # (B*P,V,3)

            # Sanity Checks
            assert (torch.isfinite(delta_v_expanded).all())
            assert (torch.isfinite(mean_shape_learned).all())
            assert (torch.isfinite(pred_v_expanded).all())

            self.predicted_vertices = mean_shape_learned[None, :, :] + delta_v

        # *********** Texture Prediction: texture_flow, textures
        textures_3D_expanded = None
        if self.cfg.net.pred_texture:
            texture = texture_output["texture"]
            txt_size = texture.size(2)  # T: texture resolution

            # (B, F, T, T, C) => (B, F, T, T, 1, C) => (B, F, T, T, T, C)
            textures_3D = texture.unsqueeze(4).expand(-1, -1, -1, -1, txt_size, -1)

            # (B, F, T, T, T, C) => (B, 1, F, T, T, T, C) => (B, P, F, T, T, T, C)
            textures_3D_mp = textures_3D[:, None, ...].repeat(1, P, 1, 1, 1, 1, 1)

            # (NUM_BATCH_POSE=B*P, F, T, T, T, C)
            textures_3D_expanded = textures_3D_mp.view((data_expanded["num_batch_pose"],) + textures_3D_mp.shape[2:])

            # Detach tensor to not be part of the computation graph, to not track operations for gradient computation.
            self.predicted_textures_3D = textures_3D.detach()
            self.predicted_texture_uv_image = texture_output["uv_image_pred"]

        # *********** Camera Pose Prediction: Select supervision approach.
        if self.cfg.multiplex.use_gt_camera:
            # No multiplex optimization: Supervised using gt-camera-pose (SfM or best pose of multiplex)
            assert (P == 1)
            # (B, 7) => (B, 1, 7)
            pred_cam_mp = data_dict["gt_camera_pose"][:, None, :]

        else:
            # Unsupervised while optimizing camera multiplex parameters
            # (B, P, 7) & (B, P)
            pred_cam_mp, cam_scores = self.multiplex(data_dict["frame_ids"],
                                                     data_dict["gt_camera_pose"][..., 0:3])

        # ************************** (B) RENDERING ***************************** #
        rend_mask, rend_texture, rend_texture_flip = self.rendering(data_expanded["num_batch_pose"],
                                                                    vertices=pred_v_expanded,
                                                                    textures=textures_3D_expanded,
                                                                    camera=pred_cam_mp,
                                                                    texture_flipCam=self.cfg.renderer.texture_flipCam)

        import matplotlib.pyplot as plt
        # Plot the result
        plt.figure()
        plt.imshow(rend_mask[0].detach().cpu().numpy())
        plt.figure()
        plt.imshow(rend_texture[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        # ************************** (C) LOSS COMPUTATION ***************************** #
        loss_dict = self.loss.update_loss(self.cfg.loss, P,
                                          data_dict["batch_size"],
                                          data_dict["gt_camera_pose"],

                                          data_expanded["orig_img_batch"],
                                          data_expanded["num_batch_pose"],
                                          data_expanded["mask"],
                                          data_expanded["mask_dt"],

                                          pose_output, pred_cam_mp, pred_cam, keypoint_indices,
                                          rend_mask, rend_texture, rend_texture_flip,
                                          delta_v_expanded, pred_v_expanded, mean_shape_learned,

                                          device=data_dict["input_img"].device)

        # ********************* (D) UPDATE/SAVE CAMERA POSE DICTIONARY OF MULTIPLEX ************************* #
        self.multiplex.update_camera_multiplex_batch(data_dict["frame_ids"].squeeze(1),
                                                     pred_cam_mp,
                                                     loss_dict["quat_score_multiplex"],
                                                     data_dict["gt_camera_pose"][..., 0:3],
                                                     loss_dict["mask_iou_multiplex"])

        # ************************** (D) UPDATE BATCH STATISTICS ***************************** #
        self.batch_stats['input_img'].append(data_dict['input_img'].detach().cpu())
        self.batch_stats['frame_id'].append(data_dict['frame_id'].detach().cpu())
        self.batch_stats['camera_gt'].append(data_dict["gt_camera_pose"].detach().cpu())

        self.batch_stats['mean_shape_learned'].append(mean_shape_learned.detach().cpu())
        self.batch_stats['shape'].append(self.predicted_vertices.detach().cpu())
        self.batch_stats['texture_3D'].append(self.predicted_textures_3D.detach().cpu())
        self.batch_stats['texture_uv'].append(self.predicted_texture_uv_image.detach().cpu())
        self.batch_stats['camera_network'].append(pred_cam.detach().cpu())
        self.batch_stats['cameras_multiplex'].append(pred_cam_mp.detach().cpu())

        self.batch_stats['quat_score_multiplex'].append(loss_dict['quat_score_multiplex'].detach().cpu())
        self.batch_stats['quat_error_multiplex'].append(loss_dict['quat_error_multiplex'].detach().cpu())
        self.batch_stats['quat_error_network'].append(loss_dict['quat_error_network'].detach().cpu())
        self.batch_stats['mask_iou_multiplex'].append(loss_dict['mask_iou_multiplex'].detach().cpu())

        return loss_dict


















