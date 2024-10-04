import numpy as np
import torch
import torchvision
from tqdm import tqdm
import sys
import os

from networks.netOps import geom_mapping_toolkit as geom_utils
from networks.netOps import loss_metrics as loss_utils
from networks.netOps.geom_quaternion_toolkit import EulerQuat, GramSchmidtQuat, SVDQuat

from networks.netOps.nmr_local import NeuralRenderer_pytorch as NeuralRenderer
from networks.netOps.nmr_local import SoftRas as SoftRas
from networks.netOps import mesh


class MultiplexOptimizer(torch.nn.Module):
    def __init__(self, configs):
        # First call the parent's __init__ method
        super(MultiplexOptimizer, self).__init__()

        self.cfg = configs

        # Load category-specific mean shape
        verts, faces, verts_uv, faces_uv = self.load_mean_shape(self.cfg.shape.shape_path)

        # Load mean shape components into buffer to not be updated during training but remain part of the model state
        self.register_buffer('verts_uv', verts_uv.float())
        self.register_buffer('faces_uv', faces_uv.float())
        self.register_buffer('verts', verts.float())
        self.register_buffer('faces', faces.long())

        self.resnet_transform = torchvision.transforms.Normalize(mean=self.cfg.BGR_MEAN, std=self.cfg.BGR_STD)

        if self.cfg.multiplex.rotation2quaternion == 'svd':
            self.rot2quat = SVDQuat()
            self.n_rot = 9

        elif self.cfg.multiplex.rotation2quaternion == 'gram-schmidt':
            self.rot2quat = GramSchmidtQuat()
            self.n_rot = 6

        elif self.cfg.multiplex.rotation2quaternion == 'euler':
            self.rot2quat = EulerQuat()
            self.n_rot = 3

        else:
            raise ValueError('Rotation to quaternion approach must be either svd, gram-schmidt or euler!')

        # Dict of camera poses from configs.cameraPoseDict, the path to pre-computed camera pose dict for entire dataset
        if self.cfg.multiplex.cameraPoseDict:
            self.datasetCameraPoseDict = np.load(self.cfg.multiplex.cameraPoseDict, allow_pickle=True)['campose'].item()
            text = "Camera Multiplex Initialization"
            print("\n\n" + "+" + "-" * (143 - len(text) - 2) + "+" + text + "+" + "-" * (143 - len(text) - 2) + "+")
            print(f"Loading camera pose dictionary for the entire dataset of size {len(self.datasetCameraPoseDict)}.\n"
                  f"Expected size of datasetCameraPoseDict: 5964.\n"
                  f"P is the number of camera poses (e.g., 5x8=40)."
                  f"Format of datasetCameraPoseDict:\n"
                  f"  dict[frame_id] = [\n"
                  f"    cam_pose (Px7): [scale (1D), trans (2D), quat (4D)],\n"
                  f"    cam_score (Px1),\n"
                  f"    gt_st (Px3): [gtscale (1D), gttrans (2D)]\n"
                  f"  ]")
            print("+" + "-" * (143 - len(text) - 2) + "+" + text + "+" + "-" * (143 - len(text) - 2) + "+" + "\n\n")

        else:
            self.datasetCameraPoseDict = {}
            # Base Quaternions Initialization
            self.base_quats = self.rot2quat.get_base_quaternions(num_pose_az=self.cfg.multiplex.num_multiposeAz,
                                                                 num_pose_el=self.cfg.multiplex.num_multiposeEl,
                                                                 initial_quat_bias_deg=self.cfg.multiplex.initial_quat_bias_deg,
                                                                 elevation_bias=self.cfg.multiplex.baseQuat_biasEl,
                                                                 azimuth_bias=self.cfg.multiplex.baseQuat_biasAz)
        # Renderers
        if self.cfg.renderer.method == 'nmr':
            self.renderer = NeuralRenderer(self.cfg.renderer.img_size, perspective=self.cfg.renderer.perspective)
        elif self.cfg.renderer.method == 'softras':
            self.renderer = SoftRas(self.cfg.renderer.img_size, perspective=self.cfg.renderer.perspective)

        # Loss function
        self.rend_mask_loss_fn = loss_utils.mask_l2_dt_loss

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

        verts_uv = torch.from_numpy(verts_uv).float()  # (V=555, 2)
        verts = torch.from_numpy(verts).float()        # (V=555, 3)
        faces = torch.from_numpy(faces).long()         # (F=1106, 3)
        faces_uv = torch.from_numpy(faces_uv).float()  # (F=1106, 3, 2)

        assert (verts_uv.shape[0] == verts.shape[0])
        assert (verts_uv.shape[1] == 2)
        assert (verts.shape[1] == 3)
        assert (faces.shape[1] == 3)
        assert (faces_uv.shape == faces.shape + (2,))

        return verts, faces, verts_uv, faces_uv

    def create_input(self, batch, configs):
        """
        Create input dictionary including camera parameters of the batch samples.
        Batch contents used in this function:
        batch['inds']: indices of the batch samples
        batch['img']: RGB image (B, C, H, W).
        batch['mask']: mask image (B, H, W).
        batch['mask_dt']: mask dt image (B, H, W).
        batch['sfm_pose']: Ground-truth camera pose (SfM pose) (B, 7)

        :param batch: Batch.
        :param configs: Configurations.
        :return: Input dictionary.
        """

        frame_id, img, mask, mask_dt, gt_cam_pose = map(batch.get, ['ids', 'img', 'mask', 'mask_dt', 'sfm_pose'])
        gt_cam_pose = gt_cam_pose.float()

        # *********** Check shapes of batch items
        B, H, W = mask.shape
        assert (img.shape[0] == B)
        assert (img.shape[2:] == (H, W))
        print(f"The batch size is {B}, and image shape is {H}, {W}.")

        # ********** Deep features
        input_img = img.clone()
        for b in range(input_img.size(0)):
            input_img[b] = self.resnet_transform(input_img[b])

        # *********** Get camera poses for the batch items
        camera_poses_scores = [self.get_camera_pose(f.item(), gtcam[:3], configs.multiplex.scale_bias,)
                               for f, gtcam in zip(frame_id, gt_cam_pose)]

        # *********** Filter and adjust batch items based on available camera poses in self.camera_pose_dict
        self.filter_batch(frame_id, camera_poses_scores, img, input_img, mask, mask_dt, gt_cam_pose)

    def filter_batch(self, frame_ids, camera_poses_scores, img, input_img, masks, masks_dt, gt_camera_pose):
        """
        This function filters and adjusts batch items by processing input samples that aren't already present
        in self.camera_pose_dict.

        :param frame_ids: Frame indices. (B, 1)
        :param camera_poses_scores: Camera pose scores.
        :param img: Image. (B, C, H, W).
        :param input_img: Input image deep features. (N, F, H, W).
        :param masks: Masks. (B, H, W).
        :param masks_dt: Mask dt features. (B, H, W).
        :param gt_camera_pose: Ground-truth camera pose (SfM pose) (B, 7)
        :return:
        """

        # Get camera poses and scores for all batch items
        cam_poses = [c for c, s in camera_poses_scores]    # (B, P, 7) where some items may have None pose.
        cam_scores = [s for c, s in camera_poses_scores]   # (B, P, 1) where some items may have None score.

        idx_ok = [i for i, c in enumerate(cam_poses) if c is not None]
        if len(idx_ok) == 0:
            self.invalid_batch = True
            return
        else:
            self.invalid_batch = False
        idx_ok = torch.tensor(idx_ok, dtype=torch.long, device=frame_ids.device)
        minimum_required_batch_size = len(self.model.device_ids)
        while len(idx_ok) < minimum_required_batch_size:
            idx_ok = torch.cat([idx_ok, idx_ok[-1]], dim=0)

        cam_poses = torch.stack([cam_poses[i] for i in idx_ok], dim=0)
        cam_scores = torch.stack([cam_scores[i] for i in idx_ok], dim=0)
        frame_id = torch.index_select(frame_ids, 0, idx_ok)
        img = torch.index_select(img, 0, idx_ok)
        input_img = torch.index_select(img, 0, idx_ok)
        masks = torch.index_select(masks, 0, idx_ok)
        masks_dt = torch.index_select(masks_dt, 0, idx_ok)
        gt_camera_pose = torch.index_select(gt_camera_pose, 0, idx_ok)

        self.frame_ids = frame_ids
        self.img = img.cuda(non_blocking=True)
        self.input_img = input_img.cuda(non_blocking=True)
        self.mask = masks.cuda(non_blocking=True)
        self.mask_dt = masks_dt.cuda(non_blocking=True)
        self.cam_poses = cam_poses.cuda(non_blocking=True)
        self.cam_scores = cam_scores.cuda(non_blocking=True)
        self.gt_camera_pose = gt_camera_pose.cuda(non_blocking=True)

    def get_camera_pose(self, frame_id, gt_st0, scale_bias):
        """
        This function extracts camera poses and scores for an item of a batch with frame_id as index.
        :param frame_id: Index of the frame samples.
        :param gt_st0: ground truth camera pose (SfM pose): Scale and Translation only.
        :param scale_bias: Scale of the camera pose quaternion bias required for initialization.
        :return:
        """

        frame_id_orig = min(frame_id, int(1e6) - frame_id)
        if frame_id != frame_id_orig:
            gt_st0 = gt_st0.clone()
            gt_st0[1] *= -1

        if frame_id_orig in self.datasetCameraPoseDict.keys():
            cam_pose, score, gt_st1 = self.datasetCameraPoseDict[frame_id_orig]
            cam_pose = cam_pose.clone()
            return None, None

        # Initialize the camera pose for the batch items
        quats = self.base_quats.clone()
        trans = torch.zeros(quats.shape[0], 2).float()
        scale = torch.zeros(quats.shape[0], 1).float() + scale_bias
        cam_pose = torch.cat([scale, trans, quats], dim=-1)
        score = torch.ones(cam_pose.shape[0]).float() / cam_pose.shape[0]
        gt_st1 = gt_st0

        cam_pose[..., 0] *= gt_st0[0] / gt_st1[0]
        cam_pose[..., 1:3] += gt_st0[1:3] - gt_st1[1:3]

        if frame_id == frame_id_orig:
            return cam_pose, score
        else:
            # Reflect camera pose
            cam_pose = geom_utils.reflect_cam_pose(cam_pose)

            return cam_pose, score

    def expand_for_multipose(self, P):
        """
        This function expands the input attributes to fit for the number of multiple poses to facilitate computations.
        :param num_multipose: Number of multiple poses per instance (e.g., 5x8=40)
        :return:
        """

        B, H, W = self.input_mask.shape
        assert (self.input_img.shape[0] == B)
        assert (self.input_img.shape[2:] == (H, W))
        self.input_B = B
        self.input_H = H
        self.input_W = W
        self.n_BP = B * P

        # Expand mask for multi_pose
        self.mask = self.input_mask[:, None, :, :].repeat(1, P, 1, 1)
        self.mask = self.mask.view(B * P, H, W)
        self.mask_dt = self.input_mask_dt[:, None, :, :].repeat(1, P, 1, 1)
        self.mask_dt = self.mask_dt.view(B * P, H, W)

        # Initialize parameters and expand
        mean_shape = self.model_mean_shape  # (V, 3)
        pred_v = mean_shape[None, :, :].expand(self.n_BP, -1, -1)  # (B*P, V, 3)

        base_quat = self.cam_poses[..., 3:7]  # (B,P,4)
        base_score = self.cam_scores          # (B,P,1)

        cam_params_init = torch.zeros((B, P, 3 + self.n_rot),
                                      dtype=base_quat.dtype,
                                      device=base_quat.device)  # (B, P, [(3: scale(1), trans(2)) + self.n_rot])

        # Replace scale (1D) and translation (2D)
        cam_params_init[..., 0:3] = self.cam_poses[..., 0:3]

        base_quat = base_quat.view((self.n_BP,) + base_quat.shape[2:])  # (B*P, 4)

        cam_params_init = cam_params_init.view((self.n_BP,) + cam_params_init.shape[2:])  # (B*P, 3 + self.n_rot)

        return base_quat, base_score, pred_v, cam_params_init

    def init_optimizer(self, init_camera_params, lr, beta, opt_type):
        """
        This function initializes the optimizer.
        :param lr: Initial learning rate.
        :param beta: Beta value for the optimizer.
        :param opt_type: Type of optimizer.
        :param init_camera_params: Initial camera parameters.
        :return: Optimizer.
        """

        cam_params = init_camera_params.detach().clone()  # (B*P, 6)  Where n_batch x n_poses

        parameters = []

        scale = torch.nn.Parameter(cam_params[..., 0:1])                   # Scale (B*P,1)
        trans = torch.nn.Parameter(cam_params[..., 1:3])                   # Translation (B*P,2)
        rot_param = torch.nn.Parameter(cam_params[..., 3:self.n_rot])  # Rotation (B*P,3) or (B*P,6) or (B*P,9)

        parameters.append(scale)
        parameters.append(trans)
        parameters.append(rot_param)

        if opt_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=lr, betas=(beta, 0.999))
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=lr)
        else:
            raise NotImplementedError

        return optimizer, parameters

    def optimize_camera_pose(self, optimizer, params, base_quat, pred_v, mask, opt_steps,
                             opt_az_range, opt_el_range, opt_cr_range, mask_dt=None):
        """
        This function optimizes the camera poses.
        :param optimizer: Optimizer.
        :param params: Parameters.
        :param base_quat: Base quaternion.
        :param pred_v: Predicted vertices.
        :param mask: Mask.
        :param opt_steps: Optimization steps.
        :param opt_az_range: Optimization azimuth range.
        :param opt_el_range: Optimization elevation range.
        :param opt_cr_range: Optimization camera rotation range.
        :param mask_dt: Mask dt.

        :return: Optimized camera poses.
        """

        scale = params[..., 0:1]  # (B*P, 1)
        trans = params[..., 1:3]  # (B*P, 2)
        rot = params[..., 3:]     # (B*P, self.n_rot) where self.n_rot can be 3 or 6 or 9

        pbar = tqdm(range(opt_steps), dynamic_ncols=True, desc='cam_opt')
        for _ in pbar:

            optimizer.zero_grad()

            # .clone(): any changes inside rot2quat() won't affect the original rot parameters.
            quat = self.rot2quat.tranform(rot.clone(), opt_az_range, opt_el_range, opt_cr_range,
                                          _convert=True, base_quat=base_quat)
            cam_pred = torch.cat(([scale, trans, quat]), dim=-1)  # (N,7)

            # Forward pass through renderer
            faces_batch = self.faces[None, :, :].expand(pred_v.shape[0], -1, -1)
            rend_mask = self.renderer.forward(pred_v, faces_batch.int(), cam_pred)

            # Compute loss for each camera of the multiplex (B,P,1)
            loss = self.rend_mask_loss_fn(rend_mask, mask, reduction='none', mask2_dt=mask_dt).mean()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(rloss=loss.item())

        # Return the final camera parameters after the optimization loop
        cam_params = torch.cat([scale, trans, rot], dim=-1)

        return cam_params.detach()

    def get_quat_error(self, gt_cam_pose, cam_quat, num_multipose):
        """
        This function calculates the camera quaternion error.
        :param gt_cam_pose: Ground truth camera pose.
        :param cam_quat: Camera quaternion.
        :param num_multipose: Numer of camera poses per instance (e.g., 5x8=40).
        :return: Quaternion error.
        """

        # Calculate quaternion error
        quat = cam_quat[:, 3:7].view(self.input_B, num_multipose, 4)
        quat_error = geom_utils.hamilton_product(quat, geom_utils.quat_inverse(gt_cam_pose[:, None, 3:7]))
        _, quat_error = geom_utils.quat2axisangle(quat_error)
        quat_error = torch.min(quat_error, 2 * np.pi - quat_error)
        quat_error = quat_error * 180 / np.pi

        return quat_error

    def camera_update_loss(self, pred_v, cam_quat,
                           num_multipose, lossToScorePower, quatScorePeakiness, mask, mask_dt):
        """
        This function calculates multiplex update loss using the optimized camera quaternions.
        :param pred_v: Predicted vertices.
        :param cam_quat: Camera quaternion.
        :param mask: Ground-truth Mask.
        :param mask_dt: Ground-truth Mask time step.
        :param num_multipose: Numer of camera poses per instance (e.g., 5x8=40).
        :param lossToScorePower: Loss to score power.
        :param quatScorePeakiness: Quaternion score peakiness.
        :return: Multiplex loss (B,P,1), total loss (B,1), quaternion scores (B,P,1)
        """

        # Render mask
        faces_batch = self.faces[None, :, :].expand(pred_v.shape[0], -1, -1)
        rend_mask = self.renderer.forward(pred_v, faces_batch.int(), cam_quat)

        # Silhouette loss
        rend_mask_loss_mp = self.rend_mask_loss_fn(rend_mask, mask, reduction='none', mask2_dt=mask_dt)
        rend_mask_loss_mp = rend_mask_loss_mp.mean(dim=(-1, -2)).view(self.input_B, num_multipose)  # (B,P,1)

        # Silhouette loss with probability weights applied
        loss_min, _ = rend_mask_loss_mp.min(dim=1)
        loss_max, _ = rend_mask_loss_mp.max(dim=1)
        loss_rescaled = (rend_mask_loss_mp - loss_min[:, None]) / (loss_max[:, None] - loss_min[:, None])
        loss_rescaled = loss_rescaled.pow(lossToScorePower)
        quat_score = torch.nn.functional.softmin(loss_rescaled * quatScorePeakiness, dim=1)  # (B,P,1)
        rend_mask_loss = (rend_mask_loss_mp * quat_score).sum(dim=1).mean()  # (B,1)

        return rend_mask_loss_mp, rend_mask_loss, quat_score

    def update_camera_multiplex(self, frame_id, gt_st0, cam_pose, score):
        """
        This function updates the camera multiplex pose of a sample based on the optimization results.
        :param frame_id: Frame id of the sample.
        :param gt_st0: Ground truth camera pose of the sample, scale and translation only. (3,)
        :param cam_pose: Predicted P (e.g., 40) camera poses of the sample. (P, 7)
        :param score: P (e.g., 40) camera scores of the sample. (P, 1)
        :return:
        """
        frame_id_orig = min(frame_id, int(1e6) - frame_id)
        if frame_id != frame_id_orig:
            gt_st0 = gt_st0.clone()
            gt_st0[1] *= -1
            cam_pose = geom_utils.reflect_cam_pose(cam_pose)
        self.datasetCameraPoseDict[frame_id_orig] = (cam_pose.detach().cpu(), score.detach().cpu(),
                                                     gt_st0.detach().cpu())

    def save_datasetCameraPoseDict(self, epoch):

        # Ensure the directory exists
        stats_dir = self.configs.learning.save_path
        os.makedirs(stats_dir, exist_ok=True)

        # Save the dictionary as a .npz file
        path = f'{stats_dir}/campose_{epoch}.npz'
        np.savez(path, **self.datasetCameraPoseDict)

    def forward(self, configs, batch):

        self.create_input(batch, configs)

        # Expand for multiple poses and get number of batch x number pf poses
        base_quat, base_score, pred_v, cam_params_init = self.expand_for_multipose(configs.multiplex.num_multipose)

        optimizer, parameters = self.init_optimizer(cam_params_init,
                                                    configs.optimizer.learning_rate,
                                                    configs.optimizer.mp_beta,
                                                    configs.optimizer.mp_optimizer)

        camera_params = self.optimize_camera_pose(optimizer, parameters,
                                                  base_quat,
                                                  pred_v,
                                                  configs.optimizer.optimizeSteps,
                                                  configs.multiplex.optimizeAzRange,
                                                  configs.multiplex.optimizeElRange,
                                                  configs.multiplex.optimizeCrRange,
                                                  self.mask, mask_dt=self.mask_dt)

        quat_values = self.rot2quat.tranform(camera_params[..., 3:],
                                             configs.multiplex.optimizeAzRange,
                                             configs.multiplex.optimizeElRange,
                                             configs.multiplex.optimizeCrRange,
                                             _convert=True, base_quat=base_quat
                                             )
        camera_pred = torch.cat((camera_params[..., :3], quat_values), dim=-1)

        quat_error = self.get_quat_error(self.gt_cam_pose, camera_pred, configs.num_multipose)

        cam_loss_mp, total_loss, quat_scores_mp = self.camera_update_loss(pred_v, camera_pred,
                                                                          configs.multiplex.num_multipose,
                                                                          configs.optimizer.lossToScorePower,
                                                                          configs.multiplex.quatScorePeakiness,
                                                                          self.mask, self.mask_dt)

        # ### Update camera multiplex with optimization output results
        cam_pred = camera_pred.view(self.input_B, configs.multiplex.num_multipose, 7)
        gt_st = self.gt_camera_pose[..., 0:3]  # scale(1D) trans(2D)
        for i in range(self.frame_ids.shape[0]):
            self.update_camera_multiplex(self.frame_id[i].item(), gt_st[i], cam_pred[i], quat_scores_mp[i])

        # # Save statistics
        # pred_v = pred_v.view((self.input_B, configs.num_multipose,) + pred_v.shape[1:])[:, 0, :, :]
        #
        # # ### Statistics Dict
        # self.update_epoch_statistics(quat_error, self.cam_pred[..., 3:7], quat_scores_mp, self.gt_camera_pose[:, 3:7])
        #
        # self.save_statistics()

        # self.total_loss_num = torch.zeros_like(self.total_loss) + len(self.input_img)
        # loss_keys = {
        #     # Losses
        #     'total_loss',
        #     'rend_mask_loss',
        # }
        # gpu_weights = self.total_loss_num / self.total_loss_num.sum()
        # for k in loss_keys:
        #     setattr(self, k, (self.return_dict[k] * gpu_weights).mean())

        return {"total_loss_multiplex": cam_loss_mp,
                "total_loss": total_loss,
                "quat_scores_multiplex": quat_scores_mp,
                "quat_error": quat_error
                }
















