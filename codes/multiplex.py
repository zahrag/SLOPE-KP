
from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch
import numpy as np

from networks.netOps.geom_quaternion_toolkit import EulerQuat, GramSchmidtQuat, SVDQuat
from networks.netOps.geom_mapping_toolkit import reflect_cam_pose, get_base_quaternions
from networks.netOps import geom_mapping_toolkit as geom_utils


class CameraMultiplex(nn.Module):
    def __init__(self, cfg, dataset_size):
        super().__init__()

        self.cfg = cfg

        self.datasetCameraPoseDict = {}

        if cfg.rotation2quaternion == 'svd':
            self.rot2quat = SVDQuat()
            self.n_rot = 9

        elif cfg.rotation2quaternion == 'gram-schmidt':
            self.rot2quat = GramSchmidtQuat()
            self.n_rot = 6

        elif cfg.rotation2quaternion == 'euler':
            self.rot2quat = EulerQuat()
            self.n_rot = 3

        else:
            raise ValueError('Rotation to quaternion approach must be either svd, gram-schmidt or euler!')

        # Create camera pose parameters by reloading an existing one or initializing a fresh one.
        camPoseDict = self.create_dataset_cams_dict(dataset_size)

        self.dataset_camera_poses = camPoseDict["poses"]
        self.dataset_camera_scores = camPoseDict["scores"]
        self.dataset_camera_gt_st = camPoseDict["gt_st"]

        # Shape: (dataset_size, num_multipose, [1 (scale) + 2 (translation) + n_rot (rotation)])
        self.dataset_camera_params = torch.nn.Parameter(
            torch.zeros((dataset_size, cfg.num_multipose, 1 + 2 + self.n_rot)).float())

    def forward(self, frame_ids, gt_st0):
        """
        This function extract the original frame ids using the normalization applied in Phase_I, and the original
        camera pose and camera parameters from the dataset camera dictionary reloaded if it exits or initialized
        as afresh one.

        :param frame_ids: Batch frames indices (B,)
        :param gt_st0: Ground-truth scale and translation of the current batch (B, 3 (scale:1D, translation:2D))

        self.dataset_camera_poses: Dataset camera poses # (dataset_size, P, 7)
        self.dataset_camera_scores: Dataset camera scores (dataset_size, P)
        self.dataset_camera_gt_st: Dataset ground-truth scale & translation (dataset_size, 3:[scale:1D, translation:2D])
        self.dataset_camera_params: Dataset camera parameters (dataset_size, P, 6)

        :return: Final camera pose (B, P, 7) and camera scores (B, P)
        """

        # Fetch the original frame ids
        frame_ids = frame_ids.squeeze(1)
        frame_ids_orig = torch.min(frame_ids, int(1e6) - frame_ids)

        # ----- Fetch batch samples from original camera data of the entire dataset
        # Camera pose
        poses = torch.index_select(self.dataset_camera_poses, 0, frame_ids_orig)        # (B,P,7)

        # Camera score
        scores = torch.index_select(self.dataset_camera_scores, 0, frame_ids_orig)      # (B,P)

        # Ground-truth scale and translation
        gt_st1 = torch.index_select(self.dataset_camera_gt_st, 0, frame_ids_orig)       # (B,3)

        # Camera parameters
        params = torch.stack([self.dataset_camera_params[i] for i in frame_ids_orig])        # (B,P,6)

        # ----- Calculate the final pose and score of the batch samples
        camera_pose_final, scores = self.get_camera_pose(frame_ids.to(gt_st0.device),
                                                         frame_ids_orig.to(gt_st0.device),
                                                         gt_st0,
                                                         poses.to(gt_st0.device),
                                                         scores.to(gt_st0.device),
                                                         gt_st1.to(gt_st0.device),
                                                         params.to(gt_st0.device),
                                                         self.cfg.optimizeAzRange,
                                                         self.cfg.optimizeElRange,
                                                         self.cfg.optimizeCrRange)

        return camera_pose_final, scores

    def get_camera_pose(self, frame_ids, frame_ids_orig, gt_st0, poses, scores, gt_st1, params, azOR, elOR, crOR):
        """
        The goal is to adjust the original camera components, reloaded from a previous optimization process,
        to account for the random augmentations applied during their optimization or to the current batch instances.

        :param frame_ids: Batch frames indices (B,)
        :param frame_ids_orig: Batch frame indices original. (B,)
        :param gt_st0: Ground-truth scale and translation of the input batch instances (B, 3 (scale:1D, translation:2D)).
        :param poses: Dataset camera poses (B, P, 7)
        :param scores: Dataset camera scores (B, P)
        :param gt_st1: Ground-truth scale and translation of the dataset (B, 3 (scale:1D, translation:2D)).
        :param params: Dataset camera parameters (B, P, 6)
        :param azOR: Azimuth optimize range
        :param elOR: Elevation optimize range
        :param crOR: Cyclo rotation optimize range

        :return: Final camera pose (B, P, 7) and camera scores (B, P)
        """

        # Flip translation in x-direction in gt_st0 if frame_ids corresponds to flipped cameras
        frame_ids_flipped = (frame_ids > frame_ids_orig)
        frame_ids_flipped = frame_ids_flipped
        # Define the flipping tensor once outside any loops (if applicable)
        flip_tensor = torch.tensor([1, -1, 1], dtype=gt_st0.dtype, device=gt_st0.device)
        # Apply the flip to the ground truth tensor
        gt_st0_flip = gt_st0 * flip_tensor
        gt_st0 = torch.where(frame_ids_flipped[:, None], gt_st0_flip, gt_st0)

        # Adjust for dataloader perturbation (to address data augmentation)
        scale_factor = gt_st0[:, None, 0:1] / gt_st1[:, None, 0:1]

        # Create the reference values applying scale and translation
        base_scale = poses[:, :, 0:1] * scale_factor
        base_trans = (poses[:, :, 1:3] - gt_st1[:, None, 1:3]) * scale_factor + gt_st0[:, None, 1:3]
        base_quat = poses[:, :, 3:7]

        # Create values applying the scale and translation to the camera parameters (for optimization)
        scale = base_scale + params[:, :, 0:1] * scale_factor
        trans = base_trans + params[:, :, 1:3] * scale_factor
        quat = self.rot2quat.tranform(params[:, :, 3:6].clone(), azOR, elOR, crOR, _convert=True, base_quat=base_quat)

        # Flip camera pose if frame_ids_flipped
        camera_pose = torch.cat((scale, trans, quat), dim=-1)  # (B,P,7)
        # the second (translation-x), sixth (quat-3rd), and seventh (quat-4th) are flipped
        camera_pose_base = reflect_cam_pose(camera_pose)
        camera_pose_fin = torch.where(frame_ids_flipped[:, None, None], camera_pose_base, camera_pose)

        # torch.where(condition, x, y): It returns elements chosen from x or y depending on whether
        # the corresponding element in condition is True or False.
        # If condition is True for a specific element, it selects the value from x (in this case, camera_pose_base).
        # If condition is False, it selects the value from y (in this case, camera_pose).
        # If a frame is flipped (True in frame_ids_flipped): the camera pose
        # of that frame will be selected from camera_pose_base.
        # If a frame is not flipped (False in frame_ids_flipped), the camera pose will be selected from camera_pose.
        return camera_pose_fin, scores

    def update_camera_multiplex_batch(self, frame_ids, poses, scores, gt_st, mask_iou):
        """
        This function updates the camera multiplex with optimized camera parameters of a batch.

        Camera multiplex dictionary has the following format:
        dict[frame_id] = (Px7: [scale(1D) trans(2D) quat(4D)], P:score, gt_st:[gtscale(1D) gttrans(2D)])

        self.datasetCameraPoseDict: Camera dictionary of the entire dataset.
        :param frame_ids: Frame indices for the batch. (B,)
        :param poses: Camera multiplex poses. (B, P, 7)
        :param scores: Camera scores. (B, P)
        :param gt_st: Ground-truth scale and translation (B, 3)
        :param mask_iou: Mask Intersection Over Union (IOU). (B, P)
        :return:
        """
        assert (frame_ids.shape[0] == poses.shape[0] == scores.shape[0] == gt_st.shape[0] == mask_iou.shape[0])
        assert (frame_ids.dim() == 1)
        assert (scores.dim() == 2)
        assert (gt_st.dim() == 2)
        assert (poses.dim() == 3)

        frameids = frame_ids.detach().cpu()
        cams = poses.detach().cpu()
        scores = scores.detach().cpu()
        gt_st = gt_st.detach().cpu()

        frame_id_is_flip = frameids > (int(1e6)-frameids)
        flip_cams = geom_utils.reflect_cam_pose(cams)
        flip_gt_st = gt_st * torch.tensor([1, -1, 1], dtype=gt_st.dtype, device=gt_st.device)

        gt_st = torch.where(frame_id_is_flip[:, None], flip_gt_st, gt_st)
        cams = torch.where(frame_id_is_flip[:, None, None], flip_cams, cams)
        frameids = torch.where(frame_id_is_flip, int(1e6)-frameids, frameids)

        for i in range(frameids.shape[0]):
            f = frameids[i].item()
            if f not in self.datasetCameraPoseDict:
                self.datasetCameraPoseDict[f] = (cams[i, :, :], scores[i, :], gt_st[i, :])

    def create_dataset_cams_dict(self, dataset_size):
        """
        This function creates a dictionary of camera poses, scores and ground-truth scale and translation by
        loading it from an existing pretrained one, or initializing a fresh one.
        
        self.cfg.cameraPoseDict: Path to the camera pose dictionary of the entire dataset.
        self.dataset_size: Size of the dataset, the number of samples in the entire dataset.
        self.cfg.num_multipose: Total number of camera poses (e.g., 5x8=40).
        self.cfg.num_multiposeAz: Number of camera poses in azimuth axis direction (e.g., 8).
        self.cfg.num_multiposeEl: Number of camera poses in azimuth axis direction (e.g., 5).
        self.cfg.initial_quat_bias_deg: Initial camera pose quaternion bias, in degrees required for initialization.
        self.cfg.scale_bias: Scale of the camera pose quaternion bias required for initialization.

        :return: Dataset camera poses (dataset_size, P, 7),
                 Dataset camera scores (dataset_size, P)
                 Dataset ground-truth scale and translation (dataset_size, 3),
        """

        try:
            _dd = np.load(self.cfg.cameraPoseDict, allow_pickle=True)['campose'].item()
            print(f'Loaded cam_pose_dict of size {len(_dd)} (should be {dataset_size})')
            text = "Camera Multiplex Initialization"
            print("\n\n" + "+" + "-" * (143 - len(text) - 2) + "+" + text + "+" + "-" * (143 - len(text) - 2) + "+")
            print(f"Loading camera pose dictionary for the entire dataset of size {len(_dd)}.\n"
                  f"Expected size of datasetCameraPoseDict: 5964.\n"
                  f"P is the number of camera poses (e.g., 5x8=40)."
                  f"Format of datasetCameraPoseDict:\n"
                  f"  dict[frame_id] = [\n"
                  f"    cam_pose (Px7): [scale (1D), trans (2D), quat (4D)],\n"
                  f"    cam_score (Px1),\n"
                  f"    gt_st (Px3): [gtscale (1D), gttrans (2D)]\n"
                  f"  ]")
            print("+" + "-" * (143 - len(text) - 2) + "+" + text + "+" + "-" * (143 - len(text) - 2) + "+" + "\n\n")

            # Ensure keys are in range 0 -> n-1
            _kk = torch.tensor(list(_dd.keys()))
            assert ((_kk.sort()[0] == torch.arange(_kk.shape[0])).all()), "Frame IDs are out of range!"

            # Stack poses, scores, and gt_st from dictionary
            _dd = [_dd[k] for k in range(len(_dd))]
            dataset_camera_poses = torch.stack([c for (c, s, st) in _dd])   # cam_pose (Px7)
            dataset_camera_scores = torch.stack([s for (c, s, st) in _dd])  # cam_score (Px1)
            dataset_camera_gt_st = torch.stack([st for (c, s, st) in _dd])  # gt_st (Px3)

            # Sanity check on dimensions
            assert (dataset_camera_poses.shape[0] == dataset_size)
            assert (dataset_camera_poses.shape[1] == self.cfg.num_multipose)

        except (FileNotFoundError, KeyError):
            print("Camera pose dictionary file not found or is invalid. Initializing from scratch ...")

            # Initialize quaternion poses
            quats = get_base_quaternions(num_pose_az=self.cfg.num_multiposeAz,
                                         num_pose_el=self.cfg.num_multiposeEl,
                                         initial_quat_bias_deg=self.cfg.initial_quat_bias_deg)

            trans = torch.zeros(quats.shape[0], 2).float()
            scale = torch.zeros(quats.shape[0], 1).float() + self.cfg.scale_bias
            poses = torch.cat([scale, trans, quats], dim=-1)
            scores = torch.ones(poses.shape[0]).float() / poses.shape[0]
            gt_st = torch.tensor([self.cfg.scale_bias, 0., 0.]).float()

            dataset_camera_poses = poses[None, ...].expand(dataset_size, -1, -1)  # (dataset_size x P x 7)
            dataset_camera_scores = scores[None, ...].expand(dataset_size, -1)    # (dataset_size x P)
            dataset_camera_gt_st = gt_st[None, ...].expand(dataset_size, -1)      # (dataset_size x 3)

        return {"poses": dataset_camera_poses, "scores": dataset_camera_scores, "gt_st": dataset_camera_gt_st}

    def get_params(self):
        """
        This function retrieve all trained model parameters of the camera multiplex model:
        :return: Returns model parameters as a dictionary.
        """
        param_groups = [{'params': self.dataset_camera_params,},]

        return param_groups


if __name__ == "__main__":
    raise NotImplementedError
