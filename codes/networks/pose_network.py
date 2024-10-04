
from __future__ import absolute_import, division, print_function
import sys
sys.path.append('.')

import torch.nn as nn

from .network_blocks import fc_stack
from .camera_network import (CameraNetQuat, CameraNetEuler, CameraNetGramSchmidt, CameraNetSVD)
from .keypoint_network import PosePredictor


class PoseNet(nn.Module):
    def __init__(self, input_shape, nz_feat, faces, verts, layer_channels,
                 pose_configs, is_training):

        super().__init__()
        self.input_shape = input_shape
        self.nz_feat = nz_feat
        self.faces = faces
        self.verts = verts

        self.layer_channels = layer_channels
        self.num_keypoints = pose_configs.num_keypoints
        self.perspective = pose_configs.perspective
        self.pred_pose_quat = pose_configs.pred_pose_quat
        self.pred_pose_gramschmidt = pose_configs.pred_pose_gramschmidt
        self.pred_pose_svd = pose_configs.pred_pose_svd
        self.pred_pose_keypoint = pose_configs.pred_pose_keypoint

        self.transformer = pose_configs.transformer
        self.robust = pose_configs.robust
        self.end2end = pose_configs.end2end
        self.is_training = is_training

        # Pose
        # Model predicts Quaternions: 4D
        if self.pred_pose_quat:

            self.posePred = nn.Sequential(fc_stack(input_shape, self.nz_feat, 2),
                                          CameraNetQuat(nz_feat=self.nz_feat,
                                                        scale_bias=(3 if self.perspective else 0.75)))

        # Model predicts Rotation (SVD): 3x3=9D
        elif self.pred_pose_svd:
            self.posePred = nn.Sequential(fc_stack(input_shape, self.nz_feat, 2),
                                          CameraNetSVD(nz_feat=self.nz_feat,
                                                       scale_bias=(3 if self.perspective else 0.75)))

        # Model predicts Rotation (gramschmidt): 3x2=6D
        elif self.pred_pose_gramschmidt:

            self.posePred = nn.Sequential(fc_stack(input_shape, self.nz_feat, 2),
                                          CameraNetGramSchmidt(nz_feat=self.nz_feat,
                                                               scale_bias=(3 if self.perspective else 0.75)))

        # Model predicts heatmap keypoints
        elif self.pred_pose_keypoint:

            # Destructure input shape
            height, width = input_shape

            # Compute focal lengths and center coordinates
            focal_length_x = width
            focal_length_y = height
            center_x = width / 2
            center_y = height / 2

            # Create intrinsic camera matrix K as a tuple
            K = (focal_length_x, focal_length_y, center_x, center_y)

            self.posePred = PosePredictor(intrinsic_camera_matrix=K,
                                          layer_channels=self.layer_channels,
                                          transformer=self.transformer,
                                          end2end=self.end2end,
                                          robust=self.robust,
                                          faces=self.faces,
                                          verts=self.verts,
                                          is_training=self.is_training,
                                          num_keypoints=self.num_keypoints,
                                          image_size=self.input_shape)

    def forward(self, feat, resnet_feat, mean_shape, shape):
        """
        :param feat: Resnet output features of shape (B, N1, H, W)
        :param resnet_feat: Resnet last layer features of shape (B, N2, H, W)
        :param mean_shape: Mean shape of shape (B, V, D)
        :param shape: Shape (B, V, D)

        returns:
        if keypoint: Dictionary of pose pred components.
        else: Camera: (B, 7)
        """

        if self.pred_pose_keypoint:
            verts = mean_shape + shape
            pose_net = self.posePred(resnet_feat, verts)
        else:
            pose_net = self.posePred(feat)

        return pose_net


if __name__ == "__main__":
    raise NotImplementedError
