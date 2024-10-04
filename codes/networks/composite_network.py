
from __future__ import absolute_import, division, print_function
import sys
sys.path.append('.')

import torch.nn as nn
import torch

from .mesh_network import MultiBlockConvEncoder as ConvEnc
from .shape_network import ShapeNet
from .texture_network import TextureNet
from .pose_network import PoseNet


class CompositeNet(nn.Module):
    def __init__(self, input_size, is_training, net_cfg,
                 shape_cfg=None, texture_cfg=None, pose_cfg=None,
                 mean_shape=None, faces=None, verts=None, verts_uv=None, faces_uv=None):

        super().__init__()
        self.input_shape = (input_size, input_size)
        self.register_buffer('faces', faces + 0)

        # Conv Encoder Network
        self.convEncNet = ConvEnc(self.input_shape, num_blocks=4)

        # Shape Network
        if net_cfg.pred_shape:
            self.shapeNet = ShapeNet(self.convEncNet.out_shape, net_cfg.nz_feat, mean_shape, shape_cfg)

        else:
            self.shapeNet = None

        # Texture Network
        if net_cfg.pred_texture:
            self.textureNet = TextureNet(net_cfg.nz_feat, self.faces, verts_uv, faces_uv, texture_cfg)

        else:
            self.textureNet = None

        # Pose Network
        if net_cfg.pred_pose:

            # To prevent gradients from being computed for any operations performed within it
            with torch.no_grad():
                self.convEncNet.eval()
                _, tmp = self.convEncNet(torch.rand((1, 3, 224, 224)))
                layer_channels = {L: tmp[L].shape[1] for L in tmp.keys()}

            self.poseNet = PoseNet(self.convEncNet.out_shape, net_cfg.nz_feat, self.faces, verts, layer_channels,
                                   pose_cfg, is_training)

        else:
            self.poseNet = None

    def forward(self, img):
        """
        :param img:
        :return: shape deformation, (B: Batch, V: Num Vertices, D: 3D),
                 textures, (B: Batch, F: Num Faces, T: Texture Size, T: Texture Size, C: Channel)
                 pose (B: Batch, 7: [(7D: scale (1D), translation (2D), quaternions (4D))]) or a dict if keypoint used.
        """

        output_feat, resnet_feat = self.convEncNet(img)
        feat_flat = output_feat.view(output_feat.shape[0], -1)

        shape = self.shapeNet(img, feat_flat) if self.shapeNet else None
        texture = self.textureNet(img, output_feat) if self.textureNet else None
        pose = self.poseNet(feat_flat, resnet_feat, self.shapeNet.get_mean_shape(), shape) if self.poseNet else None

        return shape, texture, pose

    def get_params(self):
        """
        Retrieve all trained model parameters:
        1. Feature extraction convolutional encoder parameters.
        2,3. Shape network parameters and category-specific mean shape parameters.
        4. Texture network parameters.
        5. Pose network parameters.

        :return: Model parameters as a list of dictionaries.
        """
        param_groups = [{'params': self.convEncNet.parameters()}]  # Conv encoder parameters

        # Add shapeNet (including mean_shape), textureNet, and poseNet parameters if they exist
        for net in [self.shapeNet, self.textureNet, self.poseNet]:
            if net is not None:
                param_groups.append({'params': net.parameters()})

        return param_groups


if __name__ == "__main__":
    raise NotImplementedError
