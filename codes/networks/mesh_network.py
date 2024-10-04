"""
Taken from Goel et al., "ucmr" GitHub, https://github.com/shubham-goel/ucmr
Modifications done for SLOPE-KP
"""
from __future__ import absolute_import, division, print_function
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .network_blocks import (conv2d, net_init, fc_stack, fc)
from .netOps.spade import SPADEGenerator_noSPADENorm


class MultiBlockResNetFeatureExtractor(nn.Module):
    def __init__(self, num_blocks=4):
        super(MultiBlockResNetFeatureExtractor, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.num_blocks = num_blocks

    def forward(self, input_tensor):
        # Pass through the initial layers
        features = self.resnet.conv1(input_tensor)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)

        # Initialize output dictionary
        output_features = {"pool1": features}

        # Extract features from specified ResNet layers
        for i in range(1, self.num_blocks + 1):
            layer = getattr(self.resnet, f'layer{i}', None)
            if layer is not None:
                features = layer(features)
                output_features[f'layer{i}'] = features

        return output_features


class MultiBlockConvEncoder(nn.Module):
    """
    This class encodes features extracted from a multi-block ResNet
    and processes them through additional convolutional layers.
    """

    def __init__(self, input_shape, num_blocks=4, batch_norm=True):
        super(MultiBlockConvEncoder, self).__init__()
        self.feature_extractor = MultiBlockResNetFeatureExtractor(num_blocks=num_blocks)
        self.enc_conv1 = conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        self.out_shape = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        net_init(self.enc_conv1)

    def forward(self, img):
        # Extract features using the feature extractor
        resnet_features = self.feature_extractor(img)

        # Process features through the convolutional layer
        output = self.enc_conv1(resnet_features['layer4'])  # Using features from the last ResNet layer

        return output, resnet_features  # Return both the encoded output and extracted features


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = MultiBlockResNetFeatureExtractor(num_blocks=4)
        self.enc_conv1 = conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = fc_stack(nc_input, nz_feat, 2)

        net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv(img)

        out_enc_conv1_bx4x4 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1_bx4x4.view(img.size(0), -1)
        feat = self.enc_fc(out_enc_conv1)

        return feat, out_enc_conv1_bx4x4


class TextureMapPredictor_SPADE_noSPADENorm(nn.Module):
    """
    Outputs UV texture map (no sampling)
    Stores mean paramters, conditions output on input feature using SPADE-normalizations
    """
    def __init__(self, opts, img_H=64, img_W=128, nc_final=3, predict_flow=False, nc_init=256):
        super().__init__()
        # nc_init should match value in Encoder()
        self.SPADE_gen = SPADEGenerator_noSPADENorm(opts, img_H, img_W, nc_init,
                                                    predict_flow=predict_flow, nc_out=nc_final)

    def forward(self, conv_feat_bxzxfhxfw):
        self.uvimage_pred = self.SPADE_gen(conv_feat_bxzxfhxfw)
        return self.uvimage_pred


class TexturePredictorUVShubham(nn.Module):
    """
    Outputs mesh texture
    """
    def __init__(self, nz_feat, uv_sampler, opts,
                    img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False):
        super().__init__()
        self.opts = opts
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.F = uv_sampler.size(0)
        self.T = uv_sampler.size(1)
        self.predict_flow = predict_flow

        if predict_flow:
            nc_final = 2
        else:
            nc_final = 3
        self.uvtexture_predictor = TextureMapPredictor_SPADE_noSPADENorm(
            opts, img_H, img_W,
            nc_final=nc_final,
            predict_flow=predict_flow
        )

        assert(uv_sampler.shape == (self.F, self.T, self.T, 2))
        # (F, T, T, 2) => (F, T*T, 2)
        self.register_buffer('uv_sampler', uv_sampler.view(self.F, self.T*self.T, 2))

    def forward(self, feat):

        uv_image_pred = self.uvtexture_predictor(feat)  # (B, C, H, W) = (Batch, Channel, Height, Width)

        uv_sampler_batch = self.uv_sampler[None].expand(uv_image_pred.shape[0], -1, -1, -1)  # (B, F, T, T, 2:[u,v])
        tex_pred = torch.nn.functional.grid_sample(uv_image_pred, uv_sampler_batch)  # (B, C, F, T, T)
        texture_pred = tex_pred.view(uv_image_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)  # (B, F, T, T, C)

        return {"texture": texture_pred.contiguous(), "uv_image_pred": uv_image_pred}


class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """
    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        delta_v = self.pred_layer(feat)
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        return delta_v

