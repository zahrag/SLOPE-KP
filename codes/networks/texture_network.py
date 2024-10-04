
from __future__ import absolute_import, division, print_function
import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn

from .mesh_network import TexturePredictorUVShubham
from .netOps.mesh import (compute_uvsampler_softras, compute_uvsampler_softras_unwrapUV)
from .network_blocks import net_init
from .netOps.geom_mapping_toolkit import (convert_3d_to_uv_coordinates, convert_uv_to_3d_coordinates)


class TextureNet(nn.Module):
    def __init__(self, nz_feat, faces, verts_uv_vx2, faces_uv_fx3x2, texture_cfg):

        super().__init__()
        self.nz_feat = nz_feat
        self.faces = faces
        self.verts_uv = verts_uv_vx2
        self.faces_uv = faces_uv_fx3x2
        self.texture_cfg = texture_cfg  # Texture configurations

        sphere_verts_np = convert_uv_to_3d_coordinates(verts_uv_vx2).numpy()
        if not texture_cfg.textureUnwrapUV:
            uv_sampler = compute_uvsampler_softras(
                sphere_verts_np,
                faces.numpy(),
                tex_size=texture_cfg.tex_size,
                shift_uv=texture_cfg.texture_uvshift
            )
        else:
            faces_uv_fx3x2_np = faces_uv_fx3x2.detach().cpu().numpy()
            uv_sampler = compute_uvsampler_softras_unwrapUV(
                faces_uv_fx3x2_np,
                faces.numpy(),
                tex_size=texture_cfg.tex_size,
                shift_uv=texture_cfg.texture_uvshift
            )

        uv_sampler = torch.FloatTensor(uv_sampler)   # (F, T, T, 2)
        if not texture_cfg.textureImgCustomDimension:
            img_H = int(2**np.floor(np.log2(np.sqrt(faces.shape[0]) * texture_cfg.tex_size)))
            img_W = 2 * img_H
        else:
            img_H = texture_cfg.textureImgH
            img_W = texture_cfg.textureImgW
        print(f'textureImg:     {img_H}x{img_W}')
        self.texturePred = TexturePredictorUVShubham(nz_feat, uv_sampler, texture_cfg, img_H=img_H, img_W=img_W,
                                                     predict_flow=texture_cfg.texture_predict_flow)
        net_init(self.texturePred)

    def forward(self, img, feat):
        """
        img tensor: (B: Batch, C: Channel, H: Height, W: Width)
        returns code: Texture prediction results: (B: Batch, F: Faces, T: Texture Height, T: Texture Width, C: Channel)
        """

        assert (feat is not None)
        assert self.texture_cfg.texture_use_conv_featz
        texture = self.texturePred(feat)

        return texture


if __name__ == "__main__":
    raise NotImplementedError
