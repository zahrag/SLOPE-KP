
from __future__ import absolute_import, division, print_function
import sys
sys.path.append('.')

import torch
import torch.nn as nn

from .mesh_network import ShapePredictor
from .network_blocks import fc_stack
from .netOps.mesh import find_symmetry


class ShapeNet(nn.Module):
    def __init__(self, input_shape, nz_feat, mean_shape, shape_cfg):
        super().__init__()

        # Parametrize symmetric mesh
        if shape_cfg.symmetric_mesh:
            left_sym_idx, right_partner_idx, indep_idx = find_symmetry(mean_shape)
        else:
            left_sym_idx = torch.tensor([], dtype=torch.long)
            right_partner_idx = torch.tensor([], dtype=torch.long)
            indep_idx = torch.arange(mean_shape.shape[0])

        ilr_idx = torch.cat([indep_idx, left_sym_idx, right_partner_idx], dim=0)
        ilr_idx_inv = torch.zeros(mean_shape.shape[0], dtype=torch.long)
        ilr_idx_inv[ilr_idx] = torch.arange(mean_shape.shape[0])

        # These are not updated during training but are part of the model state
        self.register_buffer('mean_shape_orig', mean_shape + 0)
        self.register_buffer('left_sym_idx', left_sym_idx)
        self.register_buffer('right_partner_idx', right_partner_idx)
        self.register_buffer('indep_idx', indep_idx)
        self.register_buffer('ilr_idx_inv', ilr_idx_inv)

        learnable_idx = torch.cat([indep_idx, left_sym_idx])
        assert (learnable_idx.max() < mean_shape.shape[0])
        mean_shape = torch.index_select(mean_shape, 0, learnable_idx)

        # Mean shape is also optimized during training
        self.mean_shape = torch.nn.Parameter(mean_shape)

        self.reflection_tensor = torch.tensor([-1, 1, 1], dtype=torch.float32)
        assert ((self.mean_shape_orig - self.get_mean_shape()).norm(dim=-1).max() <= 1e-4)

        # Shape
        self.shapePred = nn.Sequential(fc_stack(input_shape, nz_feat, 2),
                                       ShapePredictor(nz_feat, num_verts=self.mean_shape_orig.shape[0]))

    def forward(self, img, feat):
        """
        img tensor (B: Batch, C: Channel, H: Height, W: Width)
        returns: shape tensor (B: Batch, V: Vertices, D: Dimension (x-y-z))
        """

        shape = self.shapePred(feat)
        shape = self.symmetrize_mesh(shape)  # Should it be conditioned?!!
        mean_shape = self.get_mean_shape()

        return {"shape": shape, "mean_shape_learned": mean_shape}

    def get_mean_shape(self):
        return self.symmetrize_mesh(self.mean_shape)

    def symmetrize_mesh(self, verts):
        """
        Assumes vertices are arranged as [indep, left]
        """
        num_indep = self.indep_idx.shape[0]
        indep = verts[..., :num_indep, :]
        left = verts[..., num_indep:, :]
        # right = verts[..., num_indep:, :] * torch.tensor([-1, 1, 1], dtype=verts.dtype,
        #                                                 device=verts.device).view((1,) * (verts.dim() - 1) + (3,))
        right = verts[..., num_indep:, :] * self.reflection_tensor.to(verts.device).view((1,) * (verts.dim() - 1) + (3,))
        ilr = torch.cat([indep, left, right], dim=-2)
        assert (self.ilr_idx_inv.max() < ilr.shape[-2]), f'idx ({self.ilr_idx_inv.max()}) >= dim{-2} of {ilr.shape}'
        verts_full = torch.index_select(ilr, -2, self.ilr_idx_inv)

        return verts_full


if __name__ == "__main__":
    raise NotImplementedError
