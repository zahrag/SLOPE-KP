
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import roma


class EulerQuat:
    '''
       This class handles Euler angle rotation (R3) to unit quaternion.
       '''

    def get_base_quaternions(self, num_pose_az=1, num_pose_el=1, initial_quat_bias_deg=45., elevation_bias=0,
                             azimuth_bias=0):
        """
        This function creates base quaternions of multiple poses around multiple azimuth and elevation axes only.
        :param num_pose_az: Number of pose azimuths (e.g., 8).
        :param num_pose_el: Number of pose elevations (e.g., 5).
        :param initial_quat_bias_deg: Initial quaternion bias in degree.
        :param elevation_bias: Elevation bias.
        :param azimuth_bias: Azimuth bias.
        :return: Base Quaternions.
        """
        _axis = torch.eye(3).float()

        # Quaternion base bias
        xxx_base = [1., 0., 0.]
        aaa_base = initial_quat_bias_deg
        axis_base = torch.tensor(xxx_base).float()
        angle_base = torch.tensor(aaa_base).float() / 180. * np.pi
        qq_base = self.axisangle2quat(axis_base, angle_base)  # 4

        # Quaternion multiple pose around azimuth axis
        azz = torch.as_tensor(np.linspace(0, 2 * np.pi,
                                          num=num_pose_az, endpoint=False)).float() + azimuth_bias * np.pi / 180

        # Quaternion multiple pose around elevation axis
        ell = torch.as_tensor(np.linspace(-np.pi / 2, np.pi / 2, num=(num_pose_el + 1),
                                          endpoint=False)[1:]).float() + elevation_bias * np.pi / 180

        quat_azz = self.axisangle2quat(_axis[1], azz)  # num_pose_az,4
        quat_ell = self.axisangle2quat(_axis[0], ell)  # num_pose_el,4
        quat_el_az = self.hamilton_product(quat_ell[None, :, :], quat_azz[:, None, :])  # num_pose_az,num_pose_el,4
        quat_el_az = quat_el_az.view(-1, 4)  # num_pose_az * num_pose_el,4
        _quat = self.hamilton_product(quat_el_az, qq_base[None, ...]).float()

        return _quat

    def tranform(self, R, opt_az_range, opt_el_range, opt_cr_range, _convert=False, base_quat=None):
        """
        :param R: rotation parameters: (N, 3)
        :param base_quat: (N, 4)
        :param opt_az_range: azimuth range.
        :param opt_el_range: elevation range.
        :param opt_cr_range: cyclo-rotation range.
        :param _convert: if convert rotation?
        :return: Quaternions: (N, 4)
        """
        rot = self.convert_rotation(R, opt_az_range, opt_el_range, opt_cr_range, _convert=_convert)
        quat = self.azElRot_to_quat(rot)
        quat = self.hamilton_product(quat, base_quat)

        return quat

    def convert_rotation(self, rot, opt_az_range, opt_el_range, opt_cr_range, _convert=False):

        if not _convert:
            return rot

        az = torch.tanh(0.1 * rot[..., 0:1]) * np.pi * opt_az_range / 180
        el = torch.tanh(0.1 * rot[..., 1:2]) * np.pi * opt_el_range / 180
        cr = torch.tanh(0.1 * rot[..., 2:3]) * np.pi * opt_cr_range / 180
        rot_converted = torch.cat([az, el, cr], dim=-1)

        return rot_converted

    def azElRot_to_quat(self, azElRot):
        """
        azElRot: ...,az el ro
        """
        _axis = torch.eye(3, dtype=azElRot.dtype, device=azElRot.device)
        num_dims = azElRot.dim() - 1
        _axis = _axis.view((1,) * num_dims + (3, 3))
        azz = azElRot[..., 0]
        ell = azElRot[..., 1]
        rot = azElRot[..., 2]
        quat_azz = self.axisangle2quat(_axis[..., 1], azz)  # ...,4
        quat_ell = self.axisangle2quat(_axis[..., 0], ell)  # ...,4
        quat_rot = self.axisangle2quat(_axis[..., 2], rot)  # ...,4

        quat = self.hamilton_product(quat_ell, quat_azz)
        quat = self.hamilton_product(quat_rot, quat)

        return quat

    def axisangle2quat(self, axis, angle):
        """
        axis: B x 3: [axis]
        angle: B: [angle]
        returns quaternion: B x 4
        """
        axis = torch.nn.functional.normalize(axis, dim=-1)
        angle = angle.unsqueeze(-1) / 2
        quat = torch.cat([angle.cos(), angle.sin() * axis], dim=-1)
        return quat

    def hamilton_product(self, qa, qb):
        """Multiply qa by qb.

        Args:
            qa: B X N X 4 quaternions
            qb: B X N X 4 quaternions
        Returns:
            q_mult: B X N X 4
        """
        qa_0 = qa[..., 0]
        qa_1 = qa[..., 1]
        qa_2 = qa[..., 2]
        qa_3 = qa[..., 3]

        qb_0 = qb[..., 0]
        qb_1 = qb[..., 1]
        qb_2 = qb[..., 2]
        qb_3 = qb[..., 3]

        # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
        q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
        q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
        q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
        q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

        return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


class GramSchmidtQuat:
    '''
    This class calculate rotation using partial Gram-Schmidt method, which convert rotation matrix (R6) to unit quaternion.
    '''

    def tranform(self, rot, base_quat=None):
        """
        :param rot: rotation parameters: (N, 6)
        :param base_quat: (N, 6) (Not needed in this implementation)
        :return: Quaternions: (N, 4)
        """

        M = rot.view(-1, 3, 2)
        # R = roma.special_gramschmidt(M, epsilon=0)
        R = self.special_gramschmidt(M)
        assert roma.is_rotation_matrix(R, epsilon=1e-5)
        # print('R shape', R.shape)

        # Convert Rotation to Quaternions
        quat_batch = []
        for b in range(R.shape[0]):
            quat = self.rot2quat(R[b])
            quat_batch.append(quat.cuda())
        quat_batch = torch.stack(quat_batch).squeeze(1)
        return quat_batch

    def special_gramschmidt(self, M, epsilon=1e-5):
        """
        Returns the 3x3 rotation matrix obtained by Gram-Schmidt orthonormalization of two 3D input vectors (first two columns of input matrix M).
        Args:
            M (...x3xN tensor): batch of 3xN matrices, with N >= 2.
                Only the first two columns of the matrices are used for orthonormalization.
            epsilon (float >= 0): optional clamping value to avoid returning *Not-a-Number* values in case of ill-defined input.
        Returns:
            batch of rotation matrices (...x3x3 tensor).
        Warning:
            In case of ill-defined input (colinear input column vectors), the output will not be a rotation matrix.
        """
        M, batch_shape = self.flatten_batch_dims(M, -3)

        assert (M.dim() == 3)
        x = M[:, :, 0]
        y = M[:, :, 1]
        x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), epsilon)
        y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
        y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), epsilon)
        z = torch.cross(x, y, dim=-1)
        R = torch.stack((x, y, z), dim=-1)
        # return self.unflatten_batch_dims(R, batch_shape)
        return R

    def flatten_batch_dims(self, tensor, end_dim):
        """
        :meta private:
        Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
        """
        batch_shape = tensor.shape[:end_dim + 1]
        flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
        return flattened, batch_shape

    def unflatten_batch_dims(self, tensor, batch_shape):
        """
        :meta private:
        Revert flattening of a tensor.
        """
        return tensor.unflatten(dim=0, sizes=batch_shape) if len(batch_shape) > 0 else tensor.squeeze(0)

    def rot2quat(self, R):
        """
        Convert 3x3 rotation matrix to unit quaternion
        NB: Assumes that input R is a rotation matrix
        """

        n = torch.tensor([[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]]).cuda()
        nn = torch.norm(n).cuda()
        if nn > 0:
            n = n / nn
            tn = R[0, 0] + R[1, 1] + R[2, 2] - 1
            phi = torch.atan2(nn, tn)
        else:
            phi = torch.tensor(0.0)

        quat = torch.zeros(1, 4)
        quat[0, 0] = torch.cos(phi / 2)
        quat[0, 1:4] = torch.sin(phi / 2) * n

        return quat


class SVDQuat:
    '''
    This class creates rotation using special orthogonalization using SVD.
    outputs camera_bx7 [scale, trans_x, trans_y, quat (4D)]
    '''

    def tranform(self, rot, base_quat=None):
        """
        :param rot: rotation parameters: (N, 9)
        :param base_quat: (N, 9) (Not needed in this implementation)
        :return: Quaternions: (N, 4)
        """
        M = rot.view(-1, 3, 3)
        # R = roma.special_procrustes(M, epsilon=0)
        R = self.special_procrustes(M)
        assert roma.is_rotation_matrix(R, epsilon=1e-5)
        # print('R shape', R.shape)

        # Convert Rotation to Quaternions
        quat_batch = []
        for b in range(R.shape[0]):
            quat = self.rot2quat(R[b])
            quat_batch.append(quat.cuda())
        quat_batch = torch.stack(quat_batch).squeeze(1)
        return quat_batch

    def procrustes(self, M, force_rotation=False, gradient_eps=1e-5):
        """
        Returns the orthonormal matrix :math:`R` minimizing Frobenius norm :math:`\| M - R \|_F`.
        Args:
            M (...xNxN tensor): batch of square matrices.
            force_rotation (bool): if True, forces the output to be a rotation matrix.
            gradient_eps (float > 0): small value used to enforce numerical stability during backpropagation.
        Returns:
            batch of orthonormal matrices (...xNxN tensor).
        """
        M, batch_shape = self.flatten_batch_dims(M, -3)
        R = _ProcrustesManualDerivatives.apply(M, force_rotation, gradient_eps)
        # return self.unflatten_batch_dims(R, batch_shape)
        return R

    def special_procrustes(self, M, gradient_eps=1e-5):
        """
        Returns the rotation matrix :math:`R` minimizing Frobenius norm :math:`\| M - R \|_F`.
        Args:
            M (...xNxN tensor): batch of square matrices.
            gradient_eps (float > 0): small value used to enforce numerical stability during backpropagation.
        Returns:
            batch of rotation matrices (...xNxN tensor).
        """
        return self.procrustes(M, True, gradient_eps)

    def flatten_batch_dims(self, tensor, end_dim):
        """
        :meta private:
        Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
        """
        batch_shape = tensor.shape[:end_dim + 1]
        flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
        return flattened, batch_shape

    def unflatten_batch_dims(self, tensor, batch_shape):
        """
        :meta private:
        Revert flattening of a tensor.
        """
        return tensor.unflatten(dim=0, sizes=batch_shape) if len(batch_shape) > 0 else tensor.squeeze(0)

    def rot2quat(self, R):
        """
        Convert 3x3 rotation matrix to unit quaternion
        NB: Assumes that input R is a rotation matrix
        """

        n = torch.tensor([[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]]).cuda()
        nn = torch.norm(n).cuda()
        if nn > 0:
            n = n / nn
            tn = R[0, 0] + R[1, 1] + R[2, 2] - 1
            phi = torch.atan2(nn, tn)
        else:
            phi = torch.tensor(0.0)

        quat = torch.zeros(1, 4)
        quat[0, 0] = torch.cos(phi / 2)
        quat[0, 1:4] = torch.sin(phi / 2) * n

        return quat


class _ProcrustesManualDerivatives(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, force_rotation, gradient_eps):
        assert (M.dim() == 3 and M.shape[1] == M.shape[2]), "Input should be a BxDxD batch of matrices."
        U, D, V = roma.internal.svd(M)
        # D is sorted in descending order
        SVt = V.transpose(-1, -2)
        if force_rotation:
            # We flip the smallest singular value to ensure getting a rotation matrix
            with torch.no_grad():
                flip = (torch.det(U) * torch.det(V) < 0)
                # flip = (fast_det_3x3(U) * fast_det_3x3(V) < 0)
            if torch.is_grad_enabled():
                # This is needed to avoid a runtime error "one of the variables needed for gradient computation has been modified by an inplace operation"
                SVt = DVt.clone()
            SVt[flip, -1, :] *= -1
        else:
            flip = None
        R = U @ SVt
        # Store data for backprop
        ctx.save_for_backward(U, D, V, flip)
        ctx.gradient_eps = gradient_eps
        return R

    @staticmethod
    def backward(ctx, grad_R):
        U, D, V, flip = ctx.saved_tensors
        gradient_eps = ctx.gradient_eps

        Uik_Vjl = torch.einsum('bik,bjl -> bklij', U, V)
        Uil_Vjk = Uik_Vjl.transpose(1, 2)

        Dl = D[:, None, :, None, None]
        Dk = D[:, :, None, None, None]

        # Default Omega
        Omega_klij = (Uik_Vjl - Uil_Vjk) * roma.internal._pseudo_inverse(Dk + Dl, gradient_eps)
        # Diagonal should already be 0 thanks to clamping even in case of rank deficient input
        # Deal with flips (det(U) det(V) < 0)
        if flip is not None:
            # k!=d, l=d
            Omega_klij[flip, :-1, -1, :, :] = (Uik_Vjl[flip, :-1, -1] - Uil_Vjk[flip, :-1,
                                                                        -1]) * roma.internal._pseudo_inverse(
                Dk[flip, :-1, -1] - Dl[flip, :, -1], gradient_eps)

            # k=d, l!=d
            Omega_klij[flip, -1, :-1, :, :] = -Omega_klij[flip, :-1, -1, :, :]

        UOmega = torch.einsum('bkm, bmlij -> bklij', U, Omega_klij)
        Janalytical = torch.einsum('bkmij, bml -> bklij', UOmega, V.transpose(-1, -2))
        grad_M = torch.einsum('bkl, bklij -> bij', grad_R, Janalytical)
        return grad_M, None, None
