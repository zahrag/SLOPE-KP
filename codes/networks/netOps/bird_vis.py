"""
Visualization helpers specific to birds.
Taken from Goel et al., "ucmr" GitHub, https://github.com/shubham-goel/ucmr
"""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch

from nmr import NeuralRenderer_pytorch as NeuralRenderer
# import SoftRas
import geom_transformation3D_toolkit as transform


class VisRenderer(object):
    """
    Utility to render meshes using pytorch NMR
    faces are F x 3 or 1 x F x 3 numpy
    """

    def __init__(self, img_size, faces, t_size=3, perspective=True):
        self.renderer = NeuralRenderer(img_size,perspective=perspective)
        # self.renderer = SoftRas(img_size,
        #                         perspective=perspective,
        #                         light_intensity_ambient=1.0,
        #                         light_intensity_directionals=0.0
        #                         )
        self.faces = torch.as_tensor(faces).int().cuda()
        if self.faces.dim() == 2:
            self.faces = torch.unsqueeze(self.faces, 0)
        default_tex = np.ones((1, self.faces.shape[1], t_size, t_size, t_size,
                               3))
        blue = np.array([156, 199, 234.]) / 255.
        default_tex = default_tex * blue
        # Could make each triangle different color
        self.default_tex = torch.FloatTensor(default_tex).cuda()
        # rot = transform.quaternion_about_axis(np.pi/8, [1, 0, 0])
        # This is median quaternion from sfm_pose
        # rot = np.array([ 0.66553962,  0.31033762, -0.02249813,  0.01267084])
        # This is the side view:
        import cv2
        R0 = cv2.Rodrigues(np.array([np.pi / 3, 0, 0]))[0]
        R1 = cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0]
        R = R1.dot(R0)
        R = np.vstack((np.hstack((R, np.zeros((3, 1)))), np.array([0, 0, 0,
                                                                   1])))
        rot = transform.quaternion_from_matrix(R, isprecise=True)
        cam = np.hstack([0.75, 0, 0, rot])
        self.default_cam = torch.FloatTensor(cam).cuda()
        self.default_cam = torch.unsqueeze(self.default_cam, 0)

        self.set_ambient(False)

    def set_ambient(self, set_on=True):
        if set_on:
            self.set_light_dir([0, 1, -1], 0, 1)
        else:
            self.set_light_dir([0, 1, -1], 0.38)

    def __call__(self, verts, cams=None, texture=None, rend_mask=False, to_numpy=True, vertex_texture=None):
        """
        verts is |V| x 3 cuda torch Tensor
        cams is 7, cuda torch Tensor
        Returns N x N x 3 numpy
        """
        if texture is None:
            texture = self.default_tex
        elif texture.dim() == 5:
            # Here input it F x T x T x T x 3 (instead of F x T x T x 3)
            # So add batch dim.
            texture = torch.unsqueeze(texture, 0)
        elif texture.dim() == 2:
            vertex_texture = texture[None]
        elif texture.dim() == 3:
            vertex_texture = texture

        if cams is None:
            cams = self.default_cam
        elif cams.dim() == 1:
            cams = torch.unsqueeze(cams, 0)

        if verts.dim() == 2:
            verts = torch.unsqueeze(verts, 0)


        if rend_mask:
            rend = self.renderer.forward(verts, self.faces, cams)
            rend = rend.repeat(3, 1, 1)
            rend = rend.unsqueeze(0)
        else:
            if vertex_texture is None:
                rend = self.renderer.forward(verts, self.faces, cams, texture)
            else:
                rend = self.renderer_vertices.forward(verts, self.faces.long(), cams, vertex_texture)


        if to_numpy:
            rend = rend.detach().cpu().numpy()[0].transpose((1, 2, 0))
            rend = np.clip(rend, 0, 1) * 255.0
            return rend.astype(np.uint8)
        else:
            return rend.detach().cpu()

    def rgba(self, verts, cams=None, texture=None):
        a = self(verts, rend_mask=True, cams=cams, texture=texture)
        rgb = self(verts, cams=cams, texture=texture)
        return np.concatenate((rgb,a[...,0:1]), axis=-1)

    def rotated(self, vert, deg, axis=[0, 1, 0], cam=None, texture=None, rgba=False, extra_elev=False):
        """
        vert is N x 3, torch FloatTensor
        """
        import cv2
        new_rot = cv2.Rodrigues(np.deg2rad(deg) * np.array(axis))[0]

        if extra_elev is not False:
            if extra_elev is True:
                # Left multiply the camera by 20deg on X.
                extra_elev = np.pi / 9
            else:
                extra_elev = np.pi / 180 * extra_elev

            R_elev = cv2.Rodrigues(np.array([extra_elev, 0, 0]))[0]
            new_rot = R_elev.dot(new_rot)

        new_rot = convert_as(torch.FloatTensor(new_rot), vert)

        center = vert.mean(0)
        new_vert = torch.t(torch.matmul(new_rot,
                                        torch.t(vert - center))) + center
        # new_vert = torch.matmul(vert - center, new_rot) + center

        if rgba:
            return self.rgba(new_vert, cams=cam, texture=texture)
        else:
            return self.__call__(new_vert, cams=cam, texture=texture)

    def diff_vp(self,
                verts,
                cam=None,
                angle=90,
                axis=[1, 0, 0],
                texture=None,
                kp_verts=None,
                new_ext=None,
                extra_elev=False,
                rgba=False):
        if cam is None:
            cam = self.default_cam[0]
        if new_ext is None:
            new_ext = [0.6, 0, 0]
        # Cam is 7D: [s, tx, ty, rot]
        import cv2
        quat = cam[-4:].view(1, 1, -1)
        R = transform.quaternion_matrix(
            quat.squeeze().detach().cpu().numpy())[:3, :3]
        rad_angle = np.deg2rad(angle)
        rotate_by = cv2.Rodrigues(rad_angle * np.array(axis))[0]
        # new_R = R.dot(rotate_by)

        new_R = rotate_by.dot(R)
        if extra_elev:
            # Left multiply the camera by 30deg on X.
            R_elev = cv2.Rodrigues(np.array([np.pi / 9, 0, 0]))[0]
            new_R = R_elev.dot(new_R)
        # Make homogeneous
        new_R = np.vstack(
            [np.hstack((new_R, np.zeros((3, 1)))),
             np.array([0, 0, 0, 1])])
        new_quat = transform.quaternion_from_matrix(
            new_R, isprecise=True)
        new_quat = torch.Tensor(new_quat).cuda()
        # new_cam = torch.cat([cam[:-4], new_quat], 0)
        new_ext = torch.Tensor(new_ext).cuda()
        new_cam = torch.cat([new_ext, new_quat], 0)

        if rgba:
            rend_img = self.rgba(verts, cams=new_cam, texture=texture)
        else:
            rend_img = self.__call__(verts, cams=new_cam, texture=texture)
        if kp_verts is None:
            return rend_img
        else:
            kps = self.renderer.project_points(
                kp_verts.unsqueeze(0), new_cam.unsqueeze(0))
            kps = kps[0].detach().cpu().numpy()
            return kp2im(kps, rend_img, radius=1)

    def set_bgcolor(self, color):
        self.renderer.set_bgcolor(color)

    def set_light_dir(self, direction, int_dir=0.8, int_amb=0.8):
        renderer = self.renderer.renderer
        renderer.light_direction = direction
        renderer.light_intensity_directional = int_dir
        renderer.light_intensity_ambient = int_amb

class VisRendererBatch(object):
    """
    Utility to render meshes using pytorch NMR
    faces are F x 3 or 1 x F x 3 numpy
    """

    def __init__(self, img_size, faces, t_size=3, perspective=True):
        self.renderer = NeuralRenderer(img_size,perspective=perspective)
        # self.renderer = SoftRas(img_size,perspective=perspective,light_intensity_ambient=1.0, light_intensity_directionals=0.0)
        # self.renderer_vertices = DIBRenderer(img_size,perspective=perspective)
        self.faces = torch.as_tensor(faces).int().cuda()
        if self.faces.dim() == 2:
            self.faces = torch.unsqueeze(self.faces, 0)
        default_tex = np.ones((1, self.faces.shape[1], t_size, t_size, t_size,
                               3))
        blue = np.array([156, 199, 234.]) / 255.
        default_tex = default_tex * blue
        # Could make each triangle different color
        self.default_tex = torch.FloatTensor(default_tex).cuda()
        # rot = transform.quaternion_about_axis(np.pi/8, [1, 0, 0])
        # This is median quaternion from sfm_pose
        # rot = np.array([ 0.66553962,  0.31033762, -0.02249813,  0.01267084])
        # This is the side view:
        import cv2
        R0 = cv2.Rodrigues(np.array([np.pi / 3, 0, 0]))[0]
        R1 = cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0]
        R = R1.dot(R0)
        R = np.vstack((np.hstack((R, np.zeros((3, 1)))), np.array([0, 0, 0,
                                                                   1])))
        rot = transform.quaternion_from_matrix(R, isprecise=True)
        cam = np.hstack([0.75, 0, 0, rot])
        self.default_cam = torch.FloatTensor(cam).cuda()
        self.default_cam = torch.unsqueeze(self.default_cam, 0)

        self.set_ambient(False)

    def set_ambient(self, set_on=True):
        if set_on:
            self.set_light_dir([0, 1, -1], 0, 1)
        else:
            self.set_light_dir([0, 1, -1], 0.38)

    def __call__(self, verts, cams=None, texture=None, rend_mask=False, to_numpy=True, vertex_texture=None):
        """
        verts is b x |V| x 3 cuda torch Tensor
        cams is b x 7, cuda torch Tensor
        Returns b x N x N x 3 numpy
        """

        if verts.dim() == 2:
            verts = torch.unsqueeze(verts, 0)
        batch_size = verts.shape[0]

        if texture is None:
            texture = self.default_tex.expand(batch_size, -1, -1, -1, -1, -1)
        elif texture.dim() == 5:
            # Here input it F x T x T x T x 3 (instead of F x T x T x 3)
            # So add batch dim.
            texture = torch.unsqueeze(texture, 0)
            texture = texture.expand(batch_size, -1, -1, -1, -1, -1)
        elif texture.dim() == 2:
            raise NotImplementedError
            vertex_texture = texture[None]
        elif texture.dim() == 3:
            raise NotImplementedError
            vertex_texture = texture

        if cams is None:
            cams = self.default_cam.expand(batch_size, -1)
        elif cams.dim() == 1:
            cams = torch.unsqueeze(cams, 0)
            cams = cams.expand(batch_size, -1)

        faces = self.faces.expand(batch_size, -1, -1)
        if rend_mask:
            rend = self.renderer.forward(verts, faces, cams)
            rend = rend[:,None,:,:].repeat(1, 3, 1, 1)
            # rend = rend.unsqueeze(0)
        else:
            if vertex_texture is None:
                rend = self.renderer.forward(verts, faces, cams, texture)
            else:
                rend = self.renderer_vertices.forward(verts, faces.long(), cams, vertex_texture)

        if to_numpy:
            rend = rend.detach().cpu().numpy().transpose((0, 2, 3, 1))
            rend = np.clip(rend, 0, 1) * 255.0
            return rend.astype(np.uint8)
        else:
            return rend.detach().cpu()

    def rgba(self, verts, cams=None, texture=None):
        a = self(verts, rend_mask=True, cams=cams, texture=texture)
        rgb = self(verts, cams=cams, texture=texture)
        return np.concatenate((rgb,a[...,0:1]), axis=-1)

    def rotated(self, vert, deg, axis=[0, 1, 0], cam=None, texture=None, rgba=False, extra_elev=False):
        """
        vert is b x N x 3, torch FloatTensor
        """
        assert(vert.dim()==3)
        batch_size = vert.shape[0]

        import cv2
        new_rot = cv2.Rodrigues(np.deg2rad(deg) * np.array(axis))[0]

        if extra_elev is not False:
            if extra_elev is True:
                # Left multiply the camera by 20deg on X.
                extra_elev = np.pi / 9
            else:
                extra_elev = np.pi / 180 * extra_elev

            R_elev = cv2.Rodrigues(np.array([extra_elev, 0, 0]))[0]
            new_rot = R_elev.dot(new_rot)

        new_rot = convert_as(torch.FloatTensor(new_rot), vert)
        new_rot = new_rot.unsqueeze(0).expand(batch_size, -1, -1)

        center = vert.mean(1, keepdim=True)
        new_vert = torch.transpose(torch.matmul(new_rot,
                                                torch.transpose(vert - center,1,2)
                                                )
                                    , 1, 2) + center
        # new_vert = torch.matmul(vert - center, new_rot) + center

        if rgba:
            return self.rgba(new_vert, cams=cam, texture=texture)
        else:
            return self.__call__(new_vert, cams=cam, texture=texture)

    def diff_vp(self,
                verts,
                cam=None,
                angle=90,
                axis=[1, 0, 0],
                texture=None,
                kp_verts=None,
                new_ext=None,
                extra_elev=False,
                rgba=False):
        if cam is None:
            cam = self.default_cam[0]
        if new_ext is None:
            new_ext = [0.6, 0, 0]
        # Cam is 7D: [s, tx, ty, rot]
        import cv2
        quat = cam[-4:].view(1, 1, -1)
        R = transform.quaternion_matrix(
            quat.squeeze().detach().cpu().numpy())[:3, :3]
        rad_angle = np.deg2rad(angle)
        rotate_by = cv2.Rodrigues(rad_angle * np.array(axis))[0]
        # new_R = R.dot(rotate_by)

        new_R = rotate_by.dot(R)
        if extra_elev:
            # Left multiply the camera by 30deg on X.
            R_elev = cv2.Rodrigues(np.array([np.pi / 9, 0, 0]))[0]
            new_R = R_elev.dot(new_R)
        # Make homogeneous
        new_R = np.vstack(
            [np.hstack((new_R, np.zeros((3, 1)))),
             np.array([0, 0, 0, 1])])
        new_quat = transform.quaternion_from_matrix(
            new_R, isprecise=True)
        new_quat = torch.Tensor(new_quat).cuda()
        # new_cam = torch.cat([cam[:-4], new_quat], 0)
        new_ext = torch.Tensor(new_ext).cuda()
        new_cam = torch.cat([new_ext, new_quat], 0)

        if rgba:
            rend_img = self.rgba(verts, cams=new_cam, texture=texture)
        else:
            rend_img = self.__call__(verts, cams=new_cam, texture=texture)
        if kp_verts is None:
            return rend_img
        else:
            kps = self.renderer.project_points(
                kp_verts.unsqueeze(0), new_cam.unsqueeze(0))
            kps = kps[0].detach().cpu().numpy()
            return kp2im(kps, rend_img, radius=1)

    def set_bgcolor(self, color):
        self.renderer.set_bgcolor(color)

    def set_light_dir(self, direction, int_dir=0.8, int_amb=0.8):
        renderer = self.renderer.renderer
        renderer.light_direction = direction
        renderer.light_intensity_directional = int_dir
        renderer.light_intensity_ambient = int_amb


def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src


def convert2np(x):
    # Assumes x is gpu tensor..
    if type(x) is not np.ndarray:
        return x.detach().cpu().numpy()
    return x


def tensor2mask(image_tensor, imtype=np.uint8, color=[1.,1.,1.]):
    # Input is H x W
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.expand_dims(image_numpy, 2) * 255.0
    image_numpy = np.tile(image_numpy, (1, 1, 3)) * np.array(color)[None,None,:]
    return image_numpy.astype(imtype)


def kp2im(kp, img, radius=None):
    """
    Input is numpy array or torch.cuda.Tensor
    img can be H x W, H x W x C, or C x H x W
    kp is |KP| x 2

    """
    kp_norm = convert2np(kp)
    img = convert2np(img)

    if img.ndim == 2:
        img = np.dstack((img, ) * 3)
    # Make it H x W x C:
    elif img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:  # Gray2RGB for H x W x 1
            img = np.dstack((img, ) * 3)

    # kp_norm is still in [-1, 1], converts it to image coord.
    kp = (kp_norm[:, :2] + 1) * 0.5 * img.shape[0]
    if kp_norm.shape[1] == 3:
        vis = kp_norm[:, 2] > 0
        kp[~vis] = 0
        kp = np.hstack((kp, vis.reshape(-1, 1)))
    else:
        vis = np.ones((kp.shape[0], 1))
        kp = np.hstack((kp, vis))

    kp_img = draw_kp(kp, img, radius=radius)

    return kp_img


def draw_kp(kp, img, radius=None):
    """
    kp is 15 x 2 or 3 numpy.
    img can be either RGB or Gray
    Draws bird points.
    """
    if radius is None:
        radius = max(4, (np.mean(img.shape[:2]) * 0.01).astype(int))

    num_kp = kp.shape[0]
    # Generate colors
    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    colors = 255 * np.array([cm(1. * i / num_kp)[:3] for i in range(num_kp)])
    white = np.ones(3) * 255

    image = img.copy()

    if isinstance(image.reshape(-1)[0], np.float32):
        # Convert to 255 and np.uint8 for cv2..
        image = (image * 255).astype(np.uint8)

    kp = np.round(kp).astype(int)

    for kpi, color in zip(kp, colors):
        # This sometimes causes OverflowError,,
        if kpi[2] == 0:
            continue
        cv2.circle(image, (kpi[0], kpi[1]), radius + 1, white, -1)
        cv2.circle(image, (kpi[0], kpi[1]), radius, color, -1)

    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # plt.imshow(image)
    # import ipdb; ipdb.set_trace()
    return image


def vis_verts(mean_shape, verts, face, mvs=None, textures=None):
    """
    mean_shape: N x 3
    verts: B x N x 3
    face: numpy F x 3
    textures: B x F x T x T (x T) x 3
    """
    from psbody.mesh.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewers
    if mvs is None:
        mvs = MeshViewers((2, 3))

    num_row = len(mvs)
    num_col = len(mvs[0])

    mean_shape = convert2np(mean_shape)
    verts = convert2np(verts)

    num_show = min(num_row * num_col, verts.shape[0] + 1)

    mvs[0][0].set_dynamic_meshes([Mesh(mean_shape, face)])
    # 0th is mean shape:

    if textures is not None:
        tex = convert2np(textures)
    for k in np.arange(1, num_show):
        vert_here = verts[k - 1]
        if textures is not None:
            tex_here = tex[k - 1]
            fc = tex_here.reshape(tex_here.shape[0], -1, 3).mean(axis=1)
            mesh = Mesh(vert_here, face, fc=fc)
        else:
            mesh = Mesh(vert_here, face)
        mvs[int(k % num_row)][int(k / num_row)].set_dynamic_meshes([mesh])


def vis_vert2kp(verts, vert2kp, face, mvs=None):
    """
    verts: N x 3
    vert2kp: K x N

    For each keypoint, visualize its weights on each vertex.
    Base color is white, pick a color for each kp.
    Using the weights, interpolate between base and color.

    """
    from psbody.mesh.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewer, MeshViewers
    from psbody.mesh.sphere import Sphere

    num_kp = vert2kp.shape[0]
    if mvs is None:
        mvs = MeshViewers((4, 4))
    # mv = MeshViewer()
    # Generate colors
    import pylab
    cm = pylab.get_cmap('gist_rainbow')
    cms = 255 * np.array([cm(1. * i / num_kp)[:3] for i in range(num_kp)])
    base = np.zeros((1, 3)) * 255
    # base = np.ones((1, 3)) * 255

    verts = convert2np(verts)
    vert2kp = convert2np(vert2kp)

    num_row = len(mvs)
    num_col = len(mvs[0])

    colors = []
    for k in range(num_kp):
        # Nx1 for this kp.
        weights = vert2kp[k].reshape(-1, 1)
        # So we can see it,,
        weights = weights / weights.max()
        cm = cms[k, None]
        # Simple linear interpolation,,
        # cs = np.uint8((1-weights) * base + weights * cm)
        # In [0, 1]
        cs = ((1 - weights) * base + weights * cm) / 255.
        colors.append(cs)

        # sph = [Sphere(center=jc, radius=.03).to_mesh(c/255.) for jc, c in zip(vert,cs)]
        # mvs[int(k/4)][k%4].set_dynamic_meshes(sph)
        mvs[int(k % num_row)][int(k / num_row)].set_dynamic_meshes(
            [Mesh(verts, face, vc=cs)])


def tensor2im(image_tensor, imtype=np.uint8, scale_to_range_1=False):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if scale_to_range_1:
        image_numpy = image_numpy - np.min(image_numpy, axis=2, keepdims=True)
        image_numpy = image_numpy / np.max(image_numpy)
    else:
        # Clip to [0, 1]
        image_numpy = np.clip(image_numpy, 0, 1)

    return (image_numpy * 255).astype(imtype)

def batchtensor2im(image_tensor, imtype=np.uint8, scale_to_range_1=False):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (0,2, 3, 1))
    if scale_to_range_1:
        image_numpy = image_numpy - np.min(image_numpy, axis=3, keepdims=True)
        image_numpy = image_numpy / np.max(np.max(np.max(image_numpy, axis=1, keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True)
    else:
        # Clip to [0, 1]
        image_numpy = np.clip(image_numpy, 0, 1)

    return (image_numpy * 255).astype(imtype)


def visflow(flow_img):
    # H x W x 2
    flow_img = convert2np(flow_img)
    from matplotlib import cm
    x_img = flow_img[:, :, 0]

    def color_within_01(vals):
        # vals is Nx1 in [-1, 1] (but could be larger)
        vals = np.clip(vals, -1, 1)
        # make [0, 1]
        vals = (vals + 1) / 2.
        # Append dummy end vals for consistent coloring
        weights = np.hstack([vals, np.array([0, 1])])
        # Drop the dummy colors
        colors = cm.plasma(weights)[:-2, :3]
        return colors

    # x_color = cm.plasma(x_img.reshape(-1))[:, :3]
    x_color = color_within_01(x_img.reshape(-1))
    x_color = x_color.reshape([x_img.shape[0], x_img.shape[1], 3])
    y_img = flow_img[:, :, 1]
    # y_color = cm.plasma(y_img.reshape(-1))[:, :3]
    y_color = color_within_01(y_img.reshape(-1))
    y_color = y_color.reshape([y_img.shape[0], y_img.shape[1], 3])
    vis = np.vstack([x_color, y_color])
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.imshow(x_color)
    return vis


def visflow_jonas(flow_img):
    from ..utils.viz_flow import viz_flow

    # H x W x 2
    flow = convert2np(flow_img)

    # viz_flow expects the top left to be zero.
    # Conver to image coord
    h,w,_=flow.shape

    flow_img = viz_flow(flow[:, :, 0]*w, flow[:, :, 1]*h)

    return flow_img


if __name__ == '__main__':

    # Test vis_vert2kp:
    from ..utils import mesh
    verts, faces = mesh.create_sphere()
    num_kps = 15
    num_vs = verts.shape[0]

    ind = np.random.randint(0, num_vs, num_vs)
    dists = np.stack([
        np.linalg.norm(verts - verts[np.random.randint(0, num_vs)], axis=1)
        for k in range(num_kps)
    ])
    vert2kp = np.exp(-.5 * (dists) / (np.random.rand(num_kps, 1) + 0.4))
    vert2kp = vert2kp / vert2kp.sum(1).reshape(-1, 1)

    vis_vert2kp(verts, vert2kp, faces)
