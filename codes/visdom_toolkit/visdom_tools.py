
import torch
from visdom_wrapper import Visdom


def visualize_rendered_data(opts, mean_shape, img, input_image, masks,
                            rend_texture, rend_mask, verts, faces, batch_ind=0,
                            heatmap_gt=None, rendered_texture=None, heatmap=None):

    if not opts.visdom_plot:
        return

    vis = Visdom(visdom_info={'port': 8097})

    vis.register(data=img[batch_ind].detach(), mode='image', debug_level=0, title='original image')

    vis.register(data=input_image[batch_ind].detach(), mode='image', debug_level=0, title='image')

    vis.register(data={"pcds": [mean_shape], "c": None}, mode='point_clouds', debug_level=0, title='Mean_Shape')

    # vis.register(data={"pcds": [verts[batch_ind]], "c": None}, mode='point_clouds', debug_level=0, title='3D_Shape')

    vis.register(data=masks[batch_ind], mode='image', debug_level=0, title='masks')

    vis.register(rend_mask[batch_ind], 'image', debug_level=0, title='rend_mask')

    vis.register(rend_texture[batch_ind], 'image', debug_level=0, title='rend_texture')

    # vis.register(data={"verts": mean_shape, "faces": faces.cpu()}, mode='mesh', debug_level=0, title='mean_shape')

    # vis.register(data={"verts": verts[batch_ind].detach().cpu(), "faces": faces.cpu()}, mode='mesh',
    #                      debug_level=0,
    #                      title='predicted_mesh')

    # vis.register(data=rend_texture[batch_ind].detach(), mode='image', debug_level=0,
    #                      title='rend_texture')
    #
    # vis.register(data=rend_mask[batch_ind].detach(), mode='image', debug_level=0,
    #                      title='rend_mask')

    # Pose prediction using Key-points
    if heatmap_gt:
        vis.register(255 * heatmap_gt[batch_ind].sum(dim=0), 'image', debug_level=0, title='heatmap_gt')
        vis.register(255 * rendered_texture[batch_ind], 'image', debug_level=0, title='gt colors')
        vis.register(255 * torch.relu(heatmap[batch_ind].detach()).sum(dim=0), 'image', debug_level=0, title='heatmap_pred')
        vis.register(255 * torch.relu(heatmap.detach()[0, 0]), 'image', debug_level=0, title='heatmap_pred 0')
        vis.register(255 * heatmap_gt[0, 0], 'image', debug_level=0, title='heatmap_gt 0')


def visualize_3DMesh_data(opts, mean_shape, img, input_image, masks, verts, faces, batch_ind=0):


    vis = Visdom(visdom_info={'port': 8097})

    vis.register(data=img[batch_ind].detach(), mode='image', debug_level=0, title='original image')
    vis.register(data=input_image[batch_ind].detach(), mode='image', debug_level=0, title='image')

    vis.register(data={"pcds": [mean_shape], "c": None}, mode='point_clouds', debug_level=0, title='Mean_Shape')
    # vis.register(data={"verts": mean_shape, "faces": faces.cpu()}, mode='mesh', debug_level=0, title='Mean_Shape_')


    vis.register(data=masks[batch_ind], mode='image', debug_level=0, title='masks')

    vis.register(data={"pcds": [verts[batch_ind]], "c": None}, mode='point_clouds', debug_level=0,
                 title='3D_Shape_Predicted')

    # vis.register(data={"verts": verts[batch_ind].detach().cpu(), "faces": faces.cpu()}, mode='mesh', debug_level=0,
    #                     title='3D_Shape_Predicted_')

