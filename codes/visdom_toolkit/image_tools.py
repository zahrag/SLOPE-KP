
from __future__ import absolute_import, division, print_function

import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
from imageio import imwrite

import plotly.graph_objects as go

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')

import torch
import torchvision
import numpy as np

import torchvision.transforms.functional as F


def get_img_name(img_path):
    image_path = img_path[0]
    var = []
    r_var = []
    cnt = 0
    for element in image_path[::-1]:
        if cnt < 2:
            var += element
            if element == '/':
                cnt += 1
                for i in range(len(var) - 1):
                    r_var += var[-(i + 2)]
                if cnt == 1:
                    Name = ''.join(r_var[:-4])
                    var = []
                    r_var = []
                elif cnt == 2:
                    Folder = ''.join(r_var)
                    var = []
                    r_var = []

    return [Name, Folder]


def make_torchvision_plot(image, path='', plot_image=False):

    if not plot_image:
        return

    if not isinstance(image, list):
        imgs = [image]
    torchvision.utils.save_image(imgs, path)


def show_save_images(opts, input_imgs_batch, input_masks_batch, input_masks_pred_batch, input_texts_pred_batch,
                     name_folder, path='', batch_ind=0):

    """
        :param opts: Hyperparameter settings
        :param input_imgs_batch: Original image batch
        :param input_masks_batch: Ground-truth mask batch
        :param input_masks_pred_batch: Rendered mask
        :param input_texts_pred_batch: Rendered texture

        :param name_folder: Folder name
        :param path: Path directory to save plots
        :param batch_ind: index of batch
        :param make_plot: make plots?
        :return:
        """
    if not opts.save_images:
        return

    # Select the corresponding image of the batch
    input_img = input_imgs_batch[batch_ind]
    input_mask = input_masks_batch[batch_ind]
    input_mask_pred = input_masks_pred_batch[batch_ind]
    input_text_pred = input_texts_pred_batch[batch_ind]

    image = input_img.cpu()
    texture_pred = input_text_pred.cpu()

    mask = input_mask[None, :, :].cpu()
    mask = torch.cat((mask, mask, mask), dim=0)

    mask_pred = input_mask_pred[None, :, :].cpu()
    mask_pred = torch.cat((mask_pred, mask_pred, mask_pred), dim=0)

    # ######### Save Images ##############
    images_path = osp.join(path, name_folder[1])
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Make images together
    imgs = torchvision.utils.make_grid([image, texture_pred, mask, mask_pred])
    make_torchvision_plot(imgs, path=f'{images_path}/{name_folder[0]}_all4.png', plot_image=True)

    # Make images separately
    img = torchvision.utils.make_grid([image])
    make_torchvision_plot(img, path=f'{images_path}/{name_folder[0]}_image.png', plot_image=False)

    mask_gt = torchvision.utils.make_grid([mask])
    make_torchvision_plot(mask_gt, path=f'{images_path}/{name_folder[0]}_mask_gt.png', plot_image=False)

    text_pred = torchvision.utils.make_grid([texture_pred])
    make_torchvision_plot(text_pred, path=f'{images_path}/{name_folder[0]}_texture_pred.png', plot_image=True)

    mask_pred = torchvision.utils.make_grid([mask_pred])
    make_torchvision_plot(mask_pred, path=f'{images_path}/{name_folder[0]}_mask_pred.png', plot_image=True)

    # ######### Show Images ##############
    if opts.show_images:
        titles = ['original_image', 'rendered_texture', 'gt_mask', 'rendered_mask']
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            plt.show()


def predicted_3Dmesh_plot(opts, img_input, img_shape, img_shape_cam, img_shape_gtcam,
                          images_path, name_folder, batch_ind=0):

    if not opts.plot_3D_Mesh:
        return

    # ######### Save Images ##############
    path = f'{images_path}/{name_folder[1]}'
    if not os.path.exists(path):
        os.makedirs(path)

    imwrite(osp.join(path, f'{name_folder[0]}_img_input.png'), img_input[batch_ind])
    imwrite(osp.join(path, f'{name_folder[0]}_img_shape.png'), img_shape[batch_ind])
    imwrite(osp.join(path, f'{name_folder[0]}_img_shape_cam.png'), img_shape_cam[batch_ind])
    imwrite(osp.join(path, f'{name_folder[0]}_img_shape_gtcam.png'), img_shape_gtcam[batch_ind])


def mesh_3D_plot(opts, verts, sfm_kps, plot_meanshape=False):

    if not plot_meanshape:
        return

    verts_n = verts.cpu().data.numpy()
    fig = go.Figure(data=[go.Mesh3d(x=[row[0] for row in verts_n],
                                    y=[row[1] for row in verts_n],
                                    z=[row[2] for row in verts_n],
                                    color='lightpink', opacity=0.50)])

    kps = sfm_kps.cpu().data.numpy()
    fig.add_trace(go.Mesh3d(x=[row[0] for row in kps],
                            y=[row[1] for row in kps],
                            z=[row[2] for row in kps],
                            color='cyan', opacity=0.50))
    # fig1.show()
    # fig.write_image(f"{opts.cache_dir}/mean_shape.png")

    ax = plt.axes(projection='3d')
    ax.plot_trisurf(verts_n[:, 0], verts_n[:, 1], verts_n[:, 2]) #, linewidth=0, antialiased=False)
    # ax.scatter(verts_n[:, 0], verts_n[:, 1], verts_n[:, 2])
    # ax.scatter(kps[0][:, 0], kps[0][:, 1], kps[0][:, 2])
    # ax.plot_trisurf(kps[0][:, 0], kps[0][:, 1], kps[0][:, 2], linewidth=0, antialiased=False)

    plt.savefig(f"{opts.cache_dir}/mean_shape.png")


def pose_pred_keypoints_plots(opts, color_map, image_batch, heatmap_gt_input, heatmap_pred_input, rendered_texture,
                              name_folder, batch_ind=0, path=''):

    """
        :param opts: Hyperparameter settings
        :param heatmap_gt: ground-truth heatmap
        :param heatmap_pred: predicted heatmap using semantic segmentation network
        :param rendered_texture: Rendered texture employing 3D keypoint and camera-pose from UCMR pretrained model,
                                 together with labeled texture from ground-truth color map.

        :param name_folder: Folder name
        :param path: Path directory to save plots.
        :param key_plot: make plots?
        :return:
        """

    if not opts.plot_keypoints:
        return

    images_path = osp.join(path, name_folder[1])
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # settings to set colors
    k = 1  # 255

    color_map = k*color_map
    color_map = color_map.cpu()

    image = image_batch[batch_ind]
    image = image.cpu()

    colors_gt = k * rendered_texture[batch_ind]
    colors_gt = colors_gt.cpu()

    heatmap_gt = k * heatmap_gt_input[batch_ind].sum(dim=0)
    h_gt = heatmap_gt[None, :, :].cpu()
    h_gt = torch.cat((h_gt, h_gt, h_gt), dim=0)

    heatmap_pred = k * torch.relu(heatmap_pred_input[batch_ind].detach()).sum(dim=0)
    h_pred = heatmap_pred[None, :, :].cpu()
    h_pred = torch.cat((h_pred, h_pred, h_pred), dim=0)

    heatmap_gt0 = k * heatmap_gt_input[0, 0]
    h_gt0 = heatmap_gt0[None, :, :].cpu()
    h_gt0 = torch.cat((h_gt0, h_gt0, h_gt0), dim=0)

    heatmap_pred0 = k * torch.relu(heatmap_pred_input.detach()[0, 0])
    h_pred0 = heatmap_pred0[None, :, :].cpu()
    h_pred0 = torch.cat((h_pred0, h_pred0, h_pred0), dim=0)

    # ### Ground-truth color map
    cmap = torchvision.utils.make_grid([color_map])
    make_torchvision_plot(cmap, path=f'{images_path}/{name_folder[0]}_cmap.png', plot_image=True)

    # ### Original Input Image
    img = torchvision.utils.make_grid([image])
    make_torchvision_plot(img, path=f'{images_path}/{name_folder[0]}_image.png', plot_image=False)

    # ### Ground-truth color map: labeled texture
    grid_color = torchvision.utils.make_grid([colors_gt])
    make_torchvision_plot(grid_color, path=f'{images_path}/{name_folder[0]}_gt_colors.png', plot_image=False)

    # (1) Plot together
    # ### Heatmaps
    grid_h = torchvision.utils.make_grid([h_gt, h_pred])
    make_torchvision_plot(grid_h, path=f'{images_path}/{name_folder[0]}_heatmaps.png', plot_image=False)

    grid_h0 = torchvision.utils.make_grid([h_gt0, h_pred0])
    make_torchvision_plot(grid_h0, path=f'{images_path}/{name_folder[0]}_heatmaps0.png', plot_image=False)

    # (2) Plot separately
    # ### Heatmaps ground-truth
    grid_hgt = torchvision.utils.make_grid([h_gt])
    make_torchvision_plot(grid_hgt, path=f'{images_path}/{name_folder[0]}_heatmaps_gt.png', plot_image=False)

    # ### Heatmap predicted
    grid_hpred = torchvision.utils.make_grid([h_pred])
    make_torchvision_plot(grid_hpred, path=f'{images_path}/{name_folder[0]}_heatmaps_pred.png', plot_image=False)

    # ### Heatmap ground-truth-0
    grid_hgt0 = torchvision.utils.make_grid([h_gt0])
    make_torchvision_plot(grid_hgt0, path=f'{images_path}/{name_folder[0]}_heatmaps_gt0.png', plot_image=False)

    # ### Heatmap predicted-0
    grid_hpred0 = torchvision.utils.make_grid([h_pred0])
    make_torchvision_plot(grid_hpred0, path=f'{images_path}/{name_folder[0]}_heatmaps_pred0.png', plot_image=False)

    # (3) 4 images in row
    grid_4 = torchvision.utils.make_grid([image, colors_gt, h_gt, h_pred])
    make_torchvision_plot(grid_4, path=f'{images_path}/{name_folder[0]}_imgs4.png', plot_image=False)


def save_result(opts,  mean_iou, mean_qerr, path):

    info = f'Mean intersection of union:{mean_iou}\nMean quaternion error:{mean_qerr}'
    with open(f'{path}/{opts.file_name}.txt', "w") as fp:
        fp.write(info)


def save_hist(opts, bench_stats, path):

    if not opts.hist_plot:
        return

    n, bins, patches = plt.hist(bench_stats['quat_error'], 50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Quaternion error')
    plt.grid(True)
    plt.savefig(f'{path}/{opts.file_name}.png')
    plt.show()


def save_keypoints_plots(opts, cams_preds, path, name_folder):

    if not opts.save_keypoints:
        return

    points2d = cams_preds["points2d"].detach()
    points3d = cams_preds["points3d"].detach()

    images_path = osp.join(path, name_folder[1])
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    torch.save(points3d, f'{images_path}/{name_folder[0]}_keypoints3D.pt')
    torch.save(points2d, f'{images_path}/{name_folder[0]}_keypoints2D.pt')







