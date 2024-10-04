
from __future__ import division, print_function

import scipy.io as sio
import os.path as osp
import cv2
import numpy as np
import torch
import torchvision
from absl import app, flags
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from networks.netOps import geom_mapping_toolkit as geom_utils
from networks.netOps import image as image_utils

import networks.netOps.geom_transformation3D_toolkit as transformations


class CUBDataset(Dataset):
    """
        Base class for dataset handling.

        Args:
            cfg (Config): Configuration object containing dataset parameters.
            annotations (list): List of annotations for the dataset.
            sfm_annotations (list): List of SFM annotations.
            keypoint_3D (np.array): 3D keypoint data.
            kp_perm (np.array): Keypoint permutation array.
            num_imgs (int): Total number of images in the dataset.
            img_dir (str): Directory where images are stored.

        """

    def __init__(self, cfg, annotations, sfm_annotations, keypoint_3D, kp_perm, num_imgs, img_dir):

        self.anno = annotations
        self.anno_sfm = sfm_annotations
        self.kp3d = keypoint_3D
        self.kp_perm = kp_perm
        self.num_imgs = num_imgs
        self.img_dir = img_dir

        self.cfg_loader = cfg.dataloader
        self.cfg_aug = cfg.augmentation
        self.cfg_aug.rngFlip = np.random.RandomState(0)
        self.flip_transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                             torchvision.transforms.RandomHorizontalFlip(1),
                                                             torchvision.transforms.ToTensor()])

        if self.cfg_loader.use_cameraPoseDict_as_gt:
            self.cameraPoseDict = get_campose_dict(self.cfg_loader.cameraPoseDict,
                                                   self.cfg_loader.cameraPoseDict_isCamPose)
            text = "Dataloader"
            print("\n\n" + "+" + "-" * (143 - len(text) - 2) + "+" + text + "+" + "-" * (143 - len(text) - 2) + "+")
            print(f"Loading camera pose dictionary for the entire dataset of size {len(self.cameraPoseDict)}.\n"
                  "The goal is to use the best camera of Multiplex as ground-truth pose replaces with 'sfm_pose'."
                  f"Expected size of datasetCameraPoseDict: {num_imgs}.\n"
                  f"P is the number of camera poses (e.g., 5x8=40)."
                  f"Format of datasetCameraPoseDict:\n"
                  f"  dict[frame_id] = [\n"
                  f"    cam_pose (Px7): [scale (1D), trans (2D), quat (4D)],\n"
                  f"    cam_score (Px1),\n"
                  f"    gt_st (Px3): [gtscale (1D), gttrans (2D)]\n"
                  f"  ]")
            text = 'Data is Loaded'
            print("+" + "-" * (143 - len(text) - 2) + "+" + text + "+" + "-" * (143 - len(text) - 2) + "+" + "\n\n")
            if self.cfg_loader.cameraPoseDict_mergewith:
                for _ll in self.cfg_loader.cameraPoseDict_mergewith:
                    cameraPoseDict_merge = get_campose_dict(_ll, self.cfg_loader.cameraPoseDict_isCamPose)
                    self.cameraPoseDict = {**cameraPoseDict_merge, **self.cameraPoseDict}
                    print(f'Merged cam_pose_dict of size {len(cameraPoseDict_merge)}.')

        return

    def get_anno(self, index):

        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        return img_path, data.mask, bbox, sfm_pose, kp, vis

    def forward_img(self, index):

        img_path, mask, bbox, sfm_pose, kp, vis = self.get_anno(index)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = img / 255.0

        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        assert(img.shape[:2] == mask.shape)
        mask = np.expand_dims(mask, 2)

        # Peturb bbox
        if self.cfg_aug.tight_crop:
            self.cfg_aug.padding_frac = 0.0

        # ######################## DATA AUGMENTATION ###############################
        # jitter and padding fractions during training and testing
        bbox = image_utils.peturb_bbox(bbox, pf=self.cfg_aug.padding_frac, jf=self.cfg_aug.jitter_frac)

        # # Apply augmentation (jitter fraction) during training only
        # if self.cfg_loader.split == 'train':
        #     bbox = image_utils.peturb_bbox(
        #         bbox, pf=self.cfg_aug.padding_frac, jf=self.cfg_aug.jitter_frac)
        # else:
        #     bbox = image_utils.peturb_bbox(
        #         bbox, pf=self.cfg_aug.padding_frac, jf=0)

        if self.cfg_aug.tight_crop:
            bbox = bbox
        else:
            bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)

        # scale image, and mask. And scale kps.
        if self.cfg_aug.tight_crop:
            img, mask, kp, sfm_pose = self.scale_image_tight(img, mask, kp, vis, sfm_pose)
        else:
            img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # Mirror image on random.
        if self.cfg_loader.split == 'train':
            flipped, img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)
        else:
            flipped = False

        if self.cfg_aug.occlusion:
            img = image_utils.occlude(img, self.cfg_aug)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return flipped, img, kp_norm, mask, sfm_pose, img_path

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        if kp is not None:
            vis = kp[:, 2, None] > 0
            kp = np.stack([2 * (kp[:, 0] / img_w) - 1, 2 * (kp[:, 1] / img_h) - 1, kp[:, 2]]).T
            kp = vis * kp
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1

        return kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        if kp is not None:
            assert (vis is not None)
            kp[vis, 0] -= bbox[0]
            kp[vis, 1] -= bbox[1]

            kp[vis, 0] = np.clip(kp[vis, 0], a_min=0, a_max=bbox[2] - bbox[0])
            kp[vis, 1] = np.clip(kp[vis, 1], a_min=0, a_max=bbox[3] - bbox[1])

        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]

        return img, mask, kp, sfm_pose

    def scale_image_tight(self, img, mask, kp, vis, sfm_pose):

        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[1]
        bheight = np.shape(img)[0]

        scale_x = self.cfg_aug.img_size/bwidth
        scale_y = self.cfg_aug.img_size/bheight

        img_scale = cv2.resize(img, (self.cfg_aug.img_size, self.cfg_aug.img_size))
        mask_scale = cv2.resize(mask, (self.cfg_aug.img_size, self.cfg_aug.img_size))

        if kp is not None:
            assert(vis is not None)
            kp[vis, 0:1] *= scale_x
            kp[vis, 1:2] *= scale_y
        sfm_pose[0] *= scale_x
        sfm_pose[1] *= scale_y

        return img_scale, mask_scale, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.cfg_aug.img_size / float(max(bwidth, bheight))

        img_scale, _ = image_utils.resize_img(img, scale)
        mask_scale, _ = image_utils.resize_img(mask, scale)
        if kp is not None:
            assert (vis is not None)
            kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        if self.cfg_aug.rngFlip.rand(1) > 0.5 and self.cfg_aug.flip:
            # Need copy bc torch collate doesn't like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            if kp is not None:
                # Flip kps.
                new_x = img.shape[1] - kp[:, 0] - 1
                kp = np.hstack((new_x[:, None], kp[:, 1:]))
                kp = kp[self.kp_perm, :]
                # kp_uv_flip = kp_uv[self.kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return True, img_flip, mask_flip, kp, sfm_pose
        else:
            return False, img, mask, kp, sfm_pose

    def __len__(self):
        """ This function returns the number of samples in the dataset.
        Number of iterations per epoch is equal to the total number of samples divided by the batch size.
        """
        return self.num_imgs

    def __getitem__(self, index):
        # if index == 1452:
        #     pdb.set_trace()
        datapoints = self.cfg_loader.single_datapoint
        datapoints = [int(x) for x in datapoints]
        if len(datapoints) > 0:
            index = datapoints[index % len(datapoints)]

        flipped, img, kp, mask, sfm_pose, img_path = self.forward_img(index)

        if flipped:
            index = int(1e6) - index

        sfm_pose[0].shape = 1
        elem = {
            'img_path': img_path,
            'img': img,
            'kp': 0 if kp is None else kp,
            # 'kp_uv': kp_uv,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),  # scale (1), trans (2), quat(4)
            'ids': np.array([index]),
            }
        if self.cfg_loader.dataloader_computeMaskDt:
            elem['mask_dt'] = image_utils.compute_dt(mask)
            elem['mask_dt_barrier'] = image_utils.compute_dt_barrier(mask)

        if self.cfg_aug.flip_train:
            # flip_img = self.flip_transform((img.transpose(1,2,0)*255).astype(np.uint8))
            flip_img = img[:, :, ::-1].copy()
            elem['flip_img'] = flip_img
            # elem['flip_img'] = img[:,:,-1::-1].copy()
            # flip_mask = self.flip_transform((mask[None, :, :].transpose(1,2,0)*225).astype(np.uint8))
            flip_mask = mask[:, ::-1].copy()
            elem['flip_mask'] = flip_mask
            if self.cfg_loader.dataloader_computeMaskDt:
                elem['flip_mask_dt'] = image_utils.compute_dt(flip_mask)
                elem['flip_mask_dt_barrier'] = image_utils.compute_dt_barrier(flip_mask)

        if self.cfg_loader.use_cameraPoseDict_as_gt:
            if flipped:
                cams, scores, gt_st1 = self.cameraPoseDict[int(1e6) - index]
                gt_st0 = elem['sfm_pose'][:3] * np.array([1, -1, 1], dtype=np.float32)
            else:
                cams, scores, gt_st1 = self.cameraPoseDict[index]
                gt_st0 = elem['sfm_pose'][:3]
            cams = cams.numpy()
            scores = scores.numpy()
            gt_st1 = gt_st1.numpy()
            cam_id = np.argmax(scores, axis=0)
            best_cam = cams[cam_id]
            scale_factor = gt_st0[0] / gt_st1[0]
            scale = best_cam[0:1] * scale_factor
            trans = (best_cam[1:3] - gt_st1[1:3]) * scale_factor + gt_st0[1:3]
            best_cam = np.concatenate((scale, trans, best_cam[3:]), axis=0)
            if flipped:
                best_cam = best_cam * np.array([1, -1, 1, 1, 1, -1, -1], dtype=np.float32)
            elem['sfm_pose'] = best_cam

        return elem


def collate_fn(batch):
    '''
    Globe data collator.

    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty': True}
    if len(batch) > 0:
        for key in batch[0]:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty'] = False
    return collated_batch


def get_campose_dict(campose_path, is_campose, is_perspective=False):
    '''
    This function handles different file structures:
    1- campose_{epoch}.npz: if camera components dict saved with structure as {frame_id: (cams_px7, scores_p, gt_st_3)}
    as an object of multiplex class.
    2- raw_{epoch}.npz : if camera components are obtained using statistics saved while training.

    Returns campose dict: {frame_id: (cams_px7, scores_p, gt_st_3)}
    '''
    if is_campose:
        try:
            x = np.load(campose_path, allow_pickle=True)
            campose_dict = x['campose'].item()
        except UnicodeError:
            x = np.load(campose_path, allow_pickle=True, encoding='bytes')
            campose_dict = x['campose'].item()
    else:
        x = np.load(campose_path, allow_pickle=True)
        campose_dict = {}
        gt_cam_nx7 = torch.as_tensor(x['camera_gt'])                # (dataset_size, 7)
        cams_nxpx7 = torch.as_tensor(x['cameras_multiplex'])        # (dataset_size, P:num_multipose, 7)
        score_nxp = torch.as_tensor(x['quat_score_multiplex'])      # (dataset_size, P:num_multipose)
        fids_nx1 = torch.as_tensor(x['frame_id'])                   # (dataset_size,)

        if is_perspective:
            gt_cam_flip_nx7 = geom_utils.reflect_cam_pose_perspective(gt_cam_nx7)
            cams_flip_nxpx7 = geom_utils.reflect_cam_pose_perspective(cams_nxpx7)
        else:
            gt_cam_flip_nx7 = geom_utils.reflect_cam_pose(gt_cam_nx7)
            cams_flip_nxpx7 = geom_utils.reflect_cam_pose(cams_nxpx7)

        fids_flip_nx1 = int(1e6) - fids_nx1
        flip = fids_nx1 > int(1e6)/2
        gt_cam_nx7 = torch.where(flip, gt_cam_flip_nx7, gt_cam_nx7)
        cams_nxpx7 = torch.where(flip[:, :, None], cams_flip_nxpx7, cams_nxpx7)
        fids_nx1 = torch.where(flip, fids_flip_nx1, fids_nx1)

        assert ((fids_nx1 >= 0).all())
        assert ((fids_nx1 < int(1e6)/2).all())

        for i in range(fids_nx1.shape[0]):
            fid = int(fids_nx1[i, 0])
            gt_st_3 = gt_cam_nx7[i, 0:3]
            cams_px7 = cams_nxpx7[i, :]
            score_p = score_nxp[i, :]
            campose_dict[fid] = (cams_px7, score_p, gt_st_3)

    return campose_dict


def get_dataloader(configs):

    cfg = configs.dataloader

    anno_path = osp.join(cfg.cub_cache_dir, 'data', f'{cfg.split}_cub_cleaned.mat')
    annotations = sio.loadmat(anno_path, struct_as_record=False, squeeze_me=True)['images']

    anno_sfm_path = osp.join(cfg.cub_cache_dir, 'sfm', f'anno_{cfg.split}.mat')
    sfm_annotations = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

    anno_train_sfm_path = osp.join(cfg.cub_cache_dir, 'sfm', 'anno_train.mat')
    keypoint_3D = sio.loadmat(anno_train_sfm_path, struct_as_record=False, squeeze_me=True)['S'].transpose().copy()

    img_dir = osp.join(cfg.cub_dir, 'images')
    num_imgs = len(annotations)

    Keypoint_names = {'Back':   1, 'Beak':   2, 'Belly':  3, 'Breast': 4, 'Crown':  5,
                      'Fore Head':  6, 'Left Eye':   7, 'Left Leg':   8, 'Left Wing':  9, 'Nape':   10,
                      'Right Eye':   11, 'Right Leg': 12, 'Right Wing': 13, 'Tail':  14, 'Throat': 15
                      }

    print(f'\nThere are {num_imgs} images in the CUB dataset, representing the size of the dataset.')
    print(f'\nThere are {len(keypoint_3D)} keypoints annotated on the category-specific mean shape, applying '
          f'Structure from Motion (SfM), which denotes the following keypoints:\n')

    # Printing keypoints and their IDs in a formatted way
    for keypoint, key_id in Keypoint_names.items():
        print(f'{keypoint}: {key_id}')

    kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1

    cub_dataset = CUBDataset(configs, annotations, sfm_annotations, keypoint_3D, kp_perm, num_imgs, img_dir)

    # The length of the dataloader is the number of batches that will be generated in one epoch.
    cub_dataloader = DataLoader(cub_dataset,
                                batch_size=cfg.batch_size,
                                shuffle=cfg.shuffle,
                                num_workers=cfg.n_data_workers,
                                pin_memory=cfg.pin_memory,
                                drop_last=cfg.drop_last_batch,  # Whether dataloader drop the last incomplete batch
                                                                # if the total number of samples is not divisible
                                                                # by the batch size.
                                collate_fn=collate_fn)

    return cub_dataloader
