import os
import yaml


class Configurations:
    def __init__(self, exp_config_path):

        cfg_dict = self.load_exp_config(exp_config_path)
        cfg = DictToObj(cfg_dict)

        self.dataset_path = cfg.ConfigBase.dataset_path
        self.output_path = cfg.ConfigBase.output_path
        self.shape_path = cfg.ConfigBase.shape_path
        self.pretrained_model_path = cfg.ConfigBase.pretrained_model_path
        self.pretrained_multiplex_path = cfg.ConfigBase.pretrained_multiplex_path
        self.img_size = cfg.ConfigBase.img_size
        self.BGR_MEAN = cfg.ConfigBase.BGR_MEAN
        self.BGR_STD = cfg.ConfigBase.BGR_STD

        self.dataloader = cfg.ConfigDataloader
        self.dataloader.dataset_path = self.dataset_path
        self.dataloader.cub_dir = os.path.join(self.dataset_path, self.dataloader.dataset)
        self.dataloader.cub_cache_dir = os.path.join(self.dataset_path, 'cub')

        self.augmentation = cfg.ConfigAugmentation
        self.augmentation.img_size = self.img_size

        self.multiplex = cfg.ConfigMultiplex
        self.multiplex.num_multipose = 1 if self.multiplex.use_gt_camera else (self.multiplex.num_multiposeAz *
                                                                               self.multiplex.num_multiposeEl *
                                                                               self.multiplex.num_multiposeCr)

        self.renderer = cfg.ConfigRenderer
        self.renderer.img_size = self.img_size

        self.net = cfg.ConfigNet
        self.net.img_size = self.img_size

        self.shape = cfg.ConfigShape
        self.shape.shape_path = self.shape_path

        self.pose = cfg.ConfigPose
        self.texture = cfg.ConfigTexture

        self.loss = cfg.ConfigLoss
        self.loss.quatScorePeakiness = self.multiplex.quatScorePeakiness

        self.optimizer = cfg.ConfigOptimizer

        self.learning = cfg.ConfigLearning
        self.learning.pretrained_model_path = self.pretrained_model_path
        self.learning.pretrained_multiplex_path = self.pretrained_multiplex_path
        self.learning.output_path = self.output_path

    def load_exp_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)


class DictToObj:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))  # Recursively convert dict to object
            else:
                setattr(self, key, value)




