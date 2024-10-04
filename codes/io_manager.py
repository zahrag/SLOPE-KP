import torch
import os
import numpy as np
import adam as adam_opt


class IOManager:
    def __init__(self, output_path):
        self.output_path = output_path

    def init_optimizers(self, model_params_dict, multiplex_params_dict, cfg):

        # Network optimizer
        optimizer = adam_opt.Adam(model_params_dict, lr=cfg.learning_rate, betas=(cfg.beta, 0.999))

        # Multiplex optimizer
        if cfg.mp_optimizer == 'sgd':
            mp_optimizer = torch.optim.SGD(multiplex_params_dict)

        elif cfg.mp_optimizer == 'adam':
            mp_optimizer = adam_opt.Adam_NonZeroGradOnly(multiplex_params_dict, betas=(cfg.mp_beta, 0.999))
        else:
            raise ValueError

        return optimizer, mp_optimizer

    def save_model(self, network, optimizer, epoch):
        '''
        Saves the model and optimizer states.
        '''

        # Save the main model and optimizer state
        model_checkpoint = {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(model_checkpoint, os.path.join(self.output_path, 'composite_net_checkpoint_{}.pth'.format(epoch)))

    def save_multiplex(self, network, camera_optimizer, epoch):
        """
        Saves the camera parameters and its optimizer states.
        :param network:
        :param camera_optimizer:
        :param epoch:
        :return:
        """
        # Handle camera parameters for DataParallel
        if isinstance(network, torch.nn.DataParallel):
            camera_params = network.module.dataset_camera_params
        else:
            camera_params = network.dataset_camera_params

        # Save camera parameters and optimizer state
        camera_checkpoint = {
            'epoch': epoch,
            'camera_params_state_dict': camera_params,  # Save camera parameters here
            'camera_optimizer_state_dict': camera_optimizer.state_dict()
        }
        torch.save(camera_checkpoint, os.path.join(self.output_path, 'camera_params_{}.pth'.format(epoch)))

    def save_logs(self, camPoseDict, name, epoch):
        np.savez(os.path.join(self.output_path, '{}_{}.npz'.format(name, epoch)), **camPoseDict)

    def load_pretrained_network(self, model, path, optimizer=None, mode='train'):
        """
        Reloading pretrained networks.
        :param model: The CompositeModel instance to be loaded.
        :param path: Path to the pretrained model.
        :param optimizer: Model optimizer to load the state into.
        :param mode: 'train' or 'test'
        :return:
        """

        if path == '':
            print("No pre-trained model is available!\nLearning from scratch.")
            return 0, model, optimizer  # Start from scratch (epoch 0) if no pre-trained model

        print("Loading pre-trained network from:", path)
        assert os.path.isfile(path)

        # Load the checkpoint
        checkpoint = torch.load(path)

        # Restore the model state
        model.net.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Resume from the saved epoch
        epoch = checkpoint['epoch']

        if mode == 'train':
            # Set the model to evaluation or training mode
            model.net.train()  # Switch to training mode after loading or model.eval() if test mode
        else:
            model.net.eval()

        print('Model and optimizer loaded from {}, starting from epoch {}'.format(path, epoch))

        return epoch, model, optimizer

    import os

    def load_pretrained_multiplex(self, model, path, camera_optimizer=None):
        """
        Load pretrained parameters into the multiplex model.

        :param model: The CompositeModel instance.
        :param path: Path to the pretrained model checkpoint.
        :param camera_optimizer: Optional optimizer to load its state.
        :return: The updated multiplex model.
        """
        # Check if the pretrained path exists
        if not os.path.exists(path):
            print(f"The specified pretrained model path does not exist: {path}")
            return model, camera_optimizer

        print("Loading camera multiplex parameters from:", path)
        camera_checkpoint = torch.load(path)

        # Load camera parameters
        if isinstance(model.multiplex, torch.nn.DataParallel):
            camera_params = model.multiplex.module.dataset_camera_params
        else:
            camera_params = model.multiplex.dataset_camera_params

        for i, param in enumerate(camera_params):
            if i in camera_checkpoint['camera_params_state_dict']:
                param.data.copy_(camera_checkpoint['camera_params_state_dict'][i].data)
            else:
                print(f"Warning: Parameter index {i} not found in checkpoint state dict.")

        # Load camera optimizer state if provided
        if camera_optimizer is not None:
            try:
                camera_optimizer.load_state_dict(camera_checkpoint['camera_optimizer_state_dict'])
            except KeyError as e:
                print(f"Optimizer state dict missing key: {e}")

        print("Camera parameters and optimizer loaded from {}".format(path))

        return model, camera_optimizer


