
import argparse

import torch
from init_multiplex import MultiplexOptimizer
import numpy as np
from tqdm import tqdm

from dataloader import get_dataloader
import adam as adam_opt
from configs.configurations import Configurations


def train(cfg, data_loader, model):

    parameters = [torch.nn.Parameter(torch.tensor(0.))]
    optimizer = adam_opt.Adam(parameters,
                              lr=cfg.optimizer.learning_rate,
                              betas=(cfg.optimizer.beta, 0.999))

    for ep in range(cfg.learning.max_epoch):
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='train', position=0)):

            optimizer.zero_grad()
            loss_dict = model.forward(cfg, batch)
            total_loss = loss_dict['total_loss']
            total_loss.backward()

        if (ep + 1) % configs.learning.save_interval == 0:
            model.save_datasetCameraPoseDict(ep)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--exp_configs', type=str, help='Path to the experiment configuration yaml file.',
                        default='', required=True)

    args = parser.parse_args()

    configs = Configurations(args.exp_configs)

    dataloader = get_dataloader(configs)

    np.random.seed(configs.learning.seed)

    cam_init = MultiplexOptimizer(configs)

    train(configs, dataloader, cam_init)
