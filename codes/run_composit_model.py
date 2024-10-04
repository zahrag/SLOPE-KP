
import argparse
import numpy as np

from composite_model import CompositeModel
from dataloader import get_dataloader
from io_manager import IOManager
from train import train
from configs.configurations import Configurations


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--exp_configs', type=str, help='Path to the experiment configuration yaml file.',
                        default='', required=True)

    args = parser.parse_args()
    configs = Configurations(args.exp_configs)

    np.random.seed(configs.learning.seed)

    dataloader = get_dataloader(configs)
    configs.dataset_size = len(dataloader.dataset)

    composite_model = CompositeModel(configs)

    io_module = IOManager(configs.learning.output_path)

    # Initialize optimizers for CompositeNetwork and Multiplex
    optimizer_init, mp_optimizer_init = io_module.init_optimizers(composite_model.get_params(),
                                                                  composite_model.multiplex.get_params(),
                                                                  configs.optimizer)

    # Load pretrained CompositeNetwork if exists into CompositeModel.net
    epoch, composite_model, optimizer = io_module.load_pretrained_network(composite_model,
                                                                          configs.learning.pretrained_model_path,
                                                                          optimizer_init)

    # Load pretrained Multiplex if exists into CompositeModel.multiplex
    composite_model, mp_optimizer = io_module.load_pretrained_multiplex(composite_model,
                                                                        configs.learning.pretrained_multiplex_path,
                                                                        mp_optimizer_init)

    # Train CompositeModel
    train(configs.learning, dataloader, composite_model, optimizer, mp_optimizer, io_module, start_epoch=epoch)
