
import torch
from loguru import logger

from .crs import *

Model_register_table = {
    'MHIM': MHIMModel
}


def get_model(config, model_name, device, vocab, side_data=None):
    if model_name in Model_register_table:
        model = Model_register_table[model_name](config, device, vocab, side_data)
        logger.info(f'[Build model {model_name}]')
        if config.opt["gpu"] == [-1]:
            return model
        else:
            return torch.nn.DataParallel(model, device_ids=config["gpu"])

    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
