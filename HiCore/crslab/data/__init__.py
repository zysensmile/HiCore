
from crslab.data.dataloader import *
from crslab.data.dataset import *

dataset_register_table = {
    'HReDial': HReDialDataset,
    'HTGReDial': HTGReDialDataset,
    'OpenDialKG': OpenDialKGDataset,
    'DuRecDial': DuRecDialDataset,
}

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'HReDial': 'en',
    'HTGReDial': 'zh',
    'OpenDialKG': 'en',
    'DuRecDial': 'zh',
}

dataloader_register_table = {
    'MHIM': MHIMDataLoader
}


def get_dataset(opt, tokenize, restore, save) -> BaseDataset:
    """get and process dataset

    Args:
        opt (Config or dict): config for dataset or the whole system.
        tokenize (str): how to tokenize the dataset.
        restore (bool): whether to restore saved dataset which has been processed.
        save (bool): whether to save dataset after processing.

    Returns:
        processed dataset

    """
    dataset = opt['dataset']
    if dataset in dataset_register_table:
        return dataset_register_table[dataset](opt, tokenize, restore, save)
    else:
        raise NotImplementedError(f'The dataloader [{dataset}] has not been implemented')


def get_dataloader(opt, dataset, vocab) -> BaseDataLoader:
    """get dataloader to batchify dataset

    Args:
        opt (Config or dict): config for dataloader or the whole system.
        dataset: processed raw data, no side data.
        vocab (dict): all kinds of useful size, idx and map between token and idx.

    Returns:
        dataloader

    """
    model_name = opt['model_name']
    if model_name in dataloader_register_table:
        return dataloader_register_table[model_name](opt, dataset, vocab)
    else:
        raise NotImplementedError(f'The dataloader [{model_name}] has not been implemented')
