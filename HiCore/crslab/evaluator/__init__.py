

from loguru import logger

from .standard import StandardEvaluator
from ..data import dataset_language_map

Evaluator_register_table = {
    'standard': StandardEvaluator
}


def get_evaluator(evaluator_name, dataset, file_path):
    if evaluator_name in Evaluator_register_table:
        language = dataset_language_map[dataset]
        evaluator = Evaluator_register_table[evaluator_name](language, file_path)
        logger.info(f'[Build evaluator {evaluator_name}]')
        return evaluator
    else:
        raise NotImplementedError(f'Model [{evaluator_name}] has not been implemented')
