

import os
from abc import ABC, abstractmethod
import numpy as np
import random
import torch
from loguru import logger
from torch import optim
from transformers import AdamW, Adafactor

from crslab.config import SAVE_PATH
from crslab.evaluator import get_evaluator
from crslab.evaluator.metrics.base import AverageMetric
from crslab.model import get_model
from crslab.system.utils import lr_scheduler
from crslab.system.utils.functions import compute_grad_norm

optim_class = {}
optim_class.update({k: v for k, v in optim.__dict__.items() if not k.startswith('__') and k[0].isupper()})
optim_class.update({'AdamW': AdamW, 'Adafactor': Adafactor})
lr_scheduler_class = {k: v for k, v in lr_scheduler.__dict__.items() if not k.startswith('__') and k[0].isupper()}
transformers_tokenizer = ('bert', 'gpt2')


class BaseSystem(ABC):
    """Base class for all system"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.

        """
        self.opt = opt
        if opt["gpu"] == [-1]:
            self.device = torch.device('cpu')
        elif len(opt["gpu"]) == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cuda')
        # seed
        if 'seed' in opt:
            seed = int(opt['seed'])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            logger.info(f'[Set seed] {seed}')
        # data
        if debug:
            self.train_dataloader = valid_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        else:
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        self.vocab = vocab
        self.side_data = side_data
        # model
        if 'model' in opt:
            self.model = get_model(opt, opt['model'], self.device, vocab, side_data).to(self.device)
        else:
            if 'rec_model' in opt:
                self.rec_model = get_model(opt, opt['rec_model'], self.device, vocab['rec'], side_data['rec']).to(
                    self.device)
            if 'conv_model' in opt:
                self.conv_model = get_model(opt, opt['conv_model'], self.device, vocab['conv'], side_data['conv']).to(
                    self.device)
            if 'policy_model' in opt:
                self.policy_model = get_model(opt, opt['policy_model'], self.device, vocab['policy'],
                                              side_data['policy']).to(self.device)
        model_file_name = opt.get('model_file', f'{opt["model_name"]}.pth')
        self.model_file = os.path.join(SAVE_PATH, model_file_name)
        if restore_system:
            self.restore_model()

        if not interact:
            self.evaluator = get_evaluator('standard', opt['dataset'], opt['rankfile'])

    def init_optim(self, opt, parameters):
        self.optim_opt = opt
        parameters = list(parameters)
        if isinstance(parameters[0], dict):
            for i, d in enumerate(parameters):
                parameters[i]['params'] = list(d['params'])

        # gradient acumulation
        self.update_freq = opt.get('update_freq', 1)
        self._number_grad_accum = 0

        self.gradient_clip = opt.get('gradient_clip', -1)

        self.build_optimizer(parameters)
        self.build_lr_scheduler()

        if isinstance(parameters[0], dict):
            self.parameters = []
            for d in parameters:
                self.parameters.extend(d['params'])
        else:
            self.parameters = parameters

        # early stop
        self.need_early_stop = self.optim_opt.get('early_stop', False)
        if self.need_early_stop:
            logger.debug('[Enable early stop]')
            self.reset_early_stop_state()

    def build_optimizer(self, parameters):
        optimizer_opt = self.optim_opt['optimizer']
        optimizer = optimizer_opt.pop('name')
        self.optimizer = optim_class[optimizer](parameters, **optimizer_opt)
        logger.info(f"[Build optimizer: {optimizer}]")

    def build_lr_scheduler(self):
        """
        Create the learning rate scheduler, and assign it to self.scheduler. This
        scheduler will be updated upon a call to receive_metrics. May also create
        self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        if self.optim_opt.get('lr_scheduler', None):
            lr_scheduler_opt = self.optim_opt['lr_scheduler']
            lr_scheduler = lr_scheduler_opt.pop('name')
            self.scheduler = lr_scheduler_class[lr_scheduler](self.optimizer, **lr_scheduler_opt)
            logger.info(f"[Build scheduler {lr_scheduler}]")

    def reset_early_stop_state(self):
        self.best_valid = None
        self.drop_cnt = 0
        self.impatience = self.optim_opt.get('impatience', 3)
        if self.optim_opt['stop_mode'] == 'max':
            self.stop_mode = 1
        elif self.optim_opt['stop_mode'] == 'min':
            self.stop_mode = -1
        else:
            raise
        logger.debug('[Reset early stop state]')

    @abstractmethod
    def fit(self):
        """fit the whole system"""
        pass

    @abstractmethod
    def step(self, batch, stage, mode):
        """calculate loss and prediction for batch data under certrain stage and mode

        Args:
            batch (dict or tuple): batch data
            stage (str): recommendation/policy/conversation etc.
            mode (str): train/valid/test
        """
        pass

    def backward(self, loss):
        """empty grad, backward loss and update params

        Args:
            loss (torch.Tensor):
        """
        self._zero_grad()

        if self.update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % self.update_freq
            loss /= self.update_freq
        loss.backward(loss.clone().detach())

        self._update_params()

    def _zero_grad(self):
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return
        self.optimizer.zero_grad()

    def _update_params(self):
        if self.update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, self.gradient_clip
            )
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm))
            self.evaluator.optim_metrics.add(
                'grad clip ratio',
                AverageMetric(float(grad_norm > self.gradient_clip)),
            )
        else:
            grad_norm = compute_grad_norm(self.parameters)
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm))

        self.optimizer.step()

        if hasattr(self, 'scheduler'):
            self.scheduler.train_step()

    def adjust_lr(self, metric=None):
        """adjust learning rate w/o metric by scheduler

        Args:
            metric (optional): Defaults to None.
        """
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return
        self.scheduler.valid_step(metric)
        logger.debug('[Adjust learning rate after valid epoch]')

    def early_stop(self, metric):
        if not self.need_early_stop:
            return False
        if self.best_valid is None or metric * self.stop_mode > self.best_valid * self.stop_mode:
            self.best_valid = metric
            self.drop_cnt = 0
            logger.info('[Get new best model]')
            return False
        else:
            self.drop_cnt += 1
            if self.drop_cnt >= self.impatience:
                logger.info('[Early stop]')
                return True

    def save_model(self):
        r"""Store the model parameters."""
        state = {}
        if hasattr(self, 'model'):
            state['model_state_dict'] = self.model.state_dict()
        if hasattr(self, 'rec_model'):
            state['rec_state_dict'] = self.rec_model.state_dict()
        if hasattr(self, 'conv_model'):
            state['conv_state_dict'] = self.conv_model.state_dict()
        if hasattr(self, 'policy_model'):
            state['policy_state_dict'] = self.policy_model.state_dict()

        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(state, self.model_file)
        logger.info(f'[Save model into {self.model_file}]')

    def restore_model(self):
        r"""Store the model parameters."""
        if not os.path.exists(self.model_file):
            raise ValueError(f'Saved model [{self.model_file}] does not exist')
        checkpoint = torch.load(self.model_file, map_location=self.device)
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'rec_model'):
            self.rec_model.load_state_dict(checkpoint['rec_state_dict'])
        if hasattr(self, 'conv_model'):
            self.conv_model.load_state_dict(checkpoint['conv_state_dict'])
        if hasattr(self, 'policy_model'):
            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
        logger.info(f'[Restore model from {self.model_file}]')

    @abstractmethod
    def interact(self):
        pass
