from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from models.vit import Mlp
from .SelfSupervisedContrastiveLoss import SelfSupConLoss

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def _learnable_params(self):
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        return params_to_opt

    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        params_to_opt = self._learnable_params()
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model

class ContrastivePrototypicalPrompt(Prompt):

    def __init__(self, learner_config):
        super(ContrastivePrototypicalPrompt, self).__init__(learner_config)

        self.key_prototype = dict()
        self.perturbed_prototype = dict()
        self.MLP_neck = None

        self._generate_mapping_class_to_task() # generate mapping from class_id to task_id, used for evaluation

    def _create_criterion_fn(self):
        self.criterion_fn = SelfSupConLoss()

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'cpp',prompt_param=self.prompt_param)
        return model

    def _update_prototype_set(self, prototype_set, train_loader, is_perturbed=False):
        """
        Function to update prototype of previous class. Only update prototype after updating feature_extractor(prompt)
        """
        with torch.no_grad():
            list_last_feature = list()
            list_output = list()
            for i, (x, y, task) in enumerate(train_loader):
                self.model.eval()

                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                if is_perturbed: # if update perturbed prototype then USE PROMPT
                    last_feature = self.model(x, pen=True, train=False, use_prompt=True)
                    # MAKE SURE THAT SELF.MLP_NECK IS PROPERLY UPDATED
                    last_feature = self.MLP_neck(last_feature)

                else: # if update key prototype then DO NOT USE PROMPT
                    last_feature = self.model(x, pen=True, train=False, use_prompt=False)

                list_last_feature.append(last_feature)
                list_output.append(y)

            last_features = torch.cat(list_last_feature, dim=0) # all feature vectors in train_loader
            outputs = torch.cat(list_output, dim=0) # corresponding output of all feature vectors
            uni_output = torch.unique(outputs) # retrieve all class_id in train_loader
            for class_id in uni_output:
                prototype = torch.mean(last_features[uni_output == class_id], dim=0) # calculate prototype by mean of vectors
                if is_perturbed: # perturb prototype with Unit Multivariate Gaussian
                    prototype = prototype + torch.randn_like(prototype)
                prototype_set[class_id] = prototype
            return prototype_set

    def _update_key_prototype(self, train_loader):
        self._update_prototype_set(self.key_prototype, train_loader, False)

    def _update_perturbed_prototype(self, train_loader):
        self._update_prototype_set(self.perturbed_prototype, train_loader, True)

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # update key prototype (not include prompt)
        self._update_key_prototype(train_loader)
        # re-initialize MLP neck
        self._reset_MLP_neck()
        # learn prompt
        super().learn_batch(train_loader, train_dataset, model_save_dir, val_loader=None)
        # update perturbed prototype set after learning prompt and MLP_neck
        self._update_perturbed_prototype(train_loader)

    def update_model(self, inputs, targets):
        """
        Modify update_model method because CPP has different loss function compared to L2P or DualPrompt.
        Specifically, it uses Contrastive Loss Function as criterion
        """
        # logits
        last_feature, logits, prompt_loss = self.model(inputs, pen=True, train=True, use_prompt=True)
        z_feature = self.MLP_neck(last_feature)

        # retrieve all perturbed prototype set in a single tensor
        all_previous_perturbed_prototype = list()
        all_previous_prototype_labels = list()
        for class_id, perturbed_set in self.perturbed_prototype:
            all_previous_perturbed_prototype.append(perturbed_set)
            all_previous_prototype_labels.append([class_id] * perturbed_set.shape[0])
        all_previous_perturbed_prototype = torch.cat(all_previous_perturbed_prototype, dim=0)
        all_previous_prototype_labels = torch.cat(all_previous_prototype_labels, dim=0)
        z_feature = torch.cat((z_feature, all_previous_perturbed_prototype), dim=0)
        labels_of_z_feature = torch.cat((targets, all_previous_prototype_labels), dim=0)

        contrastive_loss = self.criterion_fn(features=z_feature, labels=labels_of_z_feature, target_labels=targets)
        total_loss = contrastive_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits
    def _reset_MLP_neck(self):
        if self.MLP_neck is not None:
            del self.MLP_neck
        self.MLP_neck = Mlp(in_features=768, hidden_features=2048, out_features=768, act_layer=nn.ReLU, drop=0.).cuda()

    def _learnable_params(self):
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.MLP_neck.module.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.MLP_neck.parameters())
        return params_to_opt

    def _generate_mapping_class_to_task(self):
        self.mapping_class_to_task = dict()
        for task_id, class_id_list in enumerate(self.tasks):
            for class_id in class_id_list:
                self.mapping_class_to_task[class_id] = task_id

    def _evaluate(self, model, input, target, task, acc, task_in=None):
        with torch.no_grad():
            # retrieve prototype set in a tensor with ascending order wrt class_id
            class_id_so_far = sorted(self.key_prototype.keys())
            U = list()
            U_hat = list()
            for class_id in class_id_so_far:
                U.append(self.key_prototype[class_id])
                U_hat.append(self.perturbed_prototype[class_id])
            U = torch.cat(U, dim=0)
            U_hat = torch.cat(U_hat, dim=0) # shape == (num_classes, self.emb_d)

            x_query = self.model.retrieve_query_vector(input) # query of input, shape == (B, self.emb_d)
            B, C = x_query.shape
            # cosine similarity to match keys/queries
            n_U = nn.functional.normalize(U, dim=1)  # shape == (number of classes, self.key_d)
            q = nn.functional.normalize(x_query, dim=1).detach()  # shape == (B, self.emb_d)
            cos_sim = torch.einsum('bj,kj->bk', q, n_U)

            top_k = torch.topk(cos_sim, self.model.prompt.top_k, dim=1)
            class_idx = top_k.indices  # shape == (B, self.top_k)
            # have already had k class_id, we need to find their corresponding task to retrieve task-specific prompt
            possible_task_id = torch.zeros_like(class_idx)
            # here, we map each class_id to its corresponding task_id via mapping class_to_task
            # prototype.shape[0] is the number of classes seen so far
            for cid in range(U.shape[0]):
                possible_task_id[class_idx == cid] = self.mapping_class_to_task[cid]

            fine_grained_query = list()
            top_k = self.model.prompt.top_k
            for top in range(top_k):
                # last_feature will have shape (B, self.emb_d)
                last_feature, _ = self.model(input, pen=True, train=False, use_prompt=True, possible_task_id=possible_task_id[:, top].view(-1, 1))
                assert last_feature.shape == (B, self.model.prompt.emb_d), "Wrong in _evaluate method (1)."
                last_feature = last_feature.unsqueeze(1) # have shape (B, 1, self.emb_d)
                fine_grained_query.append(last_feature)
            fine_grained_query = torch.cat(fine_grained_query, dim=1) # have shape (B, self.top_k, self.emb_d)

            n_U_hat = nn.functional.normalize(U_hat, dim=1)  # shape == (number of classes, self.emb_d)
            n_fine_grained_query = nn.functional.normalize(fine_grained_query, dim=-1)
            assert n_fine_grained_query.shape == (B, top_k, self.model.prompt.emb_d), "Wrong in _evaluate method (2)."

            likelihood_among_top_k_classes = torch.einsum('bij,kj->bki', n_fine_grained_query, n_U_hat)
            # likelihood_among_top_k_classes.shape == (B, num_classes, self.model.prompt.top_k)
            max_likelihood_among_k_classes = torch.max(likelihood_among_top_k_classes, dim=-1)
            num_classes = likelihood_among_top_k_classes.shape[1]
            assert max_likelihood_among_k_classes.shape == (B, num_classes), "Wrong in _evaluate method (3)."

            decision = torch.argmax(max_likelihood_among_k_classes, dim=1)
            assert decision.shape == B, "Wrong in _evaluate method (4)."

            if task_in is None:
                output = max_likelihood_among_k_classes[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                output = max_likelihood_among_k_classes[:, task_in]
                acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))
            return acc
