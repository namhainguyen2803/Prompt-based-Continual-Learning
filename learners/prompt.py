from __future__ import print_function
import math
import random

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
from .CPL import ContrastivePrototypicalLoss
from models.emb_proj import EmbeddingProjection
from models.clustering_algorithm import KMeans
import bitsandbytes as bnb

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:, :self.valid_out_dim]

        # ce with heuristic
        logits[:, :self.last_valid_out_dim] = -float('inf')
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
        optimizer_arg = {'params': params_to_opt,
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)
        elif self.config["optimizer"] == "AdamW":
            optimizer_arg["betas"] = (0.9, 0.999)

        # create optimizers
        self.optimizer = bnb.optim.__dict__[self.config['optimizer']+'8bit'](**optimizer_arg)

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
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'],
                                               output_device=self.config['gpuid'][0])
        return self


# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='coda',
                                                                               prompt_param=self.prompt_param)
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
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='dual',
                                                                               prompt_param=self.prompt_param)
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
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='l2p',
                                                                               prompt_param=self.prompt_param)
        return model


class ContrastivePrototypicalPrompt(Prompt):

    def __init__(self, learner_config):
        super(ContrastivePrototypicalPrompt, self).__init__(learner_config)

        self.key_prototype = dict()
        self.value_prototype = dict()
        self.avg_variance = dict()
        self.MLP_neck = None
        self._num_anchor_value_prototype_per_class = 5
        self._num_anchor_key_prototype_per_class = 5
        self._create_mapping_from_class_to_task()
        self.first_task = True

        self.verbose = True
        self.print_every = 10
        self.scaler = torch.cuda.amp.GradScaler()
    def _create_criterion_fn(self):
        self.criterion_fn = ContrastivePrototypicalLoss(temperature=0.6, reduction="mean")

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='cpp',
                                                                               prompt_param=self.prompt_param)
        return model

    def _update_prototype_set(self, prototype_set, train_loader, use_prompt=False):
        """
        Function to update prototype of previous class.
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

                if use_prompt:
                    # can have previous tasks(?)
                    last_feature, _ = self.model(x, pen=True, train=False, use_prompt=True,
                                                 possible_task_id=task.reshape(-1, 1))
                else:
                    last_feature, _ = self.model(x, pen=True, train=False, use_prompt=False)

                list_last_feature.append(last_feature)
                list_output.append(y)

            last_features = torch.cat(list_last_feature, dim=0)
            outputs = torch.cat(list_output, dim=0)
            uni_output = sorted(torch.unique(outputs).tolist())
            for class_id in uni_output:
                if use_prompt:
                    cluster_algorithm = KMeans(num_classes=self._num_anchor_value_prototype_per_class)
                else:
                    cluster_algorithm = KMeans(num_classes=self._num_anchor_key_prototype_per_class)
                feature_set_for_class_id = last_features[outputs == class_id]
                assert feature_set_for_class_id.ndim == 2, "feature_set_for_class_id.ndim != 2."
                cluster_algorithm.fit(feature_set_for_class_id)
                prototype = cluster_algorithm.get_centroids()
                prototype_set[class_id] = prototype  # (_num_anchor_per_class, emb_d)
                check_tensor_nan(prototype, "prototype")
                check_tensor_nan(feature_set_for_class_id, "feature_set_for_class_id")
                if use_prompt:
                    # row_variances = torch.var(feature_set_for_class_id, dim=1)
                    # self.avg_variance[class_id] = torch.mean(row_variances)
                    # print(self.avg_variance[class_id])
                    self.avg_variance[class_id] = torch.tensor(1.0)
            return prototype_set

    def _update_key_prototype(self, train_loader):
        self.key_prototype = self._update_prototype_set(prototype_set=self.key_prototype, train_loader=train_loader,
                                                        use_prompt=False)

    def _update_value_prototype(self, train_loader):
        self.value_prototype = self._update_prototype_set(prototype_set=self.value_prototype, train_loader=train_loader,
                                                          use_prompt=True)

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, need_loss=True, need_acc=False):
        print("##### Attempt to update key prototype set. #####")
        self._update_key_prototype(train_loader)
        print("##### Finish updating key prototype set. #####")
        # re-initialize MLP neck
        self._reset_MLP_neck()
        print("Reset MLP neck.")
        # learn prompt
        print(f"##### Attempt to learn batch in task id: {self.model.task_id}. #####")
        self._learn_batch(train_loader, train_dataset, model_save_dir, val_loader=val_loader, need_loss=need_loss)
        print(f"##### Finish learning batch in task id: {self.model.task_id}. #####")
        print("##### Attempt to update value prototype set. #####")
        self._update_value_prototype(train_loader)
        print("##### Finish updating value prototype set. #####")

    def _learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, need_loss=True):
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                print("Cannot load model")
        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            if need_loss:
                losses = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            all_previous_value_prototype = None
            avg_var = None
            if not self.first_task:

                # retrieve all perturbed prototype set in a single tensor
                all_previous_value_prototype = list()
                for class_id, value_prototype_set in self.value_prototype.items():
                    if value_prototype_set.ndim == 1:
                        value_prototype_set = value_prototype_set.unsqueeze(0)
                    assert value_prototype_set.ndim == 2, "all_previous_value_prototype.ndim != 2."
                    all_previous_value_prototype.append(value_prototype_set)
                all_previous_value_prototype = torch.cat(all_previous_value_prototype, dim=0)
                assert all_previous_value_prototype.ndim == 2, "all_previous_value_prototype.ndim != 2."
                # all_previous_value_prototype = nn.functional.normalize(all_previous_value_prototype, dim=1)
                print(f"Check value_prototype, having shape: {all_previous_value_prototype.shape}, "
                      f"requires grad: {all_previous_value_prototype.requires_grad}")

                avg_var = list()
                for class_id, avg_var_for_each_class in self.avg_variance.items():
                    if class_id < self.last_valid_out_dim:
                        avg_var.append(avg_var_for_each_class)  # avg_var_for_each_class is a number
                avg_var = torch.tensor(avg_var)
                assert avg_var.shape[0] * self._num_anchor_value_prototype_per_class == all_previous_value_prototype.shape[0]
                # stretch avg_var to be the same size as prototype.shape[0]
                avg_var = avg_var.repeat(self._num_anchor_value_prototype_per_class).unsqueeze(-1).cuda()

            for epoch in range(self.config['schedule'][-1]):
                self.epoch = epoch
                if epoch > 0:
                    self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    # verify in train mode
                    self.model.train()
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    # model update
                    loss = self.update_model(x, y, all_previous_value_prototype, avg_var)
                    # print(loss)
                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    # measure accuracy and record loss
                    y = y.detach()
                    if need_loss:
                        losses.update(loss, y.size(0))
                    batch_timer.tic()
                # eval update
                self.log(
                    'Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch + 1, total=self.config['schedule'][-1]))
                if need_loss:
                    self.log(' * Loss {loss.avg:.3f} |'.format(loss=losses))
                    losses = AverageMeter()
                    if self.verbose == True and epoch % self.print_every == 0:
                        print(f"##### Validation time in epoch: {epoch} #####")
                        self._update_value_prototype(train_loader)
                        loss = self._calculate_validation_loss(val_loader, all_previous_value_prototype, avg_var)
                        acc = self.validation(dataloader=val_loader, model=None, task_in=None, task_metric='acc', verbal=True)
                        print(f"Accuracy in validation: {acc}, loss value: {loss}")
                        print("##### End validation #####")
        self.model.eval()
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))
        try:
            return batch_time.avg
        except:
            return None

    def _calculate_validation_loss(self, train_loader, all_previous_value_prototype, avg_var):
        with torch.no_grad():
            total_loss = 0
            total_element = 0
            for i, (x, y, task) in enumerate(train_loader):
                # verify in train mode
                self.model.train()
                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                if not self.first_task:
                    all_previous_value_prototype = self._perturb_value_prototype(all_previous_value_prototype, avg_var)
                    all_previous_value_prototype = nn.functional.normalize(all_previous_value_prototype, dim=1)
                    check_tensor_nan(all_previous_value_prototype, "all_previous_value_prototype (1)")
                with torch.cuda.amp.autocast():
                    last_feature, _ = self.model(x, pen=True, train=False,
                                                            use_prompt=True, possible_task_id = task.reshape(-1, 1))
                    check_tensor_nan(last_feature, "last_feature")
                
                    z_feature = self.MLP_neck(last_feature)
                    n_z_feature = nn.functional.normalize(z_feature, dim=1)
                    loss = self.criterion_fn(z_feature=n_z_feature, label=y,
                                                previous_prototype=all_previous_value_prototype)
                
                total_loss += loss.detach()
                total_element = x.shape[0]
            return total_loss / total_element

    def update_model(self, inputs, targets, all_previous_value_prototype=None, avg_var=None):
        # logits
        if not self.first_task:
            if avg_var is not None and all_previous_value_prototype is not None:
                all_previous_value_prototype = self._perturb_value_prototype(all_previous_value_prototype, avg_var)
            all_previous_value_prototype = nn.functional.normalize(all_previous_value_prototype, dim=1)
            check_tensor_nan(all_previous_value_prototype, "all_previous_value_prototype")
        with torch.cuda.amp.autocast():
            last_feature, _, prompt_loss = self.model(inputs, pen=True, train=True, use_prompt=True)
            check_tensor_nan(last_feature, "last_feature")
            z_feature = self.MLP_neck(last_feature)
            n_z_feature = nn.functional.normalize(z_feature, dim=1)
            total_loss = self.criterion_fn(z_feature=n_z_feature, label=targets,
                                        previous_prototype=all_previous_value_prototype)
        # step
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)

        return total_loss.detach()

    def _perturb_value_prototype(self, prototype, avg_var):
        with torch.no_grad():
            vect_dim = prototype.shape[1]
            num_instances = prototype.shape[0]
            mean = torch.zeros(vect_dim)
            covariance = torch.eye(vect_dim)
            gaussian_noise = torch.distributions.MultivariateNormal(mean, covariance).sample([num_instances]).cuda()
            return prototype + avg_var * gaussian_noise

    def _reset_MLP_neck(self):
        if self.MLP_neck is not None:
            del self.MLP_neck
        self.MLP_neck = EmbeddingProjection().cuda()

    def _learnable_params(self):
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.MLP_neck.module.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.MLP_neck.parameters())
        return params_to_opt

    def validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True):
        with torch.no_grad():
            if model is None:
                model = self.model
            # This function doesn't distinguish tasks.
            batch_timer = Timer()
            acc = AverageMeter()
            batch_timer.tic()
            orig_mode = model.training
            model.eval()
            U = list()
            U_hat = list()
            for class_id in range(self.valid_out_dim):
                key = self.key_prototype[class_id].unsqueeze(0)
                value = self.value_prototype[class_id].unsqueeze(0)
                U.append(key)
                U_hat.append(value)
            U = torch.cat(U, dim=0)  # (num_classes, num_anchors, emb_d)
            U_hat = torch.cat(U_hat, dim=0)
            assert U.ndim == 3, "Wrong in shape U."
            assert U_hat.ndim == 3, "Wrong in shape U_hat."
            print(f"Shape of U: {U.shape}, Shape of U_hat: {U_hat.shape}")
            check_tensor_nan(U, "U")
            check_tensor_nan(U_hat, "U_hat")
            total_correct = 0
            total_element = 0
            for i, (input, target, task) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                if task_in is None:
                    acc, correct_task, num_element = self._evaluate_CPP(U=U, U_hat=U_hat, model=model, input=input,
                                                                        target=target, task=task, acc=acc, task_in=None)
                else:
                    mask = target >= task_in[0]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]
                    mask = target < task_in[-1]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]
                    acc, correct_task, num_element = self._evaluate_CPP(U=U, U_hat=U_hat, model=model, input=input,
                                                                        target=target, task=task, acc=acc, task_in=task_in)
                total_correct += correct_task
                total_element += num_element
        model.train(orig_mode)
        if verbal:
            ground_truth_task = torch.unique(task).cuda()
            self.log(f"In task {ground_truth_task}, "
                  f"number of correct task: {total_correct} in {total_element} elements")
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                     .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def _create_mapping_from_class_to_task(self):
        self.mapping_class_to_task = dict()
        for task_id, class_range in enumerate(self.tasks):
            for class_id in class_range:
                self.mapping_class_to_task[class_id] = task_id

    def _evaluate_CPP(self, U, U_hat, model, input, target, task, acc, task_in=None):
        with torch.no_grad():
            top_k = model.prompt.top_k
            # retrieve prototype set in a tensor with ascending order wrt class_id
            x_query = model.retrieve_query_vector(input)
            B, C = x_query.shape
            # cosine similarity to match keys/queries
            n_U = nn.functional.normalize(U, dim=2)  # (num_classes, num_anchors, emb_d)
            q = nn.functional.normalize(x_query, dim=1).detach()  # (B, emb_d)
            cos_sim = torch.einsum('kj,bij->kbi', q, n_U)  # (B, num_classes, num_anchors)
            flatten_cos_sim = cos_sim.reshape(B, -1)  # (B, num_classes * num_anchors)
            prototype_id_ranking = torch.topk(flatten_cos_sim, top_k, dim=1)
            ranking = prototype_id_ranking.indices  # shape == (B, self.top_k)
            possible_task_id = torch.zeros_like(ranking).cuda()

            for class_id in range(self.valid_out_dim):
                # [0, 5]
                class_range = (class_id * self._num_anchor_key_prototype_per_class,
                               (class_id + 1) * self._num_anchor_key_prototype_per_class)
                for c in range(class_range[0], class_range[1]):
                    possible_task_id[ranking == c] = self.mapping_class_to_task[class_id]

            diff = possible_task_id - task.unsqueeze(1).cuda()
            same = torch.zeros_like(diff).cuda()
            same[diff == 0] = 1
            same[diff != 0] = 0
            same = torch.sum(same, dim=1)
            same[same > 1] = 1

            num_element_correct_task = torch.sum(same)

            flatten_possible_task_id = possible_task_id.reshape(-1, 1)  # flatten, shape == (B * self.top_k, 1)

            inp = input.unsqueeze(0)
            input_repeat = inp.repeat(top_k, 1, 1, 1, 1)
            input_repeat = input_repeat.permute(1, 0, 2, 3, 4)
            input_repeat = input_repeat.reshape(-1, input_repeat.shape[2], input_repeat.shape[3], input_repeat.shape[4])

            # print(f"shape of input_repeat: {input_repeat.shape}")
            last_feature, _ = self.model(input_repeat, pen=True, train=False, use_prompt=True,
                                         possible_task_id=flatten_possible_task_id)
            # last_feature.shape == (B * self.top_k, emb_d)
            # print(f"shape of last_feature: {last_feature.shape}")
            assert last_feature.shape == (B * top_k, self.model.prompt.emb_d), \
                "last_feature.shape != (B * top_k, self.model.prompt.emb_d)."
            fine_grained_query = last_feature.reshape(B, top_k, self.model.prompt.emb_d)

            n_U_hat = nn.functional.normalize(U_hat, dim=2)  # (num_classes, num_anchors, emb_d)
            n_fine_grained_query = nn.functional.normalize(fine_grained_query, dim=-1)  # (B, top_k, emb_d)
            assert n_fine_grained_query.shape == (B, top_k, self.model.prompt.emb_d), "Wrong in _evaluate method (2)."

            # likelihood_among_top_k_classes.shape == (B, num_classes, top_k, num_anchors)
            likelihood_among_top_k_classes = torch.einsum('bij,tkj->btik', n_fine_grained_query, n_U_hat)
            likelihood_among_top_k_classes = likelihood_among_top_k_classes.reshape(B, self.valid_out_dim, -1)
            max_likelihood_among_k_classes = torch.max(likelihood_among_top_k_classes, dim=-1).values
            assert max_likelihood_among_k_classes.shape == (B, self.valid_out_dim), "Wrong in _evaluate method (3)."

            if task_in is None:
                output = max_likelihood_among_k_classes
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                output = max_likelihood_among_k_classes[:, task_in]
                acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))
            return acc, num_element_correct_task, B



def check_tensor_nan(tensor, tensor_name="a"):
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        raise f"Tensor {tensor_name} is nan."