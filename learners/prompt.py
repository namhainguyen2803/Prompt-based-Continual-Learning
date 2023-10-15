from __future__ import print_function

import random

import numpy as np
import torch
import torch.nn as nn
import models
from models.ContrastiveLoss import ContrastivePrototypicalLoss
from models.LearnDistribution import get_learning_distribution_model
from utils.metric import AverageMeter, Timer
from utils.schedulers import CosineSchedule
from .default import NormalNN, accumulate_acc
from models.ClusterAlgorithm import KMeans, fit_kmeans_many_times
from models.EmbeddingProjection import EmbeddingMLP, MLP

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.prompt_type = learner_config['prompt_type']

        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        logit, _, prompt_loss = self.model(x=inputs, get_logit=True, train=True,
                                           use_prompt=True, task_id=None, prompt_type=self.prompt_type)
        logit = logit[:, :self.valid_out_dim]

        # ce with heuristic
        # logit[:, :self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logit[:, self.last_valid_out_dim:], (targets - self.last_valid_out_dim).long())

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logit

    def _set_learnable_parameter(self):
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        return params_to_opt

    def init_optimizer(self):
        params_to_opt = self._set_learnable_parameter()
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
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'],
                                               output_device=self.config['gpuid'][0])
        return self


# class SpecificClassifierPrompt(NormalNN):


class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='coda',
                                                                               prompt_param=self.prompt_param)
        return model


class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='dual',
                                                                               prompt_param=self.prompt_param)
        return model


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
        self._num_anchor_key_prototype_per_class = 20
        self._create_mapping_from_class_to_task()

        self.list_data = list()

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='cpp',
                                                                               prompt_param=self.prompt_param)
        return model

    def _create_mapping_from_class_to_task(self):
        self.mapping_class_to_task = dict()
        for task_id, class_range in enumerate(self.tasks):
            for class_id in class_range:
                self.mapping_class_to_task[class_id] = task_id

    def _create_criterion_fn(self):
        self.criterion_fn = ContrastivePrototypicalLoss(temperature=0.6, reduction="mean")

    def _learnable_params(self):
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.MLP_neck.module.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.MLP_neck.parameters())
        return params_to_opt

    def _update_prototype_set(self, prototype_set, train_loader, use_prompt=False):
        """
        Function to update prototype of previous class.
        """
        with torch.no_grad():

            list_last_feature = list()
            list_output = list()
            for i, (x, y, task) in enumerate(train_loader):
                self.model.eval()
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                if use_prompt:
                    last_feature, _ = self.model(x, get_logit=False, train=False, use_prompt=True,
                                                 task_id=task, prompt_type=self.prompt_type)
                else:
                    last_feature, _ = self.model(x, get_logit=False, train=False, use_prompt=False,
                                                 task_id=task, prompt_type=self.prompt_type)

                list_last_feature.append(last_feature)
                list_output.append(y)

            last_features = torch.cat(list_last_feature, dim=0)
            outputs = torch.cat(list_output, dim=0)
            uni_output = sorted(torch.unique(outputs).tolist())
            for class_id in uni_output:
                feature_set_for_class_id = last_features[outputs == class_id]
                assert feature_set_for_class_id.ndim == 2, "feature_set_for_class_id.ndim != 2."

                clustering_params = {
                    "num_classes": self._num_anchor_key_prototype_per_class,
                    "max_iter": 1000,
                    "init_times": 1
                }
                cluster_algorithm, _ = fit_kmeans_many_times(feature_set_for_class_id, **clustering_params)
                prototype = cluster_algorithm.get_centroids()
                prototype_set[class_id] = prototype  # (_num_anchor_per_class, emb_d)

                chosen_features = feature_set_for_class_id
                mean_feature = torch.mean(chosen_features, dim=1)
                # mean_data = mean_feature.reshape(1, -1)
                print(chosen_features.shape, prototype.shape)
                dict_data = {
                    "data": chosen_features,
                    "centroid": prototype,
                    "output_file": f"/tsne_plot_prompt_{class_id}.png"
                }
                self.list_data.append(dict_data)

            plot_many_tsne(self.list_data, self.model.task_id + 1, f"/tsne_plot_prompt_all.png")
            return prototype_set

    def _update_key_prototype(self, train_loader):
        self.key_prototype = self._update_prototype_set(prototype_set=self.key_prototype, train_loader=train_loader,
                                                        use_prompt=False)

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, need_loss=True, need_acc=False):
        print("##### Attempt to update key prototype set. #####")
        self._update_key_prototype(train_loader)
        print("##### Finish updating key prototype set. #####")

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
            for class_id in range(self.valid_out_dim):
                key = self.key_prototype[class_id].unsqueeze(0)
                U.append(key)
            U = torch.cat(U, dim=0)  # (num_classes, num_anchors, emb_d)
            total_correct = 0
            total_element = 0

            for i, (input, target, task) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                if task_in is None:
                    correct_task, num_element = self._evaluate(model=model, input=input,
                                                                    target=target, task=task,
                                                                    acc=acc, task_in=None, U=U)
                else:
                    mask = target >= task_in[0]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]
                    mask = target < task_in[-1]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]
                    correct_task, num_element = self._evaluate(model=model, input=input,
                                                                    target=target, task=task, acc=acc,
                                                                    task_in=task_in, U=U)
                total_correct += correct_task
                total_element += num_element
        model.train(orig_mode)
        if verbal:
            ground_truth_task = torch.unique(task).cuda()
            self.log(f"In task {ground_truth_task}, "
                     f"number of correct task: {total_correct} in {total_element} elements")

    def _evaluate(self, model, input, target, task, acc, task_in=None, U=None):
        with torch.no_grad():
            top_k = 1
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

            return num_element_correct_task, B


class ProgressivePrompt(Prompt):

    def __init__(self, learner_config):
        super(ProgressivePrompt, self).__init__(learner_config)
        self.prompt_MLP_params = None
        self.classifier_dict = dict()
        self.dict_last_valid_out_dim = dict()

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim,
                                                                               prompt_flag='concat',
                                                                               prompt_param=self.prompt_param)
        return model

    def _set_learnable_parameter(self):

        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + \
                            list(self.classifier_dict[self.model.task_id].parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + \
                            list(self.classifier_dict[self.model.task_id].parameters())
        return params_to_opt

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, normalize_target=True):
        self.dict_last_valid_out_dim[self.model.task_id] = self.last_valid_out_dim
        self.create_classifier(self.model.task_id)
        if not self.first_task:
            self.model.prompt.concatenate_prompt(self.model.task_id)
        super().learn_batch(train_loader=train_loader, train_dataset=train_dataset,
                            model_save_dir=model_save_dir, val_loader=val_loader, normalize_target=normalize_target)

    def create_classifier(self, task_id):
        feature_dim = self.model.prompt.emb_d
        num_classes = len(self.tasks[task_id])
        model = nn.Linear(in_features=feature_dim, out_features=num_classes).cuda()
        self.classifier_dict[task_id] = model

    def update_model(self, inputs, targets):

        feature, _ = self.model(x=inputs, get_logit=False, train=True,
                                use_prompt=True, task_id=None, prompt_type=self.prompt_type)

        logit = self.classifier_dict[self.model.task_id](feature)

        # ce with heuristic
        # logit[:, :self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logit, targets.long())

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.model.prompt.freeze_previous_prompt(self.model.task_id)
        self.optimizer.step()

        return total_loss.detach(), logit

    def _evaluate(self, model, input, target, task, acc, task_in=None):
        with torch.no_grad():
            task = torch.unique(task)[0].item()
            last_valid = self.dict_last_valid_out_dim[task]
            if task_in is None:
                feature, _ = model(input, get_logit=False, train=False, use_prompt=True,
                                   task_id=task, prompt_type=self.prompt_type)
                output = self.classifier_dict[task](feature)
                acc = accumulate_acc(output, target - last_valid, task, acc, topk=(self.top_k,))
            else:
                feature, _ = model(input, get_logit=True, train=False, use_prompt=True,
                                   task_id=task, prompt_type=self.prompt_type)
                output = self.classifier_dict[task](feature)
                acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))
            return acc


class GaussianFeaturePrompt(Prompt):
    def __init__(self, learner_config):
        super(GaussianFeaturePrompt, self).__init__(learner_config)

        self.label_embedding_optim = None
        self.label_embedding = None

        self.classifier_dict = dict()
        self.distribution = dict()

        self.validation_classifier = None

        self.logit_normalize = True
        self.logit_norm = 0.1

        self._num_anchor_key_prototype_per_class = 5

        self.key_prototype = dict()
        self.mapping_class_to_task = dict()

        self._create_mapping_from_class_to_task()

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag='cpp',
                                                                               prompt_param=self.prompt_param)
        return model

    def _set_learnable_parameter(self):

        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + \
                            list(self.classifier_dict[self.model.task_id].parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + \
                            list(self.classifier_dict[self.model.task_id].parameters())
        return params_to_opt

    def _create_mapping_from_class_to_task(self):
        for task_id, class_range in enumerate(self.tasks):
            for class_id in class_range:
                self.mapping_class_to_task[class_id] = task_id

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, normalize_target=True):
        self.create_classifier(self.model.task_id)  # create classifier for each task
        self.create_label_embedding(self.model.task_id)
        print(f"Create classifier for task id {self.model.task_id}")
        self._update_key_prototype(train_loader)
        # learn prompt
        print(f"##### Attempt to learn batch in task id: {self.model.task_id}. #####")
        self._learn_batch(train_loader=train_loader, train_dataset=train_dataset,
                          model_save_dir=model_save_dir, val_loader=val_loader, normalize_target=normalize_target)
        print(f"##### Finish learning batch in task id: {self.model.task_id}. #####")
        print()
        print(f"Start learning Gaussian distribution for each class of task id: {self.model.task_id}")
        self.get_distribution(train_loader=train_loader)
        print(f"Finish learning Gaussian distribution for each class of task id: {self.model.task_id}")
        print(f"##### Attempt to learn validation classifier in task id: {self.model.task_id}. #####")
        self.learn_validation_classifier(val_loader=val_loader)
        print(f"##### Finish learning validation classifier in task id: {self.model.task_id}. #####")

    def get_distribution(self, train_loader):
        """
        Learn distribution for each class in current task
        """
        with torch.no_grad():
            all_x = list()
            all_y = list()
            for i, (x, y, task) in enumerate(train_loader):
                # verify in train mode
                self.model.train()

                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                # model update
                all_x.append(x)
                all_y.append(y)
            all_x = torch.cat(all_x, dim=0)
            all_y = torch.cat(all_y, dim=0)

            unique_Y = torch.unique(all_y)

            for label in unique_Y:
                label = label.item()
                learning_dist_model_params = {
                    "num_clusters": 8,
                    "covariance_type": "diag"
                }
                dist = get_learning_distribution_model(model_type="gaussian", **learning_dist_model_params)
                X_class = all_x[all_y == label]
                feature, _ = self.model(x=X_class, get_logit=False, train=False,
                                        use_prompt=True, task_id=None, prompt_type=self.prompt_type)
                feature = feature.cpu()
                print(f"##### LEARN MIXTURE OF GAUSSIAN FOR LABEL: {label} #####")
                dist.learn_distribution(feature)
                print(f"##### FINISH LEARNING MIXTURE OF GAUSSIAN FOR LABEL: {label} #####")
                self.distribution[label] = dist

    def _learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None, normalize_target=False):

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        if need_train:
            # data weighting
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch = epoch

                if epoch > 0:
                    self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])

                batch_timer.tic()
                total_gaussian_loss = 0
                num_training = 0
                for i, (x, y, task) in enumerate(train_loader):
                    if normalize_target:
                        y = y - self.last_valid_out_dim
                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, gaussian_loss, output = self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                    total_gaussian_loss += gaussian_loss
                    num_training += x.shape[0]
                # eval update
                self.log(
                    'Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch + 1, total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} Gaussian Loss {gauss_loss: .3f} | Train Acc {acc.avg:.3f}'
                         .format(loss=losses, gauss_loss=total_gaussian_loss / num_training, acc=acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

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

    def update_model(self, inputs, targets):

        feature, _ = self.model(x=inputs, get_logit=False, train=True,
                                use_prompt=True, task_id=None, prompt_type=self.prompt_type)

        logit = self.classifier_dict[self.model.task_id](feature)

        # pseudo_mean = self.label_embedding(targets.unsqueeze(-1).to(torch.float32))
        pseudo_mean = self.label_embedding[targets.to(torch.int32), :]

        gaussian_penalty = torch.mean(torch.sum((feature - pseudo_mean) ** 2, dim=1))

        # ce with heuristic
        # if self.model.task_id == 0:
        # total_loss = self.criterion(logit, targets.long()) + 0.001 * gaussian_penalty
        total_loss = gaussian_penalty
        # else:
        #     kl_div =

        # step
        self.optimizer.zero_grad()
        self.label_embedding_optim.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.label_embedding_optim.step()

        return total_loss.detach(), gaussian_penalty.detach(), logit

    def _generate_synthesis_prototype(self, num_sample=256):
        x_synthesis = list()
        y_synthesis = list()
        for label, dist in self.distribution.items():
            x_sample = dist.sample(num_sample)
            x_synthesis.append(x_sample)
            y_sample = torch.ones(x_sample.shape[0]) * label
            y_synthesis.append(y_sample)
        x_synthesis = torch.cat(x_synthesis, dim=0)
        y_synthesis = torch.cat(y_synthesis, dim=0)
        return x_synthesis, y_synthesis

    def data_generator(self, x_train, y_train, batch_size=256, randomize=False):
        num_examples = len(x_train)
        if batch_size is not None:
            if randomize is True:
                ind_arr = np.arange(num_examples)
                np.random.shuffle(ind_arr)
                x_train, y_train = x_train[ind_arr], y_train[ind_arr]
            for idx in range(0, num_examples, batch_size):
                idy = min(idx + batch_size, num_examples)
                yield x_train[idx:idy], y_train[idx:idy]
        else:
            yield x_train, y_train

    def create_label_embedding(self, task):
        task_info = self.tasks[task]
        num_classes = len(task_info)
        self.label_embedding = nn.Parameter(data=torch.randn(num_classes, self.model.feature_dim, device='cuda'),
                                            requires_grad=True)
        # self.label_embedding = nn.Linear(1, self.model.feature_dim).cuda()
        self.label_embedding_optim = torch.optim.Adam(lr=0.0005, params=[self.label_embedding])

    def create_validation_classifier(self, linear_model=True):
        feature_dim = self.model.feature_dim
        if linear_model:
            self.validation_classifier = nn.Linear(feature_dim, self.valid_out_dim).cuda()
        else:
            self.validation_classifier = MLP(in_feature=feature_dim, hidden_features=[1024, 256],
                                             out_feature=self.valid_out_dim).cuda()

    def validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, **kwargs):
        U = list()
        for class_id in range(self.valid_out_dim):
            key = self.key_prototype[class_id].unsqueeze(0)
            U.append(key)
        U = torch.cat(U, dim=0)  # (num_classes, num_anchors, emb_d)
        kwargs["U"] = U
        return self._validation(dataloader, model, task_in, task_metric, verbal, **kwargs)

    def task_id_prediction(self, model, input, U, top_k=1):
        if model is None:
            model = self.model

        x_query = model.retrieve_query_vector(input)
        B, C = x_query.shape
        # cosine similarity to match keys/queries
        n_U = nn.functional.normalize(U, dim=2)  # (num_classes, num_anchors, emb_d)
        q = nn.functional.normalize(x_query, dim=1).detach()  # (B, emb_d)
        cos_sim = torch.einsum('kj,bij->kbi', q, n_U)  # (B, num_classes, num_anchors)
        flatten_cos_sim = cos_sim.reshape(B, -1)  # (B, num_classes * num_anchors)
        prototype_id_ranking = torch.topk(flatten_cos_sim, top_k, dim=1)
        ranking = prototype_id_ranking.indices  # shape == (B, self.top_k)

        possible_task_id = torch.zeros_like(ranking)
        possible_class_id = torch.zeros_like(ranking)
        for class_id in range(self.valid_out_dim):
            # [0, 5]
            class_range = (class_id * self._num_anchor_key_prototype_per_class,
                           (class_id + 1) * self._num_anchor_key_prototype_per_class)
            for c in range(class_range[0], class_range[1]):
                possible_task_id[ranking == c] = self.mapping_class_to_task[class_id]
                possible_class_id[ranking == c] = class_id

        if top_k == 1:
            return possible_task_id.squeeze(-1)

        else:

            flatten_possible_task_id = possible_task_id.reshape(-1, 1)  # flatten, shape == (B * self.top_k, 1)
            flatten_possible_task_id = flatten_possible_task_id.squeeze(-1)

            inp = input.unsqueeze(0) # (1, B, C, H, W)
            input_repeat = inp.repeat(top_k, 1, 1, 1, 1) # (top_k, B, C, H, W)
            input_repeat = input_repeat.permute(1, 0, 2, 3, 4) # (B, top_k, C, H, W)
            input_repeat = input_repeat.reshape(-1, input_repeat.shape[2], input_repeat.shape[3], input_repeat.shape[4])

            last_feature, _ = self.model(input_repeat, get_logit=False, train=False,
                                         use_prompt=True, task_id=flatten_possible_task_id,
                                         prompt_type=self.prompt_type)  # (top_k * B, emb_d)

            # fine_grained_query = last_feature.reshape(B, top_k, self.model.prompt.emb_d)

            score_likelihood = torch.zeros(B * top_k, self.valid_out_dim)

            for class_id, distribution in self.distribution.items():
                score_likelihood[:, class_id] = distribution.log_likelihood(last_feature.cpu()).cpu()

            flatten_possible_class_id = possible_class_id.reshape(-1, 1).squeeze(-1).cpu()
            selected_score = score_likelihood[
                torch.arange(start=0, end=B).reshape(-1, 1).repeat(1, top_k).reshape(-1, 1).squeeze(-1).to(torch.int32),
                flatten_possible_class_id.to(torch.int32)].reshape(B, -1)

            assert selected_score.shape == (B, top_k)

            decision = torch.max(selected_score, dim=1).indices

            res = possible_task_id[range(B), decision]

            return res

    def _validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, **kwargs):
        with torch.no_grad():
            if model is None:
                model = self.model
            # This function doesn't distinguish tasks.
            batch_timer = Timer()
            acc = AverageMeter()
            batch_timer.tic()
            orig_mode = model.training
            model.eval()

            correct_task = 0
            total_instance = 0
            for i, (input, target, task) in enumerate(dataloader):

                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                if task_in is None:
                    acc, num_correct_task, unique_task \
                        = self._evaluate(model=model, input=input, target=target, task=task, acc=acc, task_in=None,
                                         **kwargs)

                else:
                    mask = target >= task_in[0]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]
                    mask = target < task_in[-1]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]
                    acc, num_correct_task, unique_task \
                        = self._evaluate(model=model, input=input, target=target, task=task, acc=acc, task_in=task_in,
                                         **kwargs)

                correct_task += num_correct_task
                total_instance += task.cpu().numel()

        model.train(orig_mode)
        if verbal:
            self.log(f'In task {unique_task}:')
            self.log('  * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                     .format(acc=acc, time=batch_timer.toc()))
            self.log(f' * Percentage of correct task: {correct_task / total_instance}')
        return acc.avg

    def _evaluate(self, model, input, target, task, acc, task_in=None, **kwargs):
        with torch.no_grad():
            predicted_task = self.task_id_prediction(model, input, kwargs["U"], top_k=3)
            unique_task = torch.unique(task)[0].item()
            num_correct_task = torch.sum(predicted_task.cpu() == task.cpu())
            # print(f"In task: {unique_task}, percentage of correct task: {torch.sum(predicted_task.cpu() == task.cpu()) / task.cpu().numel()}")
            if task_in is None:
                feature, _ = model(input, get_logit=False, train=False, use_prompt=True,
                                   task_id=predicted_task, prompt_type=self.prompt_type)
                output = self.validation_classifier(feature)
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                feature, _ = model(input, get_logit=True, train=False, use_prompt=True,
                                   task_id=predicted_task, prompt_type=self.prompt_type)
                output = self.validation_classifier(feature)
                acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))
            return acc, num_correct_task, unique_task

    def _retrieve_validation_set_for_validation_classifier(self, dataloader, model=None):
        U = list()
        for class_id in range(self.valid_out_dim):
            key = self.key_prototype[class_id].unsqueeze(0)
            U.append(key)
        U = torch.cat(U, dim=0)  # (num_classes, num_anchors, emb_d)

        list_feature = list()
        list_label = list()
        with torch.no_grad():
            if model is None:
                model = self.model
            # This function doesn't distinguish tasks.
            orig_mode = model.training
            model.eval()
            acc = AverageMeter()
            for i, (input, target, task) in enumerate(dataloader):

                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()

                predicted_task = self.task_id_prediction(model, input, U, top_k=3)
                feature, _ = model(input, get_logit=False, train=False, use_prompt=True,
                                   task_id=predicted_task, prompt_type=self.prompt_type)

                list_feature.append(feature)
                list_label.append(target)

            list_feature = torch.cat(list_feature, dim=0)
            list_label = torch.cat(list_label, dim=0)

        return list_feature, list_label

    def _evaluate_validation_classifier(self, feature, target, classifier=None):
        with torch.no_grad():
            if classifier is None:
                classifier = self.validation_classifier
            output = classifier(feature)
            predicted_class = torch.max(output, dim=1).indices
            acc = torch.sum(predicted_class == target) / target.shape[0]
        return acc

    def learn_validation_classifier(self, max_iter=40, lr=0.01, val_loader=None):
        self.create_validation_classifier(linear_model=True)
        MAX_ITER = 10 if max_iter is None else max_iter
        LR = 0.001 if lr is None else lr
        classifier_optimizer = torch.optim.Adam(params=self.validation_classifier.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=classifier_optimizer, T_max=MAX_ITER)
        print("Start synthesize prototype")
        x_syn, y_syn = self._generate_synthesis_prototype()
        print(f"Finish synthesizing prototype, which prototype shape: {x_syn.shape, y_syn.shape}")
        print("Attempt to learn validation classifier...")
        UPPER_THRESHOLD = 0.25

        eval_feature, eval_target = self._retrieve_validation_set_for_validation_classifier(val_loader)

        for iter in range(max_iter):
            loss = 0
            old_loss = 1e9
            if iter > 0:
                scheduler.step()
            data_loader = self.data_generator(x_syn, y_syn, randomize=True)
            for (x, y) in data_loader:
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()
                out = self.validation_classifier(x)

                #############################################################
                if self.logit_normalize:
                    copied_out = out.detach().clone()
                    num_task_so_far = self.task_count
                    num_class_per_task = self.valid_out_dim // num_task_so_far
                    copied_out = copied_out.reshape(out.shape[0], num_task_so_far, num_class_per_task)
                    per_task_norm = torch.norm(copied_out, p=2, dim=-1) + 1e-7
                    assert per_task_norm.shape == (out.shape[0], num_task_so_far), \
                        "per_task_norm.shape != (out.shape[0], num_task_so_far)."
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    assert norms.shape == (out.shape[0], 1), "norms.shape != (out.shape[0], 1)."
                    out = torch.div(out, norms) / self.logit_norm
                #############################################################

                total_loss = self.criterion(out, y.long())
                classifier_optimizer.zero_grad()
                total_loss.backward()
                classifier_optimizer.step()
                loss += total_loss.detach()
            if iter % 1 == 0:
                acc = self._evaluate_validation_classifier(eval_feature, eval_target)
                print(f"Learning validation classifier... iteration {iter}, loss function: {loss}, "
                      f"accuracy on validation set: {acc}")
            if loss - old_loss > UPPER_THRESHOLD:
                break
            old_loss = loss

    def _update_prototype_set(self, prototype_set, train_loader):
        with torch.no_grad():
            list_last_feature = list()
            list_last_feature_with_prompt = list()
            list_output = list()
            for i, (x, y, task) in enumerate(train_loader):
                self.model.eval()
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                last_feature, _ = self.model(x, get_logit=False, train=False, use_prompt=False)

                last_feature_with_prompt, _ = self.model(x, get_logit=False, train=False, use_prompt=True,
                                                         task_id=task, prompt_type=self.prompt_type)

                list_last_feature_with_prompt.append(last_feature_with_prompt)
                list_last_feature.append(last_feature)
                list_output.append(y)

            last_features = torch.cat(list_last_feature, dim=0)
            last_feature_with_prompt = torch.cat(list_last_feature_with_prompt, dim=0)
            outputs = torch.cat(list_output, dim=0)

            uni_output = sorted(torch.unique(outputs).tolist())
            for class_id in uni_output:
                feature_set_for_class_id = last_features[outputs == class_id]
                assert feature_set_for_class_id.ndim == 2, "feature_set_for_class_id.ndim != 2."
                check_tensor_nan(feature_set_for_class_id, "feature_set_for_class_id")

                clustering_params = {
                    "num_classes": self._num_anchor_key_prototype_per_class,
                    "max_iter": 1000,
                    "init_times": 1
                }
                cluster_model, _ = fit_kmeans_many_times(feature_set_for_class_id, **clustering_params)
                prototype = cluster_model.get_centroids()
                prototype_set[class_id] = prototype  # (_num_anchor_per_class, emb_d)
                check_tensor_nan(prototype, "prototype")

                # initialize label_embedding data
                feature_with_prompt_for_class = last_feature_with_prompt[outputs == class_id]
                self.label_embedding.data[class_id - self.last_valid_out_dim, :] = \
                    torch.mean(feature_with_prompt_for_class, dim=0)

            return prototype_set

    def _update_key_prototype(self, train_loader):
        self.key_prototype = self._update_prototype_set(prototype_set=self.key_prototype, train_loader=train_loader)

    def create_classifier(self, task_id):
        feature_dim = self.model.prompt.emb_d
        num_classes = len(self.tasks[task_id])
        model = nn.Linear(in_features=feature_dim, out_features=num_classes).cuda()
        self.classifier_dict[task_id] = model


def plot_many_tsne(list_data, task_id, plotted_file):
    color_list = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#e6ab02", "#01a2d9", "#a6761d", "#ff33a1",
        "#ff009b", "#a6a6a6", "#636363", "#d9d9d9", "#737373",
        "#252525", "#525252", "#969696", "#cccccc", "#696969",
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
        "#e5c494", "#b3b3cc", "#cab2d6", "#ff6666", "#c2c2f0",
        "#ffb3e6", "#c2f0c2", "#ffcc99", "#c2c2a3", "#ff9999",
        "#ffcc66", "#ffff99", "#ccffcc", "#cce6ff", "#99ccff",
        "#cc99ff", "#ff99cc", "#ffccff", "#b3e0ff", "#ffcc00",
        "#ccff00", "#99ff00", "#ffcc33", "#ccff33", "#99ff33",
        "#ffcc66", "#ccff66", "#99ff66", "#ffcc99", "#ccff99",
        "#99ff99", "#ffcccc", "#ccffcc", "#99ffcc", "#ffffff",
        "#cccccc", "#333333", "#993300", "#009900", "#000099",
        "#990099", "#009999", "#999900", "#990000", "#009900",
        "#990099", "#009999", "#999900", "#009933", "#990033",
        "#003399", "#993399", "#339900", "#339933", "#339966",
        "#660033", "#660099", "#996600", "#ff0033", "#ff0099"
    ]

    plot_save_dir = f"plot/task_{task_id}"
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    with torch.no_grad():
        tsne = TSNE(n_components=2, perplexity=500, random_state=0)
        all_data = list()
        bookmark = list()
        for data_dict in list_data:
            centroid = data_dict["centroid"]
            data = data_dict["data"]
            num_data = data.shape[0]
            num_centroid = centroid.shape[0]
            print(num_data, num_centroid)
            if len(bookmark) == 0:
                bookmark.append([[0, num_data], [num_data, num_data + num_centroid]])
            else:
                prev_len = bookmark[-1][1][1]
                bookmark.append(
                    [[prev_len, num_data + prev_len], [num_data + prev_len, num_data + prev_len + num_centroid]])

            all_data.append(data)
            all_data.append(centroid)

        data = torch.cat(all_data, dim=0)
        X_tsne = tsne.fit_transform(data)

        for i in range(len(bookmark)):
            b = bookmark[i]
            b_data = b[0]
            b_centroid = b[1]

            data_class = X_tsne[b_data[0]:b_data[1], :]
            centroid_class = X_tsne[b_centroid[0]:b_centroid[1], :]
            color = color_list[i]

            print(f"Class: {i}, data ranging: {b_data}, centroid ranging: {b_centroid}")

            plt.figure(figsize=(8, 6))
            plt.scatter(data_class[:, 0], data_class[:, 1], marker='o', s=20, c=color, alpha=0.2)
            plt.scatter(centroid_class[:, 0], centroid_class[:, 1], marker='*', s=100, c=color, alpha=0.8)

            plt.title(f'Class {i+1}')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True)
            plt.legend()
            class_output_file = plot_save_dir + list_data[i]["output_file"]
            plt.savefig(class_output_file)
            plt.show()

        plt.figure(figsize=(8, 6))
        for i in range(len(bookmark)):
            b = bookmark[i]
            b_data = b[0]
            b_centroid = b[1]
            data_class = X_tsne[b_data[0]:b_data[1], :]
            centroid_class = X_tsne[b_centroid[0]:b_centroid[1], :]
            color = color_list[i]
            plt.scatter(data_class[:, 0], data_class[:, 1], marker='o', s=20, c=color, alpha=0.3)
            plt.scatter(centroid_class[:, 0], centroid_class[:, 1], marker='*', s=100, c=color, alpha=1)

        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.legend()
        all_class_save_file = plot_save_dir + plotted_file
        plt.savefig(all_class_save_file)
        plt.show()

def check_tensor_nan(tensor, tensor_name="a"):
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        raise f"Tensor {tensor_name} is nan."
