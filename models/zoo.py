from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch import autograd
from torch.autograd import Variable
from .vit import VisionTransformer
from timm.models import vit_base_patch16_224
import numpy as np
import copy


# Our method!
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            # note that emb_d and self.key_d must be MATCHED
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_3d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_3d)

        return torch.nn.Parameter(uu)

    def forward(self, x_query, l, x_block, train=False, task_id=None, possible_task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_query.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))  # number of prompts for current task
            s = int(self.task_count * pt)  # index of starting prompt for current task
            f = int((self.task_count + 1) * pt)  # index of ending prompt for current task

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    # freeze K[:s], only train K[s:f]
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)  # shape == (f, self.key_d)
                    # freeze A[:s], only train A[s:f]
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)  # shape == (f, self.key_d)
                    # freeze p[:s], only train p[s:f]
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)  # shape == (f, e_l, self.emb_d)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # x_query.shape == (B, self.emb_d)
            # A.shape == (f, self.emb_d)
            a_query = torch.einsum('bd,kd->bkd', x_query, A)  # shape == (B, f, self.emb_d)
            # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)  # shape == (f, self.key_d)
            q = nn.functional.normalize(a_query, dim=2)  # shape == (B, f, self.emb_d)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)  # shape == (B, f) # attention of set of prompt
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}', p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, emb_d, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_query, l, x_block, train=False, task_id=None, possible_task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_query.shape
            K = getattr(self, f'e_k_{l}')
            p = getattr(self, f'e_p_{l}')

            # cosine similarity to match keys/queries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_query, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)

            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_ = p[task_id].expand(B, -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]

            # select prompts
            # Prefix prompt
            # break all the prompt in the selected set into 2 evenly part alongside with self.e_p_length dimension
            # then concatenate those
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape(
                    (B, -1, self.emb_d))  # shape == (B, self.top_k * i, self.embedding_dimension)
                Ev = P_[:, :, i:, :].reshape((B, -1, self.emb_d))

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f'g_p_{l}')  # 0 based indexing here
            P_ = p.expand(len(x_query), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0, 1, 2, 3, 4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])


class ContrastivePrototypicalPrompt(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
        self._delete_garbage_parameter()

    def _delete_garbage_parameter(self):
        for l in self.e_layers:
            # delete key parameter of prompt
            k = getattr(self, f'e_k_{l}')
            del k

    def _init_smart(self, emb_d, prompt_param):
        # prompt locations
        self.e_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.g_layers = []

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

        self.task_id_bootstrap = True
        self.top_k = 3

    def forward(self, x_query, l, x_block, train=False, task_id=None, possible_task_id=None, prompt_type="tuning"):
        # e prompts
        B, C = x_query.shape
        if not train:
            if possible_task_id is None:
                possible_task_id = torch.full((B, 1), task_id, dtype=torch.int64)
        else:
            assert task_id is not None, "In train mode, task_id cannot be None."
        p_return = None
        if l in self.e_layers:
            B, C = x_query.shape
            p = getattr(self, f'e_p_{l}')
            if train:  # CPP in training time, need to access to task-specific prompt
                # no need cos-sim loss
                P_ = p[task_id].expand(B, -1, -1)
            else:  # CPP in testing time, but differs than that of DualPrompt!
                assert possible_task_id.shape == (B, 1), "Wrong in class ContrastivePrototypicalPrompt(DualPrompt)."
                P_ = p[possible_task_id]
                P_ = P_.squeeze(1)

            # select prompts
            # Prefix prompt
            if prompt_type == "prefix":
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
                p_return = [Ek, Ev]
            elif prompt_type == "tuning":
                p_return = P_

        return p_return, 0, x_block

class MaskedPrompt(ContrastivePrototypicalPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        # prompt locations
        self.e_layers = [0, 1, 2, 3]
        self.g_layers = []

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])
        self.list_prompt = None
        self.task_id_bootstrap = True
        self.top_k = 3

    def _retrieve_previous_prompt(self, task_id):
      list_prev_prompt = list()
      for l in self.e_layers:
        prev_prompt = getattr(self, f'e_p_{l}')[:task_id].detach().clone()  # (num_prev_task_id, L_p, emb_d)
        prev_prompt = prev_prompt.reshape(-1, prev_prompt.shape[2])  # (num_prev_task_id * L_p, emb_d)
        list_prev_prompt.append(prev_prompt)
      return list_prev_prompt

    def initialize_prompt(self, task_id):
        for l in self.e_layers:
            masked_prev_prompt = self.masked_prompt[l]
            non_zero_prev_prompt_indices = torch.any(masked_prev_prompt != 0, dim=1)
            non_zero_prev_prompt = masked_prev_prompt[non_zero_prev_prompt_indices]
            non_zero_prev_prompt.requires_grad = False
            p = getattr(self, f'e_p_{l}')[task_id] # initialize current prompt
            p = torch.cat((non_zero_prev_prompt, p[non_zero_prev_prompt.shape[0]:]), dim=0)
            

    def set_masked_prompt(self, masked_prev_prompt):
        self.masked_prompt = masked_prev_prompt

    def forward(self, x_query, l, x_block, train=False, task_id=None, possible_task_id=None,
                prompt_type="tuning", learn_mask=False):

        if learn_mask:
            assert self.masked_prompt is not None
            P_ = None
            if l in self.e_layers:
                B, C = x_query.shape
                P_ = self.masked_prompt[l].expand(B, -1, -1)
            return P_, 0, x_block

        else:
            B, C = x_query.shape
            if not train:
                if possible_task_id is None:
                    possible_task_id = torch.full((B, 1), task_id, dtype=torch.int64)
            else:
                assert task_id is not None, "In train mode, task_id cannot be None."
            p_return = None
            if l in self.e_layers:
                B, C = x_query.shape
                p = getattr(self, f'e_p_{l}')
                if train:
                    # no need cos-sim loss
                    P_ = p[task_id].expand(B, -1, -1)
                else:  # CPP in testing time, but differs than that of DualPrompt!
                    assert possible_task_id.shape == (B, 1), "Wrong in class ContrastivePrototypicalPrompt(DualPrompt)."
                    P_ = p[possible_task_id]
                    P_ = P_.squeeze(1)

                # select prompts
                # Prefix prompt
                if prompt_type == "prefix":
                    i = int(self.e_p_length / 2)
                    Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                    Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
                    p_return = [Ek, Ev]
                elif prompt_type == "tuning":
                    p_return = P_

            return p_return, 0, x_block


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=True, prompt_flag="l2p", prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                      num_heads=12, drop_path_rate=0)

        if pt:
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']
            del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        # if self.prompt_flag
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'cpp':
            self.prompt = ContrastivePrototypicalPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'masked':
            self.prompt = MaskedPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        self.feat = zoo_model

    def retrieve_query_vector(self, x):
        with torch.no_grad():
            q, _ = self.feat(x)
            q = q[:, 0, :]
            return q



    # pen: get penultimate(final) features
    def forward(self, x, pen=False, train=False, use_prompt=True, possible_task_id=None, learn_mask=False):
        prompt_loss = 0
        if self.prompt is not None:
            q = self.retrieve_query_vector(x)
            if use_prompt:
                out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id,
                                             possible_task_id=possible_task_id, learn_mask=learn_mask)
            else:
                out, prompt_loss = self.feat(x, prompt=None, q=q, train=train, task_id=self.task_id,
                                             possible_task_id=possible_task_id, learn_mask=learn_mask)
            last_feature = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            last_feature = out[:, 0, :]

        last_feature = last_feature.view(last_feature.size(0), -1)
        logits = self.last(last_feature)

        if not pen:
            if self.prompt is not None and train:
                return logits, prompt_loss
            else:
                return logits
        else:
            if self.prompt is not None and train:
                return last_feature, logits, prompt_loss
            else:
                return last_feature, logits


def vit_pt_imnet(out_dim, prompt_flag='l2p', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)