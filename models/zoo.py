from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
from timm.models import vit_base_patch16_224
import numpy as np
import copy


class AbstractPrompt(nn.Module, ABC):

    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super(AbstractPrompt, self).__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.prompt_param = prompt_param

        self._init_smart(self.emb_d, self.prompt_param)

    @abstractmethod
    def _init_smart(self, emb_d, prompt_param):
        pass

    @abstractmethod
    def forward(self, x_query, l, x_block, train=False, task_id=None, prompt_type="tuning"):
        pass

    def process_task_count(self):
        self.task_count += 1


class CodaPrompt(AbstractPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super(CodaPrompt, self).__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strength of ortho penalty
        self.ortho_mu = prompt_param[2]
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
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

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
            shape_2d = copy.deepcopy(vv.shape)
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
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_query, l, x_block, train=False, task_id=None, prompt_type="tuning"):
        p_return = None
        loss = 0
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_query.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
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
            a_querry = torch.einsum('bd,kd->bkd', x_query, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            selected_prompt = torch.einsum('bk,kld->bld', aq_k, p)

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0

        if e_valid:
            if prompt_type == "prefix":
                # combine prompts for prefix tuning
                i = int(self.e_p_length / 2)
                Ek = selected_prompt[:, :i, :]
                Ev = selected_prompt[:, i:, :]
                p_return = [Ek, Ev]
            elif prompt_type == "tuning":
                p_return = selected_prompt

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


class DualPrompt(AbstractPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super(DualPrompt, self).__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])  # number of g_prompt per layer
        self.e_p_length = int(prompt_param[1])  # number of e_prompt per layer
        self.e_pool_size = int(prompt_param[0])

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

    def forward(self, x_query, l, x_block, train=False, task_id=None, prompt_type="tuning"):
        p_return = None
        loss = 0
        # e prompts
        e_valid = False
        B, C = x_query.shape
        if l in self.e_layers:
            e_valid = True
            K = getattr(self, f'e_k_{l}')
            p = getattr(self, f'e_p_{l}')
            # cosine similarity to match keys/queries
            n_K = nn.functional.normalize(K, dim=1)
            # shape == (self.e_pool_size, self.key_d)
            q = nn.functional.normalize(x_query, dim=1).detach()
            # shape == (self.e_pool_size, self.e_p_length, self.embedding_dimension)
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            # 001303019002
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    selected_e_prompt = p[task_id].expand(len(x_query), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices  # shape of k_idx == (B, self.top_k)
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    selected_e_prompt = p[k_idx]  # shape == (B, self.top_k, self.e_p_length, self.embedding_dimension)
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                selected_e_prompt = p[k_idx]

            # select prompts
            if prompt_type == "prefix":
                if train and self.task_id_bootstrap:
                    i = int(self.e_p_length / 2)
                    Ek = selected_e_prompt[:, :i, :].reshape(B, -1, self.emb_d)
                    Ev = selected_e_prompt[:, i:, :].reshape(B, -1, self.emb_d)
                else:
                    i = int(self.e_p_length / 2)
                    Ek = selected_e_prompt[:, :, :i, :].reshape(B, -1, self.emb_d)
                    # shape == (B, self.top_k * i, self.embedding_dimension)
                    Ev = selected_e_prompt[:, :, i:, :].reshape(B, -1, self.emb_d)

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f'g_p_{l}')
            selected_g_prompt = p.expand(B, -1, -1)

            if prompt_type == "prefix":
                Gk = selected_g_prompt[:, :j, :]
                Gv = selected_g_prompt[:, j:, :]

        if prompt_type == "prefix":
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

        elif prompt_type == "tuning":
            if e_valid and g_valid:
                p_return = torch.cat((selected_e_prompt, selected_g_prompt), dim=1)
            elif e_valid:
                p_return = selected_e_prompt
            elif g_valid:
                p_return = selected_g_prompt
                loss = 0
            else:
                p_return = None
                loss = 0

        else:
            raise "Have not built prompt type other than tuning and prefix yet."

        return p_return, loss, x_block


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

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)


class SpecificPrompt(AbstractPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 3

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0: # deep prompt
            self.e_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, self.emb_d)
            setattr(self, f'e_p_{e}', p)

    def forward(self, x_query, l, x_block, train=False, task_id=None, prompt_type="tuning"):

        p_return = None

        if l in self.e_layers:
            B, C = x_query.shape
            p = getattr(self, f'e_p_{l}')  # shape == (num_task, e_p, emb_d)

            if not isinstance(task_id, torch.Tensor):  # convert to tensor
                task_id = torch.tensor(task_id)
            if task_id.ndim == 0:  # if number then convert to array
                task_id = task_id.unsqueeze(-1)

            if task_id.shape[0] == 1:
                selected_prompt = p[task_id].expand(B, -1, -1)  # shape == (B, e_p, emb_d)
            else:
                assert task_id.shape[0] == B, "task_id.shape[0] != B."
                selected_prompt = p[task_id]  # shape == (B, e_p, emb_d)

            assert selected_prompt.shape == (B, self.e_p_length, self.emb_d), \
                "selected_prompt.shape != (B, self.e_p_length, self.emb_d)."

            # select prompts
            if prompt_type == "prefix":
                i = int(self.e_p_length / 2)
                Ek = selected_prompt[:, :i, :].reshape(B, -1, self.emb_d)
                Ev = selected_prompt[:, i:, :].reshape(B, -1, self.emb_d)
                p_return = [Ek, Ev]
            elif prompt_type == "tuning":
                p_return = selected_prompt
            else:
                raise "Have not built prompt type other than tuning and prefix yet."

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
    def __init__(self, num_classes=10, pt=True, prompt_flag='l2p', prompt_param=None):
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
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'cpp':
            self.prompt = SpecificPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

    def get_feature_vector(self, x):
        with torch.no_grad():
            q, _ = self.feat(x)
            q = q[:, 0, :]
            return q

    def forward(self, x, get_logit=True, train=False, use_prompt=True, task_id=None, prompt_type="prefix"):
        prompt_loss = 0
        if self.prompt is not None and use_prompt is True:
            q = self.get_feature_vector(x)  # query

            if task_id is None:
                tid = self.task_id
            else:
                tid = task_id

            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=tid, prompt_type=prompt_type)

            out = out[:, 0, :]
        else:
            out = self.get_feature_vector(x)

        out = out.view(out.size(0), -1)  # last feature vector

        if get_logit:
            logit = self.last(out)
            return logit, out, prompt_loss
        else:
            return out, prompt_loss


def vit_pt_imnet(out_dim, prompt_flag='l2p', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)
