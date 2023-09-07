import torch
import torch.nn as nn
from utils_prompt import tensor_prompt

class L2P(nn.Module):
    """class used to manage the prompt thing, such as updating, forward, ..."""
    def __init__(self, embed_dim, prompt_param, key_dimension=768):
        super(L2P, self).__init__()

        self.task_count = 0
        self.embed_dim = embed_dim
        self.key_dimension = key_dimension
        self._init_smart(prompt_param)

        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, self.embed_dim) # prompt parameter
            k = tensor_prompt(self.e_pool_size, self.key_dimension) # key parameter
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, prompt_param):
        self.top_k = 5

        # define e_prompt's layer_id
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_query, layer_id, x, train=False):

        p_return = None
        loss = 0
        if layer_id in self.e_layers:
            B, C = x_query.shape # C is the same as self.key_dimension, B is number of instances
            K = getattr(self,f'e_k_{layer_id}') # get key of prompt in layer l
            p = getattr(self,f'e_p_{layer_id}') # get value of prompt in layer l

            # cosine similarity to match query/key
            normalized_prompt_key = nn.functional.normalize(K, dim=1)
            normalized_x_query = nn.functional.normalize(x_query, dim=1).detach()
            cosine_similarity = torch.einsum('bj,kj->bk', normalized_x_query, normalized_prompt_key)

            top_k = torch.topk(cosine_similarity, self.top_k, dim=1)
            selected_prompt_indices = top_k.indices # shape of selected_prompt_indices == (B, self.top_k)
            if train == True:
                loss = (1.0 - cosine_similarity[:,selected_prompt_indices]).sum()
            selected_prompt = p[selected_prompt_indices] # selected_prompt has shape == (B, self.top_k, self.e_p_length, self.embedding_dimension)

            i = int(self.e_p_length/2) # select the middle index of prompt
            # selected_prompt[:,:,:i,:].shape = (B, self.top_k, i, self.embedding_dimension)
            Ek = selected_prompt[:,:,:i,:].reshape((B,-1,self.embed_dim))
            Ev = selected_prompt[:,:,i:,:].reshape((B,-1,self.embed_dim))
            p_return = [Ek, Ev]

        return p_return, loss, x