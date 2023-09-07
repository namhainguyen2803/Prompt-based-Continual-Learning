import torch
import torch.nn as nn
from .vit import VisionTransformer
from timm.models import vit_base_patch16_224


class L2P(nn.Module):
    """class used to manage the prompt thing, such as updating, forward, ..."""
    def __init__(self, embedding_dimension, prompt_param, key_dimension=768):
        super(L2P, self).__init__()

        self.task_count = 0
        self.embedding_dimension = embedding_dimension
        self.key_dimension = key_dimension
        self._init_smart(prompt_param)

        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, self.embedding_dimension) # prompt parameter
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
            Ek = selected_prompt[:,:,:i,:].reshape((B,-1,self.embedding_dimension))
            Ev = selected_prompt[:,:,i:,:].reshape((B,-1,self.embedding_dimension))
            p_return = [Ek, Ev]

        return p_return, loss, x



# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
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
        zoo_model = VisionTransformer(img_size=224, patch_size=16, embedding_dim=768, depth=12,
                                      num_heads=12, drop_path_rate=0)

        if pt:
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feature_encoder = zoo_model

    # pen: get penultimate features
    def forward(self, x, pen=False, train=False):
        prompt_loss = 0
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feature_encoder(x)
                q = q[:,0,:].detach().clone() # [class] token!!!, having shape == (B, 1, self.embedding_dimension)
            out, prompt_loss = self.feature_encoder(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feature_encoder(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if pen == False:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

def vit_pt_imnet(out_dim, prompt_flag = 'l2p', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)