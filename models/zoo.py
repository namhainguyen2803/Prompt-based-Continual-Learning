import torch
import torch.nn as nn
from .vit import VisionTransformer
from timm.models import vit_base_patch16_224
from l2p_prompt import L2P

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
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[1])
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