import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.models.modeling import *
from src.models.vit_seg_modeling import *
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from option import parse
from torch.nn.modules.utils import _pair
import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])

from src.jf_utils.model.classifier import Classifier
from easydict import EasyDict as edict
import json
from glob import glob

class CNN(nn.Module):
    """Construct the embeddings from patch, position embeddings.
        """

    def __init__(self, config, img_size):
        super(CNN, self).__init__()

        self.hybrid_model = None
        self.config = config
        if config.transformer is not None:
            self.is_transformer = True
            if config.patches.get("grid") is not None:
                if config.which_backbone == 'D121':
                    jf_cfg_path = "../../src/jf_utils/base_11pathol.json"
                    with open(jf_cfg_path) as f:
                        cfg_jf = edict(json.load(f))

                    self.hybrid_model = Classifier(cfg_jf)
                    if args.jf_pretrain:
                        ckpt = torch.load("../../src/jf_utils/pretrained/jf_best.ckpt")
                        self.hybrid_model.load_state_dict(ckpt['state_dict'])
                        print('JF Model pre-trained weights have loaded.')
                    # in_channels = 9  # 10 of 11 except NoFinding
                    num_features = 1024 * 1
                    self.patch_embeddings = nn.Conv2d(in_channels=num_features, out_channels=config.hidden_size,
                                                      kernel_size=(1, 1))
                else:
                    self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                                 width_factor=config.resnet.width_factor)
                    num_features = self.hybrid_model.width * 16
                    self.patch_embeddings = nn.Conv2d(in_channels=num_features,
                                                      out_channels=config.hidden_size,
                                                      kernel_size=2,
                                                      stride=2)
            else:
                patch_size = _pair(config.patches["size"])
                self.patch_embeddings = Conv2d(in_channels=3,
                                               out_channels=config.hidden_size,
                                               kernel_size=patch_size,
                                               stride=patch_size)


        else:
            self.is_transformer = False
            jf_cfg_path = f"../../src/jf_utils/config/kgh_10pathol_{config.which_backbone}_noPCAM.json"
            with open(jf_cfg_path) as f:
                cfg_jf = edict(json.load(f))

            self.hybrid_model = Classifier(cfg_jf)

            if args.jf_pretrain:
                ckpt_path = glob(f"../../src/jf_utils/pretrained/{config.which_backbone}_noPCAM_best_*.ckpt")
                ckpt = torch.load(ckpt_path[0])
                self.hybrid_model.load_state_dict(ckpt['state_dict'])
                print('JF Model pre-trained weights have loaded.')

    def forward(self, x):
        if self.hybrid_model is not None:
            if self.is_transformer and self.config.which_backbone == 'R50':
                x = self.hybrid_model(x)
            else:
                _, x = self.hybrid_model(x)

            if self.is_transformer:
                x = self.patch_embeddings(x)
            else:
                x_gap = F.adaptive_avg_pool2d(x, 1).flatten(2).transpose(-1, -2)
                x = x.flatten(2).transpose(-1, -2)
                x = torch.cat((x_gap, x), dim=1)

        else:
            x = self.patch_embeddings(x)
        return x