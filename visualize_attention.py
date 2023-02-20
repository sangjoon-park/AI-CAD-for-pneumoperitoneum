# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from easydict import EasyDict as edict
import json
from jf_utils.utils import transform

import utils
import vision_transformer as vit_o
import glob
from tqdm import tqdm
from utils import getAttMap
import pandas as pd
from vision_transformer import DINOHead, CLS_head


jf_cfg_path = "jf_utils/base_11pathol.json"
with open(jf_cfg_path) as f:
    cfg_jf = edict(json.load(f))

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside save boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch save edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/COVID_8TB/sangjoon/Abdomen/AI_Pneumo_peritoneum/checkpoints/20220511_experiments/20220511_pretrain_DISTL_final/checkpoint.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="student", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_dir", default='/COVID_8TB/sangjoon/Abdomen/AI_Pneumo_peritoneum/data/test_SNUBH/', type=str, help="Path of the save to load.")
    parser.add_argument("--image_size", default=(256, 256), type=int, nargs="+", help="Resize save.")
    parser.add_argument('--output_dir', default='./probability_attention/pretrain_DISTL_SNUBH/erect/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--attn_threshold", type=float, default=0.2, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--view", default='erect', type=str, help="Path of the save to load.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vit_o.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    embed_dim = model.embed_dim
    inter_dim = 384

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # build eval model
    eval_model = utils.MultiCropWrapper(
        model,
        DINOHead(embed_dim, 65536, False), CLS_head(inter_dim, 256, 1), args)

    for p in eval_model.parameters():
        p.requires_grad = False
    eval_model.eval()
    eval_model.to(device)

    csv_ = pd.read_csv('./int_val_with_time.csv')
    csv_list = list(csv_.Image)

    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace(".backbone.", "___"): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("___", ".backbone."): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg2 = eval_model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg2))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open save
    # if args.image_path is None:
    #     # user has not specified any save - we use our own save
    #     print("Please use the `--image_path` argument to indicate the path of the save you wish to visualize.")
    #     print("Since no save path have been provided, we take the first save in our paper.")
    #     response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
    #     img = Image.open(BytesIO(response.content))
    #     img = img.convert('RGB')
    # elif os.path.isfile(args.image_path):
    #     with open(args.image_path, 'rb') as f:
    #         img = Image.open(f)
    #         img = img.convert('RGB')
    # else:
    #     print(f"Provided save path {args.image_path} is non valid.")
    #     sys.exit(1)
    # transform = pth_transforms.Compose([
    #     pth_transforms.Resize(args.image_size),
    #     pth_transforms.ToTensor(),
    # ])
    # img = transform(img)

    img_dir = args.image_dir
    all_images = glob.glob(img_dir + '**/*.png', recursive=True)

    images = []
    for image in all_images:
        if args.view in image:
            images.append(image)
        else:
            pass

    for img_path in tqdm(images):
        image = cv2.imread(img_path, 1)
        orig_img = cv2.imread(img_path, 0)

        # resize orig image
        for csv in csv_list:
            if img_path.split('/')[-1].split('.png')[0] in csv:
                orig_img = cv2.resize(orig_img, dsize=(int(csv_.loc[csv_.Image == csv].Columns), int(csv_.loc[csv_.Image == csv].Rows)))
                break

        orig_img = torch.from_numpy(orig_img / 255.)
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(image)

        img = pth_transforms.Compose(
            [utils.GaussianBlurInference(),
             pth_transforms.ToTensor()])(image)
        # image = np.array(image)
        # img = transform(image, cfg_jf)
        #
        # img = torch.from_numpy(img)

        output = eval_model(img.unsqueeze(0).to(device))
        output = torch.sigmoid(output)
        prob_np = output.detach().cpu().numpy()
        # preds = np.round(prob_np)
        weights = 1.0
        probability = prob_np * weights
        preds = np.round(np.minimum((probability), 1.))

        # make the save divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.get_last_selfattention(img.to(device))

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="bilinear")[0].cpu().numpy()

        # save attentions heatmaps
        # if j == 5:
        if 'Non-pneumoperitoneum' in img_path:
            label = 'normal'
        else:
            label = 'pneumoperitoneum'

        output_dir = args.output_dir + label + '/' + img_path.split('/')[-1] + '/'
        os.makedirs(output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(orig_img, normalize=True, scale_each=True), os.path.join(output_dir, "img.png"))
        for j in range(nh):
            fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
            attentions[j] = (attentions[j] - attentions[j].min()) / (attentions[j].max() - attentions[j].min())
            # plt.imsave(fname=fname, arr=attentions[j], format='png')
            # print(f"{fname} saved.")

            # # Threshold
            # single_attention = attentions[j]
            # single_attention[single_attention > args.attn_threshold] = 1
            # single_attention[single_attention <= args.attn_threshold] = 0

            # attentions[j] = attentions[j] * float(probability)

            rgb_image = cv2.imread(img_path)[:, :, ::-1]
            # Save probability
            white = (255, 255, 255)
            font = cv2.FONT_HERSHEY_DUPLEX
            if probability > 0.9:
                rgb_image = cv2.putText(rgb_image.copy(),
                                    "Probability: over 90%", (50, 80),
                                    font, 2, white, 2, cv2.LINE_AA)
            else:
                rgb_image = cv2.putText(rgb_image.copy(),
                                    "Probability: {:.1f}%".format(min(float(probability) * 100., 90.)), (50, 80),
                                    font, 2, white, 2, cv2.LINE_AA)
            rgb_image = np.float32(rgb_image) / 255
            gray_attention = attentions[j]
            overlap_image = getAttMap(rgb_image, gray_attention, float(probability))
            overlap_image = (overlap_image - overlap_image.min()) / (overlap_image.max() - overlap_image.min())
            overlap_image = cv2.resize(overlap_image, dsize=(
                int(csv_.loc[csv_.Image == csv].Columns), int(csv_.loc[csv_.Image == csv].Rows)))

            # cv2.imwrite(fname, overlap_image)
            plt.imsave(fname=fname, arr=overlap_image, format='png')
            print(f"{fname} saved.")

        if args.threshold is not None:
            image = skimage.io.imread(os.path.join(output_dir, "img.png"))
            for j in range(nh):
                display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
