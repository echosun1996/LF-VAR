import torch
import os
import math
import argparse
import models_mage
import numpy as np
from tqdm import tqdm
import cv2
import torch.nn.functional as F

def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    masking = (confidence <= cut_off)
    return masking


def gen_image(model, bsz, seed, num_iter=12, choice_temperature=4.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    codebook_emb_dim = 256
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 256
    _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf

    initial_token_indices = mask_token_id * torch.ones(bsz, unknown_number_in_the_beginning)

    token_indices = initial_token_indices.cuda()

    for step in range(num_iter):
        cur_ids = token_indices.clone().long()

        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = model.fake_class_label
        token_indices = token_indices.long()
        token_all_mask = token_indices == mask_token_id

        token_drop_mask = torch.zeros_like(token_indices)

        input_embeddings = model.token_emb(token_indices)

        x = input_embeddings
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :codebook_size]

        sample_dist = torch.distributions.categorical.Categorical(logits=logits)
        sampled_ids = sample_dist.sample()

        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        ratio = 1. * (step + 1) / num_iter

        mask_ratio = np.cos(math.pi / 2. * ratio)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

        selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

        mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
        mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                 torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))

        masking = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1 - ratio))
        token_indices = torch.where(masking, mask_token_id, sampled_ids)

    z_q = model.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(bsz, 16, 16, codebook_emb_dim))
    gen_images = model.vqgan.decode(z_q)
    return gen_images


parser = argparse.ArgumentParser('MAGE generation', add_help=False)
parser.add_argument('--temp', default=4.5, type=float,
                    help='sampling temperature')
parser.add_argument('--num_iter', default=12, type=int,
                    help='number of iterations for generation')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for generation')
parser.add_argument('--output_resize', default=512, type=int,
                    help='output resize')
parser.add_argument('--num_images', default=50000, type=int,
                    help='number of images to generate')
parser.add_argument('--ckpt', type=str,
                    help='checkpoint')
parser.add_argument('--model', default='mage_vit_base_patch16', type=str,
                    help='model')
parser.add_argument('--output_dir', default='output_dir/fid/gen/mage-vitb', type=str,
                    help='name')
parser.add_argument('--vqgan_jax_strongaug', default='', type=str,
                    help='name')
args = parser.parse_args()

vqgan_ckpt_path = args.vqgan_jax_strongaug

model = models_mage.__dict__[args.model](norm_pix_loss=False,
                                         mask_ratio_mu=0.55, mask_ratio_std=0.25,
                                         mask_ratio_min=0.0, mask_ratio_max=1.0,
                                         vqgan_ckpt_path=vqgan_ckpt_path)
model.to(0)

checkpoint = torch.load(args.ckpt, map_location='cpu')

old_pos_embed = checkpoint['model']['pos_embed']
old_shape = old_pos_embed.shape[1]
grid_size = int(math.sqrt(old_shape - 1))
if grid_size ** 2 != old_shape - 1:
    raise ValueError(f"Unexpected shape for pos_embed. Cannot reshape {old_shape} into a square grid.")
cls_token, spatial_tokens = old_pos_embed[:, :1, :], old_pos_embed[:, 1:, :]
spatial_tokens = spatial_tokens.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
new_pos_embed = model.state_dict()['pos_embed']
new_grid_size = int(math.sqrt(new_pos_embed.shape[1] - 1))
spatial_tokens = F.interpolate(spatial_tokens, size=(new_grid_size, new_grid_size), mode='bilinear')
spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size ** 2, -1)
new_pos_embed = torch.cat((cls_token, spatial_tokens), dim=1)
checkpoint['model']['pos_embed'] = new_pos_embed

model.load_state_dict(checkpoint['model'], strict=False)

model.eval()

num_steps = args.num_images // args.batch_size + 1
gen_img_list = []
save_folder = os.path.join(args.output_dir, "temp{}-iter{}".format(args.temp, args.num_iter))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
for i in tqdm(range(num_steps)):
    with torch.no_grad():
        gen_images_batch = gen_image(model, bsz=args.batch_size, seed=i, choice_temperature=args.temp, num_iter=args.num_iter)
    gen_images_batch = gen_images_batch.detach().cpu()
    gen_img_list.append(gen_images_batch)

    for b_id in range(args.batch_size):
        if i*args.batch_size+b_id >= args.num_images:
            break
        gen_img = np.clip(gen_images_batch[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]

        gen_img = cv2.resize(gen_img, (args.output_resize, args.output_resize), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), gen_img)