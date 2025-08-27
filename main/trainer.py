import time
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
import torch, torchvision
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import torchvision.transforms as transforms
from PIL import Image
import os
import PIL

from tqdm import tqdm
class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    @torch.no_grad()
    def cross_infer(self, ld_val: DataLoader, args):
        if args.fixed_csv_path is not None:
            tot = 0
            L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
            stt = time.time()
            training = self.var_wo_ddp.training
            self.var_wo_ddp.eval()
            resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])

            total_output_count = 0
            progress_bar = tqdm(total=73500, desc=f"Generating...", ncols=100)

            name_dict = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
            class_dict = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


            count_dict = {}

            for i in range(7):
                for j in range(7):
                    count_dict[f"{name_dict.get(i)}_{name_dict.get(j)}"] = 0

            for inp_B3HW, mask_BHW, _, img_path, radiomics in ld_val:
                label_B = []
                for path in img_path:
                    labels = [i for i, class_name in enumerate(class_dict) if class_name in path]
                    label_B.extend(labels)
                if len(label_B) != len(img_path):
                    print("len label_B", len(label_B))
                    print("len img_path", len(img_path))
                    print("img_path", img_path)
                    exit(-1)
                label_B = torch.tensor(label_B)

                if args.fixed_csv_path != None:
                    csv_path = args.fixed_csv_path
                    df = pd.read_csv(csv_path)
                    radiomics_columns = [col for col in df.columns if col.startswith("original_")]
                    radiomics_data = df.set_index("category")[radiomics_columns]
                    radiomics_features = torch.tensor(radiomics_data.values, dtype=torch.float32)
                else:
                    print("fixed_csv_path not specified")
                    exit(-1)


                B, V = label_B.shape[0], self.vae_local.vocab_size
                inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
                mask_BHW = mask_BHW.to(dist.get_device(), non_blocking=True)

                binary_mask_BHW = (mask_BHW >= 0.5).to(dtype=torch.float32)
                binary_mask_BHW = 1 - binary_mask_BHW
                mask_BHW = binary_mask_BHW.to(mask_BHW.device, non_blocking=True)

                label_B = label_B.to(dist.get_device(), non_blocking=True)

                gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

                self.var_wo_ddp.forward
                logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l, radiomics)
                L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
                L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                        gt_BL[:, -self.last_l:].reshape(-1)) * B
                acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
                acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (
                        100 / self.last_l)
                tot += B

                lbl_temp = str(label_B[0].item())
                if lbl_temp == '0':
                    random_range = 60
                elif lbl_temp == '1':
                    random_range = 40
                elif lbl_temp == '2':
                    random_range = 500
                elif lbl_temp == '3':
                    random_range = 300
                elif lbl_temp == '4':
                    random_range = 100
                elif lbl_temp == '5':
                    random_range = 10
                elif lbl_temp == '6':
                    random_range = 150
                else:
                    random_range = 150
                for random_seed in range(random_range):
                    for target_id in range(7):
                        target_name = name_dict[target_id]
                        target_radiomics = radiomics_features[target_id]
                        all_target_radiomics = torch.cat([target_radiomics.unsqueeze(0)] * B, dim=0)

                        target_label_B = []
                        for _ in range(B):
                            target_label_B.append(int(target_id))
                        target_label_B = torch.tensor(target_label_B)
                        target_label_B = target_label_B.to(dist.get_device(), non_blocking=True)

                        recon_B3HW = self.var_wo_ddp.autoregressive_infer_cfg(B=B, label_B=target_label_B, g_seed=random_seed,
                                                                            cfg=3, top_k=900, top_p=0.95, more_smooth=False,
                                                                            input_img_tokens=gt_idx_Bl, edit_mask=mask_BHW,
                                                                            save_path=args.local_out_dir_path,
                                                                            img_path=img_path, radiomics_features=all_target_radiomics)
                        exi = True

                        for idx, img_tensor in enumerate(recon_B3HW):
                            img_name = os.path.splitext(os.path.basename(img_path[idx]))[0].replace("_segmentation", "")

                            chw = torchvision.utils.make_grid(img_tensor, nrow=1, padding=0, pad_value=0)
                            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
                            lbl = label_B[idx].item()
                            save_folder_name = f"{str(name_dict.get(lbl))}_{target_name}"

                            progress_bar.set_description(f"Generating [{save_folder_name}]: {str(count_dict.get(save_folder_name))}...")
                            folder_path = os.path.join(args.local_out_dir_path, save_folder_name)
                            if os.path.exists(folder_path) and count_dict.get(save_folder_name) == 0:
                                png_count = len([f for f in os.listdir(folder_path) if
                                                f.lower().endswith('.png') and os.path.isfile(os.path.join(folder_path, f))])
                                count_dict.update({save_folder_name: png_count})
                                continue
                            os.makedirs(folder_path, exist_ok=True)

                            save_path = os.path.join(folder_path, f"{img_name}_{str(count_dict.get(save_folder_name)).zfill(5)}.png")

                            chw = PImage.fromarray(chw.astype(np.uint8))
                            chw = chw.resize((512, 512), PIL.Image.LANCZOS)
                            if count_dict.get(save_folder_name) >= 1500:
                                continue
                            exi = False
                            chw.save(save_path)
                            total_output_count += 1
                            count_dict.update({save_folder_name: count_dict.get(save_folder_name) + 1})
                            progress_bar.update(1)
                        if exi:
                            break

            self.var_wo_ddp.train(training)

            stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
            dist.allreduce(stats)
            tot = round(stats[-1].item())
            stats /= tot
            L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
            return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt
        
        elif args.tabsyn_csv_path is not None:
            tot = 0
            L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
            stt = time.time()
            training = self.var_wo_ddp.training
            self.var_wo_ddp.eval()
            resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])

            total_output_count = 0
            progress_bar = tqdm(total=73500, desc=f"Generating...", ncols=100)

            name_dict = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
            class_dict = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

            count_dict = {}

            for i in range(0, 7):
                for j in range(0, 7):
                    count_dict[f"{name_dict.get(i)}_{name_dict.get(j)}"] = 0

            for inp_B3HW, mask_BHW, _, img_path, radiomics in ld_val:
                label_B = []
                for path in img_path:
                    labels = [i for i, class_name in enumerate(class_dict) if class_name in path]
                    label_B.extend(labels)
                if len(label_B) != len(img_path):
                    print("len label_B", len(label_B))
                    print("len img_path", len(img_path))
                    print("img_path", img_path)
                    exit(-1)
                label_B = torch.tensor(label_B)

                if args.tabsyn_csv_path != None:
                    csv_path = args.tabsyn_csv_path
                    df = pd.read_csv(csv_path)
                    radiomics_columns = [col for col in df.columns if col.startswith("original_")]
                    df['category_int'] = df['category'].astype(int) - 1
                    unique_categories = sorted(df['category_int'].unique())
                    radiomics_features = []

                    for category in unique_categories:
                        category_data = df[df['category_int'] == category][radiomics_columns]
                        category_tensor = torch.tensor(category_data.values, dtype=torch.float32)
                        radiomics_features.append(category_tensor)


                else:
                    print("tabsyn_csv_path not specified")
                    exit(-1)


                B, V = label_B.shape[0], self.vae_local.vocab_size
                inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
                mask_BHW = mask_BHW.to(dist.get_device(), non_blocking=True)

                binary_mask_BHW = (mask_BHW >= 0.5).to(dtype=torch.float32)
                binary_mask_BHW = 1 - binary_mask_BHW
                mask_BHW = binary_mask_BHW.to(mask_BHW.device, non_blocking=True)

                label_B = label_B.to(dist.get_device(), non_blocking=True)

                gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

                self.var_wo_ddp.forward
                logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l, radiomics)
                L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
                L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                        gt_BL[:, -self.last_l:].reshape(-1)) * B
                acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
                acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (
                        100 / self.last_l)
                tot += B

                lbl_temp = str(label_B[0].item())
                if lbl_temp == '0':
                    random_range = 60
                elif lbl_temp == '1':
                    random_range = 40
                elif lbl_temp == '2':
                    random_range = 500
                elif lbl_temp == '3':
                    random_range = 300
                elif lbl_temp == '4':
                    random_range = 100
                elif lbl_temp == '5':
                    random_range = 10
                elif lbl_temp == '6':
                    random_range = 150
                else:
                    random_range = 150
                for random_seed in range(random_range):
                    for target_id in range(0, 7):
                        target_name = name_dict[target_id]
                        feature_index = random_seed % radiomics_features[target_id].size(0)
                        target_radiomics = radiomics_features[target_id][feature_index]

                        all_target_radiomics = torch.cat([target_radiomics.unsqueeze(0)] * B, dim=0)

                        target_label_B = []
                        for _ in range(B):
                            target_label_B.append(int(target_id))
                        target_label_B = torch.tensor(target_label_B)
                        target_label_B = target_label_B.to(dist.get_device(), non_blocking=True)

                        recon_B3HW = self.var_wo_ddp.autoregressive_infer_cfg(B=B, label_B=target_label_B, g_seed=random_seed,
                                                                            cfg=3, top_k=900, top_p=0.95, more_smooth=False,
                                                                            input_img_tokens=gt_idx_Bl, edit_mask=mask_BHW,
                                                                            save_path=args.local_out_dir_path,
                                                                            img_path=img_path, radiomics_features=all_target_radiomics)
                        exi = True

                        for idx, img_tensor in enumerate(recon_B3HW):
                            img_name = os.path.splitext(os.path.basename(img_path[idx]))[0].replace("_segmentation", "")

                            chw = torchvision.utils.make_grid(img_tensor, nrow=1, padding=0, pad_value=0)
                            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
                            lbl = label_B[idx].item()
                            save_folder_name = f"{str(name_dict.get(lbl))}_{target_name}"

                            progress_bar.set_description(f"Generating [{save_folder_name}]: {str(count_dict.get(save_folder_name))}...")
                            folder_path = os.path.join(args.local_out_dir_path, save_folder_name)
                            if os.path.exists(folder_path) and count_dict.get(save_folder_name) == 0:
                                png_count = len([f for f in os.listdir(folder_path) if
                                                f.lower().endswith('.png') and os.path.isfile(os.path.join(folder_path, f))])
                                count_dict.update({save_folder_name: png_count})
                                continue
                            os.makedirs(folder_path, exist_ok=True)

                            save_path = os.path.join(folder_path, f"{img_name}_{str(count_dict.get(save_folder_name)).zfill(5)}.png")

                            chw = PImage.fromarray(chw.astype(np.uint8))
                            chw = chw.resize((512, 512), PIL.Image.LANCZOS)
                            if count_dict.get(save_folder_name) >= 1500:
                                continue
                            exi = False
                            chw.save(save_path)
                            total_output_count += 1
                            count_dict.update({save_folder_name: count_dict.get(save_folder_name) + 1})
                            progress_bar.update(1)
                        if exi:
                            break

            self.var_wo_ddp.train(training)

            stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
            dist.allreduce(stats)
            tot = round(stats[-1].item())
            stats /= tot
            L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
            return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt


    @torch.no_grad()
    def infer(self, ld_val: DataLoader, args, radiomics_enable=False):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        total_output_count = 0
        progress_bar = tqdm(total=10500, desc=f"Generating...", ncols=100)

        name_dict = {0:'akiec',1:'bcc',2:'bkl',3:'df',4:'mel',5:'nv',6:'vasc'}
        class_dict = ['akiec', 'bcc','bkl', 'df', 'mel', 'nv', 'vasc']

        count_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}

        for inp_B3HW, mask_BHW, _, img_path,radiomics in ld_val:
            label_B = []
            for path in img_path:
                labels = [i for i, class_name in enumerate(class_dict) if class_name in path]
                label_B.extend(labels)
            if len(label_B) != len(img_path):
                print("len label_B",len(label_B))
                print("len img_path",len(img_path))
                print("img_path",img_path)
                exit(-1)
            label_B = torch.tensor(label_B)

            if args.fixed_csv_path !=None:
                print("Generating fixed radiomics...")
                csv_path = args.fixed_csv_path
                df = pd.read_csv(csv_path)
                radiomics_columns = [col for col in df.columns if col.startswith("original_")]
                radiomics_data = df.set_index("category")[radiomics_columns]
                radiomics_tensors = []
                for label in label_B.tolist():
                    category_name = name_dict[label]
                    if category_name in radiomics_data.index:
                        radiomics_features = torch.tensor(radiomics_data.loc[category_name].values, dtype=torch.float32)
                        radiomics_tensors.append(radiomics_features)
                    else:
                        raise ValueError(f"Category {category_name} not found in CSV data.")

                radiomics = torch.stack(radiomics_tensors)


            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            mask_BHW = mask_BHW.to(dist.get_device(), non_blocking=True)

            binary_mask_BHW = (mask_BHW >= 0.5).to(dtype=torch.float32)
            binary_mask_BHW = 1 - binary_mask_BHW
            mask_BHW = binary_mask_BHW.to(mask_BHW.device, non_blocking=True)

            label_B = label_B.to(dist.get_device(), non_blocking=True)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l,radiomics)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                    gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (
                        100 / self.last_l)
            tot += B

            lbl_temp = str(label_B[0].item())
            if lbl_temp == '0':
                random_range=60
            elif lbl_temp =='1':
                random_range=40
            elif lbl_temp =='2':
                random_range=500
            elif lbl_temp =='3':
                random_range=300
            elif lbl_temp =='4':
                random_range=100
            elif lbl_temp =='5':
                random_range=10
            elif lbl_temp =='6':
                random_range=150
            else:
                random_range=150
            for random_seed in range(random_range):
                recon_B3HW = self.var_wo_ddp.autoregressive_infer_cfg(B=B, label_B=label_B, g_seed=random_seed,
                                                                      cfg=3, top_k=900, top_p=0.95, more_smooth=False,
                                                                      input_img_tokens=gt_idx_Bl, edit_mask=mask_BHW,
                                                                      save_path=args.local_out_dir_path,img_path=img_path,radiomics_features=radiomics)
                exi = True

                for idx, img_tensor in enumerate(recon_B3HW):
                    img_name = os.path.splitext(os.path.basename(img_path[idx]))[0].replace("_segmentation","")

                    chw = torchvision.utils.make_grid(img_tensor, nrow=1, padding=0, pad_value=0)
                    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
                    lbl = label_B[idx].item()

                    progress_bar.set_description(f"Generating [{str(name_dict.get(lbl))}]...")
                    folder_path = os.path.join(args.local_out_dir_path, str(name_dict.get(lbl)))
                    if os.path.exists(folder_path) and count_dict.get(lbl) == 0:
                        png_count = len([f for f in os.listdir(folder_path) if
                                         f.lower().endswith('.png') and os.path.isfile(os.path.join(folder_path, f))])
                        count_dict.update({lbl: png_count})
                        continue
                    os.makedirs(folder_path, exist_ok=True)

                    save_path = os.path.join(folder_path, f"{img_name}_{str(count_dict.get(lbl)).zfill(5)}.png")

                    chw = PImage.fromarray(chw.astype(np.uint8))
                    chw = chw.resize((512, 512), PIL.Image.LANCZOS)
                    if count_dict.get(lbl) >=1500:
                        continue
                    exi = False
                    chw.save(save_path)
                    total_output_count += 1
                    count_dict.update({lbl: count_dict.get(lbl)+1})
                    progress_bar.update(1)
                if exi:
                    break

        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, mask_BHW, label_B, img_path,radiomics in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l,radiomics)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float, radiomics_features
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1
        if prog_si == len(self.patch_nums) - 1: prog_si = -1
        
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l,radiomics_features)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:
                Ltail = acc_tail = -1
            else:
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)