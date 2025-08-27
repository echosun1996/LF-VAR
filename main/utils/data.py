import os
import os.path as osp

import PIL.Image as PImage
import pandas as pd
import torch
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
def normalize_01_into_pm1(x):
    return x.add(x).add_(-1)


from torch.utils.data import Dataset


def pil_loader(path):
    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_loader_mask(path):
    
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class DatasetWithMasks(DatasetFolder):
    def _get_mask_path(self, image_path):
        
        return image_path.replace('_img_class', '_seg_class').replace('.jpg', '_segmentation.png')

    def _get_img_path(self, image_path):
        
        return image_path

    def __init__(self, root, loader, mask_loader, extensions, csv_path=None, transform=None, mask_transform=None):
        super().__init__(root, loader, extensions, transform)
        self.mask_loader = mask_loader
        self.mask_transform = mask_transform
        
        if csv_path and os.path.exists(csv_path):
            print(f"Using class mapping from CSV file: {csv_path}")

            self.class_df = pd.read_csv(csv_path)

            self.samples = [
                (path, class_idx)
                for path, class_idx in self.samples
                if '_img_class' in path
            ]


            self.filename_to_radiomics = {}
            for _, row in self.class_df.iterrows():
                filename = os.path.basename(row['file_name'])
                feature_cols = [col for col in row.index if col.startswith('original_')]
                radiomics_features = row[feature_cols].astype(float).values
                self.filename_to_radiomics[filename] = torch.tensor(radiomics_features, dtype=torch.float32)


        else:
            print("Using default folder-based class mapping")
            self.samples = [
                (path, class_idx)
                for path, class_idx in self.samples
                if '_img_class' in path
            ]

        self.mask_samples = [
            (self._get_mask_path(path), class_idx)
            for path, class_idx in self.samples
        ]

    def __getitem__(self, index):
        path, target = self.samples[index]
        mask_path, _ = self.mask_samples[index]
        
        radiomics_features = self.filename_to_radiomics.get(os.path.basename(path))

        sample = self.loader(path)
        mask = self.mask_loader(mask_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return sample, mask, target, path, radiomics_features
    


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,args=None
):
    mid_reso = round(mid_reso * final_reso)
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())

    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)




    print("args.dataset",str(args.dataset))
    if args.dataset is not None:
        print("=================")
        print("data_path",data_path)

        train_set = DatasetWithMasks(
            root=osp.join(data_path, 'test', 'HuggingFace'),
            loader=pil_loader,
            mask_loader=pil_loader_mask,
            extensions=IMG_EXTENSIONS,
            csv_path=osp.join(data_path, 'radiomics.csv'),
            transform=train_aug,
            mask_transform=train_aug
        )

        val_set = DatasetWithMasks(
            root=osp.join(data_path, 'test', 'HuggingFace'),
            loader=pil_loader,
            mask_loader=pil_loader_mask,
            extensions=IMG_EXTENSIONS,
            csv_path=osp.join(data_path, 'radiomics.csv'),
            transform=val_aug,
            mask_transform=val_aug
        )

    else:
        train_set = DatasetWithMasks(
            root=osp.join(data_path, 'train', 'HuggingFace'),
            loader=pil_loader,
            mask_loader=pil_loader_mask,
            extensions=IMG_EXTENSIONS,
            csv_path=osp.join(data_path, 'radiomics_finial.csv'),
            transform=train_aug,
            mask_transform=train_aug
        )

        val_set = DatasetWithMasks(
            root=osp.join(data_path, 'val', 'HuggingFace'),
            loader=pil_loader,
            mask_loader=pil_loader_mask,
            extensions=IMG_EXTENSIONS,
            csv_path=osp.join(data_path, 'radiomics_finial.csv'),
            transform=val_aug,
            mask_transform=val_aug
        )

    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')