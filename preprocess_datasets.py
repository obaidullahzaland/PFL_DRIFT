#!/usr/bin/env python3
import os
import argparse
import pickle
import random

import numpy as np
from PIL import Image, ImageFile


from torchvision.datasets import SVHN, CIFAR10
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
DATASETS = {
    'CIFAR10-C': [],  # we’ll glob *.npy
    'CIFAR10':   ['CIFAR10'],
    'Digit':     ['MNIST','MNIST_M','SVHN','SynthDigits','USPS'],
    'FairFace': ['Black','East Asian','Indian','Latino_Hispanic','Middle Eastern','Southeast Asian','White'],
    'OfficeHome': ['Art','Clipart','Product','Real World'],
    'PACS':      ['Photo','ArtPainting','Cartoon','Sketch'],
    'VLCS':      ['Caltech101','LabelMe','VOC2007','SUN09'],
    'DomainNet': ['clipart','infograph','painting','quickdraw','real','sketch']
}

RESIZE = {
    'CIFAR10-C': (32,32),
    'Digit':     (32,32),
    'default224': (224,224)
}

def save_pkl(path, data, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump((data, labels), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ↳ Saved {path} (N={len(labels)})")

def load_folder_images(folder, resize):
    imgs, labs = [], []
    for cls in sorted(os.listdir(folder)):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir): continue
        label = int(cls)
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
            img = Image.open(os.path.join(cls_dir, fn)).convert('RGB')
            img = img.resize(resize, Image.BICUBIC)
            imgs.append(np.array(img))
            labs.append(label)
    if not labs:
        raise RuntimeError(f"No images in {folder}")
    return np.stack(imgs), np.array(labs, dtype=np.int64)

def process_cifar10c(raw_root, out_root):

    base_raw = os.path.join(raw_root, "CIFAR-10-C")
    labels   = np.load(os.path.join(base_raw, "labels.npy"))  # (10000,)

    for fname in sorted(os.listdir(base_raw)):
        if not fname.endswith(".npy") or fname == "labels.npy":
            continue
        corruption = fname[:-4]  # e.g. "brightness"

        # load raw data
        raw_arr = np.load(os.path.join(base_raw, fname))  # shape (50000,32,32,3)
        # reshape into (5,10000,32,32,3)
        n, h, w, c = raw_arr.shape
        assert n % 10000 == 0, f"Unexpected first dim {n}"
        sev = n // 10000
        arr = raw_arr.reshape(sev, 10000, h, w, c)
        # pick severity 4 (0-indexed)
        lvl4 = arr[3]  # shape (10000,32,32,3)

        # split into test/train
        test_imgs,  test_labs  = lvl4[:500],  labels[:500]
        train_imgs, train_labs = lvl4[500:],  labels[500:]

        # write out
        client_dir = os.path.join(out_root, "CIFAR10-C", corruption)
        train_dir  = os.path.join(client_dir, "partitions")
        os.makedirs(train_dir, exist_ok=True)

        print(f"{corruption:16s}  train={train_imgs.shape[0]}  test={test_imgs.shape[0]}")

        save_pkl(os.path.join(client_dir, "test.pkl"),         test_imgs,  test_labs)
        save_pkl(os.path.join(train_dir, "train_part0.pkl"), train_imgs, train_labs)



def process_domainnet(raw_root, out_root, resize=(224,224)):
    print("→ DomainNet")
    domains = ["clipart","infograph","painting","quickdraw","real","sketch"]
    base = os.path.join(raw_root, "domainNet")

    for dom in domains:
        print(f"  • Domain '{dom}'")
        train_txt = os.path.join(base, f"{dom}_train.txt")
        test_txt  = os.path.join(base, f"{dom}_test.txt")

        # prepare output dirs
        train_out = os.path.join(out_root, "DomainNet", dom, "partitions")
        test_out  = os.path.join(out_root, "DomainNet", dom)
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(test_out,  exist_ok=True)

        # helper to load split
        def load_split(txt_path):
            img_paths, labels = [], []
            with open(txt_path, 'r') as f:
                for line in f:
                    rel, lab = line.strip().split()
                    full = os.path.join(base, rel)   # raw_root/domainnet/rel
                    img_paths.append(full)
                    labels.append(int(lab))
            # read & resize
            imgs = []
            for p in img_paths:
                img = Image.open(p).convert('RGB')
                img = img.resize(resize, Image.BICUBIC)
                imgs.append(np.array(img))
            return np.stack(imgs), np.array(labels, dtype=np.int64)

        # load train / test
        train_imgs, train_labs = load_split(train_txt)
        test_imgs,  test_labs  = load_split(test_txt)

        # save
        save_pkl(
            os.path.join(train_out, "train_part0.pkl"),
            train_imgs, train_labs
        )
        save_pkl(
            os.path.join(test_out, "test.pkl"),
            test_imgs, test_labs
        )

        print(f"     Train: {len(train_labs)} samples, Test: {len(test_labs)} samples")
def process_officehome(raw_root, out_root, test_frac=0.2, resize=(224,224), seed=42):
    print("→ Office-Home (20% per-category test split)")
    domains = ['Art','Clipart','Product','Real World']
    base_dir = os.path.join(raw_root, 'OfficeHome')

    first_dom = domains[0]
    cat_dir = os.path.join(base_dir, first_dom)
    categories = sorted(
        [d for d in os.listdir(cat_dir)
         if os.path.isdir(os.path.join(cat_dir, d))]
    )
    cat2label = {cat: idx for idx, cat in enumerate(categories)}

    random.seed(seed)

    for dom in domains:
        print(f"  • Domain '{dom}'")
        # prep output dirs
        train_out = os.path.join(out_root,'OfficeHome',dom,'partitions')
        test_out  = os.path.join(out_root,'OfficeHome',dom)
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(test_out,  exist_ok=True)

        train_imgs, train_labs = [], []
        test_imgs,  test_labs  = [], []

        for cat in categories:
            src = os.path.join(base_dir, dom, cat)
            all_files = [
                f for f in os.listdir(src)
                if f.lower().endswith(('.jpg','.jpeg','.png'))
            ]
            random.shuffle(all_files)
            n_total = len(all_files)
            n_test  = math.ceil(test_frac * n_total)

            # test slice
            for fn in all_files[:n_test]:
                img = Image.open(os.path.join(src, fn)).convert('RGB')
                img = img.resize(resize, Image.BICUBIC)
                test_imgs.append(np.array(img))
                test_labs.append(cat2label[cat])

            # train slice
            for fn in all_files[n_test:]:
                img = Image.open(os.path.join(src, fn)).convert('RGB')
                img = img.resize(resize, Image.BICUBIC)
                train_imgs.append(np.array(img))
                train_labs.append(cat2label[cat])

        train_arr = np.stack(train_imgs)
        train_lbl = np.array(train_labs, dtype=np.int64)
        save_pkl(
            os.path.join(train_out, 'train_part0.pkl'),
            train_arr, train_lbl
        )

        test_arr  = np.stack(test_imgs)
        test_lbl  = np.array(test_labs, dtype=np.int64)
        save_pkl(
            os.path.join(test_out, 'test.pkl'),
            test_arr, test_lbl
        )

        print(f"     Categories: {len(categories)}, "
              f"Train samples: {len(train_lbl)}, Test samples: {len(test_lbl)}")



def process_pacs(raw_root, out_root, resize=(224,224)):
    print("→ PACS")
    domains = ['art_painting','cartoon','photo','sketch']
    base_imgs = os.path.join(raw_root, 'PACS', 'pacs_data', 'pacs_data')
    label_dir = os.path.join(raw_root, 'PACS', 'pacs_label')

    for dom in domains:
        print(f"  • Domain '{dom}'")
        # create output dirs
        train_out = os.path.join(out_root, 'PACS', dom, 'partitions')
        test_out  = os.path.join(out_root, 'PACS', dom)
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(test_out,  exist_ok=True)

        def load_split(txt_name):
            imgs, labs = [], []
            path_txt = os.path.join(label_dir, f"{dom}_{txt_name}_kfold.txt")
            if not os.path.isfile(path_txt):
                raise FileNotFoundError(f"{path_txt} not found")
            with open(path_txt, 'r') as f:
                for line in f:
                    rel, lbl = line.strip().split()
                    lbl = int(lbl)
                    img_path = os.path.join(base_imgs, rel)
                    if not os.path.isfile(img_path):
                        continue
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(resize, Image.BICUBIC)
                    imgs.append(np.array(img))
                    labs.append(lbl)
            if not labs:
                raise RuntimeError(f"No images loaded for {dom} {txt_name}")
            return np.stack(imgs), np.array(labs, dtype=np.int64)

        tr_imgs, tr_labs = load_split('train')
        save_pkl(os.path.join(train_out, 'train_part0.pkl'), tr_imgs, tr_labs)

        te_imgs, te_labs = load_split('test')
        save_pkl(os.path.join(test_out, 'test.pkl'), te_imgs, te_labs)

        print(f"     train: {len(tr_labs)} samples, test: {len(te_labs)} samples")
def process_vlcs(raw_root, out_root, test_frac=0.2, resize=(224, 224), seed=42):
    print("→ VLCS (20% per-category test split)")
    domains = ['Caltech101','LabelMe','VOC2007','SUN09']
    base_dir = os.path.join(raw_root, 'VLCS')

    random.seed(seed)

    for dom in domains:
        print(f"  • Domain '{dom}'")

        domain_dir = os.path.join(base_dir, dom)
        categories = sorted([
            d for d in os.listdir(domain_dir)
            if os.path.isdir(os.path.join(domain_dir, d))
        ])
        cat2label = {cat: idx for idx, cat in enumerate(categories)}

        train_imgs, train_labs = [], []
        test_imgs, test_labs = [], []

        for cat in categories:
            cat_dir = os.path.join(domain_dir, cat)
            img_files = [
                f for f in os.listdir(cat_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            random.shuffle(img_files)
            n_total = len(img_files)
            n_test = math.ceil(test_frac * n_total)

            # Test
            for fn in img_files[:n_test]:
                img_path = os.path.join(cat_dir, fn)
                img = Image.open(img_path).convert('RGB').resize(resize, Image.BICUBIC)
                test_imgs.append(np.array(img))
                test_labs.append(cat2label[cat])

            # Train
            for fn in img_files[n_test:]:
                img_path = os.path.join(cat_dir, fn)
                try:
                    img = Image.open(img_path).convert('RGB').resize(resize, Image.BICUBIC)
                except Exception as e:
                    print(f"Skipping image {img_path}: {e}")
                    continue
                # img = Image.open(img_path).convert('RGB').resize(resize, Image.BICUBIC)
                train_imgs.append(np.array(img))
                train_labs.append(cat2label[cat])

        # Save
        train_out = os.path.join(out_root, 'VLCS', dom, 'partitions')
        test_out = os.path.join(out_root, 'VLCS', dom)
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(test_out, exist_ok=True)

        save_pkl(os.path.join(train_out, 'train_part0.pkl'),
                 np.stack(train_imgs), np.array(train_labs, dtype=np.int64))
        save_pkl(os.path.join(test_out, 'test.pkl'),
                 np.stack(test_imgs), np.array(test_labs, dtype=np.int64))

        print(f"     Categories: {len(categories)}, "
              f"Train samples: {len(train_labs)}, Test samples: {len(test_labs)}")
def main():
    p = argparse.ArgumentParser(
        description="Preprocess concept-drift FL datasets"
    )
    p.add_argument('dataset', choices=list(DATASETS.keys())+['all'])
    p.add_argument('--raw_root', default='../raw_data/raw_data')
    p.add_argument('--out_root', default='../data')
    args = p.parse_args()

    os.makedirs(args.raw_root, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)

    to_run = DATASETS.keys() if args.dataset=='all' else [args.dataset]
    for ds in to_run:
        if ds == 'CIFAR10-C':
            process_cifar10c(args.raw_root, args.out_root)
        elif ds in ['OfficeHome']:
            process_officehome(args.raw_root, args.out_root)
        elif ds in ['DomainNet']:
            process_domainnet("/proj/cloudrobotics-nest/users/NICO++/FL_oneshot/", args.out_root)
        elif ds in ['PACS']:
            process_pacs(args.raw_root, args.out_root)
        elif ds in ['VLCS']:
            process_vlcs(raw_root=args.raw_root, out_root=args.out_root)
        else:
            raise ValueError(f"Unknown dataset {ds}")

if __name__=='__main__':
    main()
