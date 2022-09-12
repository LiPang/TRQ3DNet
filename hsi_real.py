'''
real world image denosing
'''
import argparse
from hsi_setup import Engine, train_options
from utility import *
from torchvision.transforms import Compose
import torch
from torch.utils.data import DataLoader
import warnings
import os

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    """Train Settings"""
    parser = argparse.ArgumentParser(description="Hyperspectral Image Denosing")
    opt = train_options(parser)
    opt.no_log = True
    opt.no_ropt = True
    cuda = not opt.no_cuda
    print(f'opt settings: {opt}')

    """Set Random Status"""
    seed_everywhere(opt.seed)

    """Setup Engine"""
    engine = Engine(opt)
    # mat_dataset = MatDataFromFolder('./images/Indian_pines', fns=['Indian_pines.mat'])
    # key = 'indian_pines'
    mat_dataset = MatDataFromFolder('./images/Urban', fns=['Urban.mat'])
    key = 'a'
    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatKey(key=key), # for testing
            lambda x: x[...][None],
            minmax_normalize,
        ])
    else:
        mat_transform = Compose([
            LoadMatKey(key=key), # for testing
            minmax_normalize,
        ])

    mat_dataset = TransformDataset(mat_dataset, mat_transform)

    mat_loader = DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=cuda
                )
    saveimgdir = './images'
    engine.image_denosing(mat_loader, saveimgdir=saveimgdir)


