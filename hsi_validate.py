import argparse
import warnings
import matplotlib.pyplot as plt
import torch.utils.data
from utility import *
from hsi_setup import train_options, Engine
from functools import partial
from torchvision.transforms import Compose

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    """Train Settings"""
    parser = argparse.ArgumentParser(description="Hyperspectral Image Denosing")
    opt = train_options(parser)

    opt.no_log = True
    opt.no_ropt = True
    print(f'opt settings: {opt}')

    """Set Random Status"""
    seed_everywhere(opt.seed)

    """Setup Engine"""
    engine = Engine(opt)
    # print(engine.net)
    print('model params: %.2f' % (sum([t.nelement() for t in engine.net.parameters()])/10**6))
    print('==> Preparing data..')

    """Validation Data Settings"""
    basefolder = './data/'
    mat_names = ['icvl_512_gauss_validation', 'icvl_512_complex_validation']
    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name)) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[:, ...][None]),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt'),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

    mat_loaders = [torch.utils.data.DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]
    print('epoch: ', engine.epoch)
    engine.validate(mat_loaders[0], mat_names[0])
