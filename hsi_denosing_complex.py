import argparse
import warnings

import torch.utils.data
from utility import *
from hsi_setup import train_options, Engine
from functools import partial
from torchvision.transforms import Compose
sigmas = [10, 30, 50, 70]
if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    """Train Settings"""
    parser = argparse.ArgumentParser(description="Hyperspectral Image Denosing")
    opt = train_options(parser)
    opt.epochs = 100
    print(f'opt settings: {opt}')

    """Set Random Status"""
    seed_everywhere(opt.seed)

    """Setup Engine"""
    engine = Engine(opt)
    # print(engine.net)
    print('==> Preparing data..')

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    add_noniid_noise = Compose([
        AddNoiseNoniid(sigmas),
        SequentialSelect(
            transforms=[
                lambda x: x,
                AddNoiseImpulse(),
                AddNoiseStripe(),
                AddNoiseDeadline()
            ]
        )
    ])

    common_transform = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()

    train_transform = Compose([
        add_noniid_noise,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    icvl_64_31_TL = make_dataset(
        opt, train_transform,
        target_transform, common_transform, 64)

    """Test-Dev"""
    basefolder = opt.valroot
    mat_names = os.listdir(basefolder)

    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name), size=5) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                       transform=lambda x: x[:, ...][None]),
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
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]

    base_lr = opt.lr
    adjust_learning_rate(engine.optimizer, base_lr)
    if opt.resume:
        if engine.epoch > 85 and engine.epoch < 95:
            adjust_learning_rate(engine.optimizer, base_lr * 0.1)
        if engine.epoch > 95:
            adjust_learning_rate(engine.optimizer, base_lr * 0.01)

    rand_status_list = np.random.get_state()[1].tolist()
    epoch_per_save = 10
    # from epoch 50 to 100
    while engine.epoch < opt.epochs:
        np.random.seed(rand_status_list[engine.epoch % len(rand_status_list)])

        if engine.epoch == 85:
            adjust_learning_rate(engine.optimizer, base_lr * 0.1)

        if engine.epoch == 95:
            adjust_learning_rate(engine.optimizer, base_lr * 0.01)

        engine.train(icvl_64_31_TL)

        for mat_name, mat_loader in zip(mat_names, mat_loaders):
            engine.validate(mat_loader, mat_name)

        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

