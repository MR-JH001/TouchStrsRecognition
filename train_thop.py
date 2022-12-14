import argparse
import time

from easydict import EasyDict as edict
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info


from tensorboardX import SummaryWriter

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    if False:
        parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

        args = parser.parse_args()

        with open(args.cfg, 'r') as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.load(f)
            config = edict(config)
    if True:
        path_config_set = r"E:\article_model\CRNN_Chinese_Characters_Rec-stable\lib\config"
        path_config_name = ["OWN_config.yaml"]
        path_config_use = os.path.join(path_config_set,path_config_name[0])

        with open(path_config_use, 'r') as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.load( f )
            config = edict( config )
    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():

    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    from thop import profile
    # input_random = torch.randn(1,1,3,200,32)

    #CNN + BiLSTM
    tic = time.time()
    input_random = torch.randn( 1, 1, 3, 32, 200 )
    flops,paras = profile(model, input_random)
    toc = time.time()
    print((toc - tic) * 1000)
    print(paras/(1000**2))
    print(flops/(1000**3))

    if False:
        # refinenet
        from refinenet import RefineNet
        refinenet_modle = RefineNet()
        tic = time.time()
        input_random = torch.randn( 1, 1, 3, 32, 200 )
        flops,paras = profile(refinenet_modle, input_random)
        toc = time.time()
        print((toc - tic) * 1000)
        print(paras/(1000**2))
        print(flops/(1000**3))
    if True:
        print( "DenseNet:" )
        from densenet import DenseNet
        model = DenseNet()
        tic = time.time()
        input_random = torch.randn(1, 1, 1, 28, 28)
        flops,paras = profile(model, input_random)
        toc = time.time()
        print((toc - tic) * 1000)
        print(paras/(1000**2))
        print(flops/(1000**3))
    if True:
        print("CRAFT:")
        from craft import CRAFT
        model = CRAFT()
        tic = time.time()
        input_random = torch.randn(1, 1, 3, 32, 200)
        # input_random = torch.randn( 1, 1, 3, 800, 800 )
        flops,paras = profile(model, input_random)
        toc = time.time()
        print((toc - tic) * 1000)
        print(paras/(1000**2))
        print(flops/(1000**3))
    if True:
        print("CRAFT ???????????????CRNN")
        import models.crnn_fusion_craft_feature
        model = models.crnn_fusion_craft_feature.get_crnn(config)
        tic = time.time()
        input_random = torch.randn(1, 1, 3, 32, 200)
        # input_random = torch.randn( 1, 1, 3, 800, 800 )
        flops,paras = profile(model, input_random)
        toc = time.time()
        print((toc - tic) * 1000)
        print(paras/(1000**2))
        print(flops/(1000**3))
    exit()
    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()
    #criterion = torch.nn.CrossEntropyLoss()
    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range( last_epoch, config.TRAIN.END_EPOCH ):

        if epoch == 0:
            acc = function.validate( config, val_loader, val_dataset, converter, model, criterion, device, epoch,
                                     writer_dict, output_dict )
        function.train( config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch,
                        writer_dict, output_dict )
        lr_scheduler.step()

        acc = function.validate( config, val_loader, val_dataset, converter, model, criterion, device, epoch,
                                 writer_dict, output_dict )

        is_best = acc > best_acc
        best_acc = max( acc, best_acc )

        print( "is best:", is_best )
        print( "best acc is:", best_acc )
        # save checkpoint
        # torch.save(
        #     {
        #         "state_dict": model.state_dict(),
        #         "epoch": epoch + 1,
        #         # "optimizer": optimizer.state_dict(),
        #         # "lr_scheduler": lr_scheduler.state_dict(),
        #         "best_acc": best_acc,
        #     },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        # )

        # ????????????????????????
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join("best.pth")
        )

    writer_dict['writer'].close()

if __name__ == '__main__':

    main()