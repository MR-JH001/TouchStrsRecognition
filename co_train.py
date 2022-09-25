

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.backends.cudnn as cudnn
import subprocess
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import lib.config.alphabets as alphabets
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.core import function
from lib.dataset import get_dataset
from lib.utils.utils import model_info


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config


def train(model, config, dataset_num: int = 1):
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
    # model = crnn.get_crnn(config)

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
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        ###
        # if dataset_num == 1:
        #     FINETUNE_CHECKPOINIT = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT1
        # elif dataset_num == 2:
        #     FINETUNE_CHECKPOINIT = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT2
        # elif dataset_num == 3:
        #     FINETUNE_CHECKPOINIT = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT3

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
        if dataset_num == 1:
            model_state_file = config.TRAIN.RESUME.FILE1
        elif dataset_num == 2:
            model_state_file = config.TRAIN.RESUME.FILE2
        elif dataset_num == 3:
            model_state_file = config.TRAIN.RESUME.FILE3

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
    train_dataset = get_dataset(config)(config, is_train=True, dataset_num=dataset_num)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False, dataset_num=dataset_num)
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

        print("is best:", is_best)
        print("best acc is:", best_acc)
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

        # Add save fixed path
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(f'co_train_model{dataset_num}.pth')
        )

    writer_dict['writer'].close()


def co_train(config, model1, model2, model3, epoch):
    # create output folder
    output_dict = utils.create_log_folder(config, phase='co_train')

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
    device = torch.device(f"cuda:{format(config.GPUID)}" if torch.cuda.is_available() else "cpu")

    model1.to(device)
    model2.to(device)
    model3.to(device)
    # define loss function
    criterion = torch.nn.CTCLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model1)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    model_state_file1 = config.TRAIN.RESUME.FILE1
    model_state_file2 = config.TRAIN.RESUME.FILE2
    model_state_file3 = config.TRAIN.RESUME.FILE3

    checkpoint1 = torch.load(model_state_file1, map_location='cpu')
    checkpoint2 = torch.load(model_state_file2, map_location='cpu')
    checkpoint3 = torch.load(model_state_file3, map_location='cpu')
    model1.load_state_dict(checkpoint1['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    model3.load_state_dict(checkpoint3['state_dict'])
    
    torch.save(
        {
            "state_dict": checkpoint1['state_dict'],
            "epoch": 0,
            "best_acc": checkpoint1["best_acc"]
        }, os.path.join(f'co_train_model1.pth')
    )
    torch.save(
        {
            "state_dict": checkpoint2['state_dict'],
            "epoch": 0,
            "best_acc": checkpoint2["best_acc"]
        }, os.path.join(f'co_train_model2.pth')
    )
    torch.save(
        {
            "state_dict": checkpoint3['state_dict'],
            "epoch": 0,
            "best_acc": checkpoint3["best_acc"]
        }, os.path.join(f'co_train_model3.pth')
    )

    model_info(model1)
    model_info(model2)
    model_info(model3)
    train_dataset = get_dataset(config)(config, is_train=True, dataset_num=4)
    train_loader = DataLoader(
        dataset=train_dataset,        
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False, dataset_num=4)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    co_train_helper(model1, model2, model3, train_loader, config, device, converter)
    acc1 = function.validate(config, val_loader, val_dataset, converter, model1, criterion, device, epoch,
                        writer_dict, output_dict)
    print(f'Epoch[{epoch}], co_train model 1 test result is {acc1}')
    acc2 = function.validate(config, val_loader, val_dataset, converter, model2, criterion, device, epoch,
                            writer_dict, output_dict)
    print(f'Epoch[{epoch}], co_train model 2 test result is {acc2}')
    acc3 = function.validate(config, val_loader, val_dataset, converter, model3, criterion, device, epoch,
                            writer_dict, output_dict)
    print(f'Epoch[{epoch}], co_train model 3 test result is {acc3}')


def co_train_helper(model1, model2, model3, co_train_loader, config, device, converter):
    model1.eval()
    model2.eval()
    model3.eval()

    corr = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(co_train_loader):
            inp = inp.to(device)
            preds1 = model1(inp).cpu()
            preds2 = model2(inp).cpu()
            preds3 = model3(inp).cpu()

            preds_size1 = torch.IntTensor([preds1.size(0)] * inp.size(0))
            _, preds1 = preds1.max(2)
            preds1 = preds1.transpose(1, 0).contiguous().view(-1)
            sim_preds1_label = converter.decode_to_label(preds1.data, preds_size1.data, raw=False)
            sim_preds1 = converter.decode(preds1.data, preds_size1.data, raw=False)

            preds_size2 = torch.IntTensor([preds2.size(0)] * inp.size(0))
            _, preds2 = preds2.max(2)
            preds2 = preds2.transpose(1, 0).contiguous().view(-1)
            sim_preds2_label = converter.decode_to_label(preds2.data, preds_size2.data, raw=False)
            sim_preds2 = converter.decode(preds2.data, preds_size2.data, raw=False)

            preds_size3 = torch.IntTensor([preds3.size(0)] * inp.size(0))
            _, preds3 = preds3.max(2)
            preds3 = preds3.transpose(1, 0).contiguous().view(-1)
            sim_preds3_label = converter.decode_to_label(preds3.data, preds_size3.data, raw=False)
            sim_preds3 = converter.decode(preds3.data, preds_size3.data, raw=False)

            #for inx in idx:
                #print(inx)
                ##print( inx )
                #j=inx
            for j in range(0, len(idx)):
                j = int(j)
                inx = int(idx[j])
                #print( inx )
                sim_pred1 = sim_preds1[j]
                sim_pred2 = sim_preds2[j]
                sim_pred3 = sim_preds3[j]
                print(j)
                print(inx)
                print(sim_pred1)
                print(sim_pred2)
                print(sim_pred3)

                res, string = compare(sim_pred1, sim_pred2, sim_pred3, sim_preds1_label[j], sim_preds2_label[j], sim_preds3_label[j])
                if string:
                    # for test
                    # print(sim_preds1, sim_preds2, sim_preds3)
                    add_label_helper(config, string, res, inx)
                elif res == 4:
                    # print(4, sim_preds1, sim_preds2, sim_preds3)
                    continue
                else:
                    print('corr', (sim_pred1))
                    corr += 1


# 0: all equal; 1, preds1 is different; 4, all different
def compare(preds1, preds2, preds3, label1, label2, label3):
    if function.compare(preds1, preds2):
        if function.compare(preds3, preds2):
            return 0, None
        else:
            return 3, label2
    else:
        if function.compare(preds2, preds3):
            return 1, label3
        elif function.compare(preds1, preds3):
            return 2, label1
        else:
            return 4, None


def add_label_helper(config, string, num: int, idx: int):
    target_index = get_last_index(config, num)
    target_file_name = target_index + '.png'
    label = target_file_name + ' ' + string
    curr_file = config.DATASET4.ROOT + '/' + str(idx) + '.png'
    if num == 1:
        target_labeled_file = config.DATASET1.JSON_FILE['train']
        target_dir = config.DATASET1.ROOT
    elif num == 2:
        target_labeled_file = config.DATASET2.JSON_FILE['train']
        target_dir = config.DATASET2.ROOT
    else:
        target_labeled_file = config.DATASET3.JSON_FILE['train']
        target_dir = config.DATASET3.ROOT


    subprocess.run(["cp", curr_file, target_dir + '/' + target_file_name])
    with open(target_labeled_file, 'a') as f:
        f.write(label + '\n')
    return


def get_last_index(config, num):
    if num == 1:
        file = config.DATASET1.JSON_FILE['train']
    elif num == 2:
        file = config.DATASET2.JSON_FILE['train']
    else:
        file = config.DATASET3.JSON_FILE['train']

    with open(file, 'r') as f:
        for line in f:
            pass
    line = line.split()
    res = ""
    for i in line[0]:
        if i == '.':
            break
        res += i
    return str(int(res) + 1)


def main():
    # load config
    config = parse_arg()

    # construct face related neural networks

    # print
    model1 = crnn.get_crnn(config)
    # writing
    model2 = crnn.get_crnn(config)
    # mixed
    model3 = crnn.get_crnn(config)
    for i in range(0, 30):
        #train(model1, config, 1)
        #train(model2, config, 2)
        #train(model3, config, 3)
        co_train(config, model1, model2, model3, i)


if __name__ == '__main__':
    main()
