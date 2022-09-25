import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse

checkpoint_path = "best.pth"


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/OWN_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test_2.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_path,
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def recognition(config, img, model, converter, device):
    print(img.shape)
    h, w = img.shape



    img = 255 - img
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)


    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY,31,1)

    plt.matshow(img)
    plt.show()


    print(config.MODEL.IMAGE_SIZE.H / h, config.MODEL.IMAGE_SIZE.W / w)


    # Corroded image

    h, w = img.shape
    img = np.reshape(img, (h, w, 1))
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / img.max()

    img = img.transpose([2, 0, 1])

    plt.matshow(img[0])
    plt.show()

    model.eval()

    preds = model(torch.Tensor([img]).to(device))
    print(preds.shape)
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    def compress(input):
        last_char = -1
        res = ''
        for char in input:
            if char == 'b':
                continue
            if char == 's':
                last_char = -1
                continue
            elif last_char != char:
                res += char
                last_char = char
        return res

    sim_pred = converter.decode(preds.data, preds_size.data, raw=True)

    print('results: {0}'.format(sim_pred), '=>{0}'.format(compress(sim_pred)))

    # print('results: {0}'.format(len(sim_pred)))
    # sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #
    # print('results: {0}'.format(sim_pred))


if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    recognition(config, img, model, converter, device)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))
