import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

path_to_craft = "../CRAFT-pytorch"
sys.path.append(path_to_craft)

from craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


net = CRAFT().to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

net.load_state_dict(copyStateDict(torch.load('../craft_mod/craft_mlt_25k.pth')))
net.eval()
from torch.autograd import Variable


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):

        nc = 3
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [128, 256, 256, 512, 1024, 1024, 1024]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.embedding_conv = nn.Linear(1024, nclass )

    def forward(self, input):
        input_to_craft = input

        craft_y, craft_feature = net(input_to_craft)
        craft_feature = craft_feature.permute( 0, 2, 3, 1 )
        craft_feature = F.interpolate( craft_feature, size=(50, 1024), mode='bilinear',align_corners=False )

        ## craft_feature = craft_feature.permute( 0, 3, 1, 2 )
        conv = self.cnn(input)
        conv = F.interpolate( conv, size=(64, 18), mode='bilinear', align_corners=False )

        conv = F.interpolate( conv, size=(1, 26), mode='bilinear', align_corners=False )
        conv = conv.squeeze( 2 )
        conv = conv.permute( 2, 0, 1 )
        print( "conv.shape=", conv.shape )
        conv_shape = conv.shape
        conv = self.embedding_conv(conv.reshape(conv_shape[0]*conv_shape[1],conv_shape[2]))
        print( "conv.shape=", conv.shape )
        conv = conv.reshape(conv_shape[0],conv_shape[1],conv_shape[2])
        print( "conv.shape=", conv.shape )


        output = F.log_softmax(conv, dim=2)
        print( "output.shape=", output.shape )

        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_crnn(config):
    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    model.apply(weights_init)
    return model
global print_crnn
print_crnn = True
