from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import imgproc

import matplotlib.pyplot as plt


class _OWN(data.Dataset):
    def __init__(self, config, is_train=True, dataset_num: int = 1):
        if dataset_num == 1:
            DATASET = config.DATASET1
        if dataset_num == 2:
            DATASET = config.DATASET2
        if dataset_num == 3:
            DATASET = config.DATASET3
        if dataset_num == 4:
            DATASET = config.DATASET4

        self.root = DATASET.ROOT
        self.dataset_name = config.DATASET.DATASET
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W


        self.mean = np.array(DATASET.MEAN, dtype=np.float32)
        self.std = np.array(DATASET.STD, dtype=np.float32)

        txt_file = DATASET.JSON_FILE['train'] if is_train else DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list( self.labels[idx].keys() )[0]
        img = cv2.imread( os.path.join( self.root, img_name ) )
        # img = img.re
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #print(idx)
        #print(idx, img.shape)
        img_h, img_w, c = img.shape
        if img_h != 32 or img_w != 200:
            print( "ERROR:", img_name )
        img = img.astype( np.float32 )

        # img = np.expand_dims(img, axis=0)
        # img, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, 1280,
        #                                                                       interpolation=cv2.INTER_LINEAR,
        #                                                                       mag_ratio=1.5)

        img = imgproc.normalizeMeanVariance( img )

        img = img.transpose( 2, 0, 1 )
        # print(img.shape)
        # img = img.transpose([2, 0, 1])

        # plt.matshow(img[0])
        # plt.show()

        # cv2.imshow("",img)
        # cv2.waitKey()

        # ret,img = cv2.threshold(img,175,255,cv2.THRESH_OTSU)

        # cv2.imshow("",img)
        # cv2.waitKey()

        # print(np.mean(img),np.min(img),np.max(img))

        # img = (img/255. - self.mean) / self.std

        # img = (img- np.mean(img)) / np.std(img)

        # print(np.mean(img),np.min(img),np.max(img))

        # cv2.imshow("",img)
        # cv2.waitKey()

        # print(img.shape)

        # print(img.shape)
        # plt.matshow(img[0])
        # plt.show()

        # print(img.shape)
        # cv2.imshow("",img[0])
        # cv2.waitKey()
        return img, idx
