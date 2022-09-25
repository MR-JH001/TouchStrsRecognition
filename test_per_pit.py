# -*- coding: utf-8 -*-
import sys
import os
import time
import math
import argparse


from torchvision.transforms import Resize,Pad

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
from collections import deque

import random

from load_cnn import Classifer_box
classifer_box=Classifer_box()
from dir_utils import debug_write
from transform_utils import convert_to_binary_inv,adapt_size,convert_to_binary,gen_ToPILImage
import config



from calculate import str_to_num

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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")




parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='craft_mod/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    'Deal with cracked boxes, adjacent to the same group'
    # Breadth-first merge adjacent boxes
    # build distance matrix
    all_rect_cx_cy=np.zeros((len(boxes),2))
    for i in range(len(boxes)):
        box=boxes[i]
        left=min(box[0][0],box[1][0],box[2][0],box[3][0])
        right=max(box[0][0],box[1][0],box[2][0],box[3][0])
        top=min(box[0][1],box[1][1],box[2][1],box[3][1])
        bottom=max(box[0][1],box[1][1],box[2][1],box[3][1])
        top=int(top)
        bottom=int(bottom)
        left=int(left)
        right=int(right)

        all_rect_cx_cy[i][0]=((left+right)/2)/4
        # reduce the influence of the x-axis
        # still need to adjust
        all_rect_cx_cy[i][1]=((top+bottom)/2)
    mat_distance=[]
    for i in range(len(all_rect_cx_cy)):
        mat_distance.append(np.sqrt(np.sum((all_rect_cx_cy-all_rect_cx_cy[i])**2,axis=-1)))
    print("generate distance mat;len:",len(mat_distance))

    segment_group=[]
    ind_group=-1
    search_queue=deque()
    cnt_processed=0
    processed=set()
    # breadth first
    while cnt_processed<len(all_rect_cx_cy):     # Keep traversing as long as there is data in the search queue
        if(len(search_queue)==0):
            for i in range(len(all_rect_cx_cy)):
                if(i not in processed):
                    search_queue.append(i)
                    segment_group.append([])
                    ind_group+=1
                    break
        current_node = search_queue.popleft()  # Get nodes from the front of the queue, i.e. first in first out, which is the core of BFS
        if current_node not in processed:   # Whether the current node has been visited
            cnt_processed+=1
            processed.add(current_node)
            inds=np.argsort(mat_distance[current_node])
            segment_group[ind_group].append(boxes[current_node])
            cnt_company=0
            distance_threshold=20#max(all_rect[current_node][2],all_rect[current_node][3])
            # print(distance_threshold)
            for index in inds:  # Traverse adjacent nodes to determine whether adjacent nodes are already searching the queue
                if mat_distance[current_node][index]>distance_threshold:
                    break
                cnt_company+=1
                if cnt_company>200:
                    print("error")
                    exit()
                if index not in search_queue:        # If the adjacent node is not in the search queue, add it
                    search_queue.append(index)

    'Merge boxes in the same group'
    merge_boxes=[]
    for segment in segment_group:
        left_s=[]
        right_s=[]
        top_s=[]
        bottom_s=[]
        for box in segment:
            left=min(box[0][0],box[1][0],box[2][0],box[3][0])
            right=max(box[0][0],box[1][0],box[2][0],box[3][0])
            top=min(box[0][1],box[1][1],box[2][1],box[3][1])
            bottom=max(box[0][1],box[1][1],box[2][1],box[3][1])
            top=math.floor(top)
            bottom=math.floor(bottom)
            left=math.floor(left)
            right=math.floor(right)
            
            left_s.append(left)
            right_s.append(right)
            top_s.append(top)
            bottom_s.append(bottom)
        merge_boxes.append([
            min(left_s),
            min(top_s),
            max(right_s),
            max(bottom_s)
        ])
    
    for rect in merge_boxes:
        threshold_hw=min(rect[3]-rect[1],rect[2]-rect[0])*0.2
        crop=i_image[rect[1]:rect[3],rect[0]:rect[2]]
        ret,binary_img = cv2.threshold(crop,175,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        _, contours, _  = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        group=[]
        for i in range(len(contours)):
            rect_char=cv2.boundingRect(contours[i])
            group.append(rect_char)
        group.sort( key=lambda rect:rect[0])
        

        last_x_start=group[0][0]
        last_x_end=group[0][0]+group[0][2]
        last=group[0]
        i=1

        'merge=/etc symbols'
        while i < len(group) and i>=1:
            now=group[i]
            cx=now[0]+now[2]/2
            cy=now[1]+now[3]/2
            last_cy=last[1]+last[3]/2
            y_near=abs(last_cy-cy)<(last_x_end-last_x_start)*0.6
            if(last_x_start<cx and cx<last_x_end and y_near):
                group.pop(i)
                i-=1
                x1=min(now[0],group[i][0])
                y1=min(now[1],group[i][1])
                x2=max(now[0]+now[2],group[i][0]+group[i][2])
                y2=max(now[1]+now[3],group[i][1]+group[i][3])
                group[i]=(x1,y1,x2-x1,y2-y1)
            else:
                last_x_start=group[i][0]
                last_x_end=group[i][0]+group[i][2]
                last=group[i]
            i+=1
        if(len(group)<4 or len(group)>16):
            continue
        'Detect each box and its result'
        rect_set=[]
        res_set=[]
        def detect_rect(rect_char,binary_img):
            crop_char=binary_img[
                rect_char[1]:
                rect_char[1]+rect_char[3],
                rect_char[0]:
                rect_char[0]+rect_char[2]]
            crop_char=torch.tensor(crop_char,dtype=torch.int)
            crop_char=adapt_size(crop_char)
            crop_char=crop_char.float().cuda()
            res=classifer_box.eval(crop_char.unsqueeze(0)).squeeze().int().item()
            debug_write(crop_char[0].cpu().int().numpy().astype(np.uint8)*255,config.CLASS_toString[res])
            return res
        for i in range(len(group)):
            rect_char=group[i]
            if max(rect_char[2],rect_char[3])<threshold_hw:
                continue
            res=detect_rect(rect_char,binary_img)
            res_set.append(res)
            rect_set.append(rect_char)

        res_str=''
        for i in range(len(res_set)):
            res=res_set[i]
            res_str+=config.CLASS_toString[res]
            # print('left',res)
            'The color on the right side of the equal sign is light, and the right side is binarized and re-detected'
            if(config.CLASS_is_eq(res)):
                rect_char=rect_set[i]
                
                crop=i_image[rect[1]:rect[3],rect[0]:rect[2]][:,rect_char[0]+rect_char[2]:]
                if(crop.shape[0]*crop.shape[1]<4):
                    break
                crop=convert_to_binary_inv(crop)
                debug_write(crop,'')
                _, contours_right, _  = cv2.findContours(crop, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                group_right=[]
                for i in range(len(contours_right)):
                    rect_char_right=cv2.boundingRect(contours_right[i])
                    group_right.append(rect_char_right)
                group_right.sort( key=lambda rect:rect[0])
                for rect_char in group_right:
                    if(max(rect_char[2],rect_char[3])<crop.shape[0]*0.3):
                        continue
                    res_right=detect_rect(rect_char,crop)
                    res_str+=config.CLASS_toString[res_right]
                break
        eq=res_str.split('=')
        if(len(eq)==2):
            global i_image_3_color
            res_str=res_str.replace("/","d")
            print(res_str)
            if str_to_num(eq[0])==str_to_num(eq[1]):
                cv2.rectangle(i_image_3_color, (rect[0], rect[1]), (rect[2] , rect[3]),	(46,255,87), 2)
                cv2.imwrite('./res/'+res_str+'.png', i_image[rect[1]:rect[3],rect[0]:rect[2]])
            elif eq[1]=="":
                cv2.rectangle(i_image_3_color, (rect[0], rect[1]), (rect[2] , rect[3]),	(46,87,255), 2)
                cv2.imwrite('./res/'+res_str+'.png', i_image[rect[1]:rect[3],rect[0]:rect[2]])
            else :
                cv2.rectangle(i_image_3_color, (rect[0], rect[1]), (rect[2] , rect[3]),	(255,46,87), 2)
                cv2.imwrite('./res/x_'+res_str+'.png', i_image[rect[1]:rect[3],rect[0]:rect[2]])
            # print(str_to_num(eq[0])
            # print(str_to_num(eq[1])
            

        
        # cv2.imwrite('./res/'+res_str+'.png', binary_img)

    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]



    cv2.imshow('',i_image_3_color)
    cv2.waitKey()
    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text









global i_path
i_path='test/c1.jpg'
global i_image
i_image=cv2.imread(i_path, 0)
global i_image_3_color
i_image_3_color=cv2.imread(i_path)
if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    print('craft threshold',
        args.link_threshold,
        args.text_threshold,
        args.low_text
        )
    args.link_threshold
    args.text_threshold
    args.low_text
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    
    files=[i_path]
    for k, image_path in enumerate(files):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(files), image_path), end='\n')
        image = imgproc.loadImage(image_path)


        print(image)
        print(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
# python .\CRAFT-pytorch\test_per_pit.py --trained_model=./craft_mod/craft_mlt_25k.pth 
# python .\CRAFT-pytorch\test_per_pit.py --trained_model=./craft_mod/craft_mlt_25k.pth