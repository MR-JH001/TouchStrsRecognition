import sys
import os
import time
import math
import argparse
import json

from torchvision.transforms import Resize, Pad

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

from CRNN_recognizer import recognizer

classifer_box = Classifer_box()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
from dir_utils import debug_write
from transform_utils import convert_to_binary_inv, adapt_size, convert_to_binary, gen_ToPILImage
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


def compress(input):
    result = ''

    last_char = ''
    most_char = ''
    now_char_count = 0
    most_char_count = 0
    for x in input:
        if x == 's' or x == 'b':
            if last_char != '' and last_char != '=' and last_char != '1':
                result += last_char
                last_char = ''
                most_char = ''
                now_char_count = 0
                most_char_count = 0
            continue
        elif x == '=' or x == '1':
            if last_char != x:
                result += x
                last_char = x
        elif last_char == x:
            now_char_count += 1
            most_char = x if now_char_count > most_char_count else most_char
            most_char_count = now_char_count if now_char_count > most_char_count else most_char_count
        else:
            last_char = x
            now_char_count = 1
    return result


print(compress("===1111"))
print(compress("==="))
print(compress("1111"))

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='craft_mod/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')

# 使用refine
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')

parser.add_argument('--refiner_model', default='craft_mod/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

args = parser.parse_args()

""" For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)


result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def detect_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net, res_path):
    t0 = time.time()

    origin_image_1_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    origin_image_3_color = np.array(image)
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    cv2.imwrite("core_link.jpg", score_text * 255)
    cv2.imwrite("score_link.jpg", score_link * 255)

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    # 获取CRAFT生成的框
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    '处理裂开的box，相邻的放在同一组'
    # 广度优先合并相邻的框
    # 距离矩阵构建
    all_rect_cx_cy = np.zeros((len(boxes), 2))
    for i in range(len(boxes)):
        box = boxes[i]
        left = min(box[0][0], box[1][0], box[2][0], box[3][0])
        right = max(box[0][0], box[1][0], box[2][0], box[3][0])
        top = min(box[0][1], box[1][1], box[2][1], box[3][1])
        bottom = max(box[0][1], box[1][1], box[2][1], box[3][1])
        top = int(top)
        bottom = int(bottom)
        left = int(left)
        right = int(right)

        all_rect_cx_cy[i][0] = ((left + right) / 2) / 4
        # 减少x轴的影响
        # 还需调整
        all_rect_cx_cy[i][1] = ((top + bottom) / 2)
    mat_distance = []
    for i in range(len(all_rect_cx_cy)):
        mat_distance.append(np.sqrt(np.sum((all_rect_cx_cy - all_rect_cx_cy[i]) ** 2, axis=-1)))

    print("generate distance mat;len:", len(mat_distance))

    segment_group = []
    ind_group = -1
    search_queue = deque()
    cnt_processed = 0
    processed = set()
    # 广度优先
    while cnt_processed < len(all_rect_cx_cy):  # 只要搜索队列中有数据就一直遍历下去
        if (len(search_queue) == 0):
            for i in range(len(all_rect_cx_cy)):
                if (i not in processed):
                    search_queue.append(i)
                    segment_group.append([])
                    ind_group += 1
                    break
        current_node = search_queue.popleft()  # 从队列前边获取节点，即先进先出，这是BFS的核心
        if current_node not in processed:  # 当前节点是否被访问过
            cnt_processed += 1
            processed.add(current_node)
            inds = np.argsort(mat_distance[current_node])
            segment_group[ind_group].append(boxes[current_node])
            cnt_company = 0
            distance_threshold = 20  # max(all_rect[current_node][2],all_rect[current_node][3])
            # print(distance_threshold)
            for index in inds:  # 遍历相邻节点，判断相邻节点是否已经在搜索队列
                if mat_distance[current_node][index] > distance_threshold:
                    break
                cnt_company += 1
                if cnt_company > 200:
                    print("error")
                    exit()
                if index not in search_queue:  # 如果相邻节点不在搜索队列则进行添加
                    search_queue.append(index)

    '合并在同一组的框'
    merge_boxes = []
    for segment in segment_group:
        left_s = []
        right_s = []
        top_s = []
        bottom_s = []
        for box in segment:
            left = min(box[0][0], box[1][0], box[2][0], box[3][0])
            right = max(box[0][0], box[1][0], box[2][0], box[3][0])
            top = min(box[0][1], box[1][1], box[2][1], box[3][1])
            bottom = max(box[0][1], box[1][1], box[2][1], box[3][1])
            top = math.floor(top)
            bottom = math.floor(bottom)
            left = math.floor(left)
            right = math.floor(right)

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

    json_record = []
    for rect in merge_boxes:
        threshold_hw = min(rect[3] - rect[1], rect[2] - rect[0]) * 0.2
        crop = origin_image_1_channel[rect[1]:rect[3], rect[0]:rect[2]]
        # debug_write(crop,"exp");

        # adaptiveThreshold
        binary_img = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 31, 10)
        debug_write(binary_img, "all")

        # ret, binary_img = cv2.threshold(crop, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # debug_write(binary_img,"dilate")
        # kernel = np.ones((1, 2), np.uint8)

        # binary_img_dilate = cv2.erode(binary_img, kernel, iterations=1)

        # debug_write(binary_img_dilate,"dilate")
        # print(binary_img.max(),binary_img.min())
        _, contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        group = []
        for i in range(len(contours)):
            rect_char = cv2.boundingRect(contours[i])
            group.append(rect_char)
        group.sort(key=lambda rect: rect[0])

        if (len(group) >= 1):
            last_x_start = group[0][0]
            last_x_end = group[0][0] + group[0][2]
            last = group[0]
        i = 1

        '合并=/等符号'
        while i < len(group) and i >= 1:
            now = group[i]
            cx = now[0] + now[2] / 2
            cy = now[1] + now[3] / 2
            last_cy = last[1] + last[3] / 2
            y_near = abs(last_cy - cy) < (last_x_end - last_x_start) * 0.6
            if (last_x_start < cx and cx < last_x_end and y_near):
                group.pop(i)
                i -= 1
                x1 = min(now[0], group[i][0])
                y1 = min(now[1], group[i][1])
                x2 = max(now[0] + now[2], group[i][0] + group[i][2])
                y2 = max(now[1] + now[3], group[i][1] + group[i][3])
                group[i] = (x1, y1, x2 - x1, y2 - y1)
            else:
                last_x_start = group[i][0]
                last_x_end = group[i][0] + group[i][2]
                last = group[i]
            i += 1
        # if(len(group)<4 or len(group)>16):
        #     continue
        '检测每个框及其结果'

        json_record_perline = []

        rect_set = []
        res_set = []

        # def detect_rect(rect_char, binary_img):
        #
        #     crop_char = binary_img[
        #                 rect_char[1]:
        #                 rect_char[1] + rect_char[3],
        #                 rect_char[0]:
        #                 rect_char[0] + rect_char[2]]
        #
        #     debug_crop_char = crop_char
        #     if crop_char.shape[0]*6 < crop_char.shape[1]:
        #         return '-'
        #     if crop_char.shape[0] < 2 or crop_char.shape[1] < 2:
        #         return ''
        #     debug_write(crop_char, "detect_rect")
        #     crnn_text_result = recognizer(crop_char)
        #
        #     crop_char = torch.tensor(crop_char, dtype=torch.int)
        #
        #     crop_char = adapt_size(crop_char)
        #     crop_char = crop_char.float().to(device)
        #     res = classifer_box.eval(crop_char.unsqueeze(0)).squeeze().int().item()
        #
        #     print(config.CLASS[res], crnn_text_result)
        #
        #     return config.CLASS_toString[res]

        def detect_rect(rect_char, binary_img, before_str):

            crop_char = binary_img[
                        rect_char[1]:
                        rect_char[1] + rect_char[3],
                        rect_char[0]:
                        rect_char[0] + rect_char[2]]

            # 减号
            # print(crop_char.shape)
            # if crop_char.shape[0] * 3 < crop_char.shape[1] and crop_char.mean() > 128:
            #     return '-'
            # if crop_char.shape[1] * 3 < crop_char.shape[0] and crop_char.mean() > 128:
            #     return '1'
            # 区域过小
            if crop_char.shape[0] < 2 and crop_char.shape[1] < 2:
                return ''

            # debug_write(crop_char, "detect_rect")

            # if crop_char.shape[1] < crop_char.shape[0] // 2:
            #     fx = 4
            # else:
            #     fx = fy

            # crnn
            crnn_text_result = recognizer(crop_char)
            # debug_write(crop_char,crnn_text_result.replace('/','d'))

            # dense
            # crop_char = torch.tensor(crop_char, dtype=torch.int)
            # crop_char = adapt_size(crop_char)
            # crop_char = crop_char.float().to(device)
            # res = classifer_box.eval(crop_char.unsqueeze(0)).squeeze().int().item()

            # print(crnn_text_result,compress(crnn_text_result))


            # print(crnn_text_result)
            return compress(crnn_text_result)

        res_str = ''
        for i in range(len(group)):
            rect_char = group[i]
            if max(rect_char[2], rect_char[3]) < threshold_hw:
                continue
            res = detect_rect(rect_char, binary_img, before_str=res_str)
            res_set.append(res)
            rect_set.append(rect_char)
            res_str += res
        print(res_str)
        # for i in range(len(res_set)):
        #     res = res_set[i]
        #     res_str += config.CLASS_toString[res]
        #
        #     json_record_perline.append({'rect_char': rect_set[i], 'char': config.CLASS_toString[res]})
        #
        #     # print('left',res)
        #     '等号右边颜色浅 针对右边进行二值化后重新检测'
        #     if (config.CLASS_is_eq(res)):
        #         rect_char = rect_set[i]
        #
        #         crop = origin_image_1_channel[rect[1]:rect[3], rect[0]:rect[2]][:, rect_char[0] + rect_char[2]:]
        #
        #         # 记录相对位置
        #         relative = (rect_char[0] + rect_char[2], 0, 0, 0)
        #
        #         if (crop.shape[0] * crop.shape[1] < 4):
        #             break
        #         # 自适应算法
        #         # crop = convert_to_binary_inv(crop)
        #         crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                               cv2.THRESH_BINARY_INV, 31, 10)
        #         # debug_write(crop,'')
        #
        #         _, contours_right, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        #         group_right = []
        #         for i in range(len(contours_right)):
        #             rect_char_right = cv2.boundingRect(contours_right[i])
        #             group_right.append(rect_char_right)
        #         group_right.sort(key=lambda rect: rect[0])
        #         for rect_char in group_right:
        #             if (max(rect_char[2], rect_char[3]) < crop.shape[0] * 0.3):
        #                 continue
        #             res_right = detect_rect(rect_char, crop)
        #             res_str += config.CLASS_toString[res_right]
        #             json_record_perline.append({'rect_char': (
        #                 relative[0] + rect_char[0],
        #                 relative[1] + rect_char[1],
        #                 rect_char[2],
        #                 rect_char[3]
        #             ), 'char': config.CLASS_toString[res_right]})
        #
        #         break

        eq = res_str.split('=')
        if (len(eq) >= 2):
            res_str = res_str.replace("/", "d")

            json_record.append({'rect_expression': (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]),
                                'expression': json_record_perline})
            with open("resjson/" + res_str + ".json", 'w') as file_object:
                file_object.write(json.dumps(
                    {'rect_expression': (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]),
                     'expression': json_record_perline}))

            if str_to_num(eq[0]) == str_to_num(eq[-1]):
                # cv2.rectangle(origin_image_3_color, (rect[0], rect[1]), (rect[2] , rect[3]),	(46,255,87), 2)
                cv2.line(origin_image_3_color, (rect[0], rect[3]), (rect[2], rect[3]), (46, 255, 87), 2)
                cv2.imwrite('./res/' + res_str + '.png', origin_image_1_channel[rect[1]:rect[3], rect[0]:rect[2]])
            elif eq[-1] == "":
                cv2.rectangle(origin_image_3_color, (rect[0], rect[1]), (rect[2], rect[3]), (255, 46, 87), 2)
                cv2.imwrite('./res/O' + res_str + '.png', origin_image_1_channel[rect[1]:rect[3], rect[0]:rect[2]])
            else:
                cv2.rectangle(origin_image_3_color, (rect[0], rect[1]), (rect[2], rect[3]), (46, 87, 255), 2)
                cv2.imwrite('./res/X' + res_str + '.png', origin_image_1_channel[rect[1]:rect[3], rect[0]:rect[2]])

    print(res_path)
    cv2.imwrite(res_path, origin_image_3_color)

    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    cv2.imwrite("xxxx.png", ret_score_text)

    # for line in json_record:
    #     print(line)
    data2 = json.dumps(json_record)
    return data2


if __name__ == '__main__':
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    print('craft threshold',
          args.link_threshold,
          args.text_threshold,
          args.low_text
          )
    args.cuda = torch.cuda.is_available()
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

import flask, os, sys, time
from flask import request

interface_path = os.path.dirname(__file__)
sys.path.insert(0, interface_path)  # 将当前文件的父目录加入临时系统变量

print(__name__)
server = flask.Flask(__name__, static_folder='static')


@server.route('/', methods=['get'])
def index():
    return '<form action="/upload" method="post" enctype="multipart/form-data"><input type="file" id="img" name="img"><button type="submit">上传</button></form>'


@server.route('/upload', methods=['post'])
def upload():
    fname = request.files['img']  # 获取上传的文件

    if fname:
        img = np.array(Image.open(fname)).astype(np.uint8)
        img = imgproc.loadImage4channel(img)
        # print(img.shape)

        res_path = r'static/' + time.strftime('%Y%m%d%H%M%S') + fname.filename[-4:]
        res_path_net_save = r'./CRAFT-pytorch/' + res_path
        json = detect_net(net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly,
                          refine_net, res_path_net_save)

        # t = time.strftime('%Y%m%d%H%M%S')
        # new_fname = r'static/' + t + fname.filename
        # fname.save(new_fname)  #保存文件到指定路径
        return '<img style="max-width: 100%; max-height: 800px; "' + ' src=%s>' % res_path + "<p>" + json + "</p>"
    else:
        return '{"msg": "请上传文件！"}'


print('----------路由和视图函数的对应关系----------')
print(server.url_map)  # 打印路由和视图函数的对应关系
server.run(port=8000)
