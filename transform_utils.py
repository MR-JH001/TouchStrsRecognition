import cv2
from torchvision import transforms
from torchvision.transforms import Resize,Pad
import matplotlib.pyplot as plt
import numpy as np
import PIL

gen_ToPILImage=transforms.ToPILImage()
gen_ToTensor=transforms.ToTensor()
def convert_to_binary(test_image):
    ret,binary_img = cv2.threshold(test_image,175,255,cv2.THRESH_OTSU)
    #ret,binary_img = cv2.threshold(test_image,175,255,cv2.THRESH_BINARY_INV)
    return binary_img
def convert_to_binary_inv(test_image):
    ret,binary_img = cv2.threshold(test_image,175,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret,binary_img = cv2.threshold(test_image,175,255,cv2.THRESH_BINARY_INV)
    return binary_img
char_hw = 28
def adapt_size(cell):
    origin_h,origin_w=cell.shape
    if origin_h>origin_w:

        cell = gen_ToPILImage(cell)
        h2=char_hw
        w2=int((char_hw/origin_h)*origin_w)
        cell = Resize((h2,w2),interpolation=PIL.Image.NEAREST)(cell)
        pad=char_hw-w2
        if(pad==0):
            pass
        elif(pad % 2 ==1):
            cell = Pad((pad//2+1,0,pad//2,0),0)(cell)
        else:
            cell = Pad((pad//2,0,pad//2,0),0)(cell)
    else:
        cell = gen_ToPILImage(cell)
        w2=char_hw
        h2=int((char_hw/origin_w)*origin_h)
        cell = Resize((h2,w2),interpolation=PIL.Image.NEAREST)(cell)
        pad=char_hw-h2
        if(pad==0):
            pass
        elif(pad % 2 ==1):
            cell = Pad((0,pad//2+1,0,pad//2),0)(cell)
        else:
            cell = Pad((0,pad//2,0,pad//2),0)(cell)
    return gen_ToTensor(np.array(cell))