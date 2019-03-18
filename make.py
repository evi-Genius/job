# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2 
import math
import random
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the 
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the imageb
#     angle=angle%90
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(0,0,0))

def padding(img,height,width):
    (h,w)=img.shape[:2]
    a=height-h
    b=width-w
    if a>0 and b>0:
        an=np.zeros([a,w])
        img=np.r_[img,an]
        bn=np.zeros([height,b])
        img=np.c_[img,bn]
    elif a>0:
        an=np.zeros([a,w])
        img=np.r_[img,an]
    elif b>0:
        bn=np.zeros([h,b])
        img=np.c_[img,bn]
    return img


def make_Mask_And_Double(img1, img2):
    '''
     生成合成图片(h,w)以及mask（h,w,2）
     :param img1:
     :param img2:
     :return:
     '''
    angle1 = random.randint(0, 360)
    angle2 = random.randint(0, 360)
    # 如果角度不对，重新random
    #     if not (abs(angle2%180-angle1%180)>70 and abs(angle1%180-angle2%180)<110):
    #         continue
    img1 = 255 - img1
    img2 = 255 - img2
    img1 = rotate_bound(img1, angle1)
    img2 = rotate_bound(img2, angle2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    width = max(img1.shape[1], img2.shape[1])
    height = max(img1.shape[0], img2.shape[0])
    img1 = padding(img1, height, width)
    img2 = padding(img2, height, width)

    sum1 = img1[img1 != 0]
    sum2 = img2[img2 != 0]
    sumAnd = len(sum1) + len(sum2)
    arraryall = img1 + img2
    sumUnion = len(arraryall[arraryall != 0])
    sumDiffer = sumAnd - sumUnion

    imgall = img1 + img2
    sumUnion = len(imgall[imgall != 0])
    mask = np.zeros([height, width, 2], dtype=np.uint8)
    double = np.zeros([height, width], dtype=np.uint8)
    # 如果没有重叠，退出
    if sumDiffer == 0:
        return False, mask, double

    for i in range(0, height):
        for j in range(0, width):
            # 重叠部分
            if img1[i, j] != 0 and img2[i, j] != 0:
                mask[i, j, 0] = 255
                mask[i, j, 1] = 255
                sum = (int(img1[i, j] * 0.7) + int(img2[i, j] * 0.7))
                double[i, j] = 255 if sum > 255 else sum
                continue
            elif img1[i, j] != 0:
                mask[i, j, 0] = 255
            elif img2[i, j] != 0:
                mask[i, j, 1] = 255
            else:
                mask[i, j, 0], mask[i, j, 1] = 0, 0
            double[i, j] = img1[i, j] + img2[i, j]
    double = cv2.GaussianBlur(double, (3, 3), 0)
    double=255-double
    return True, mask, double

if __name__ == "__main__":
    path = './img/'
    img_list = os.listdir(path)
    for img_name in img_list:
        a = random.randint(0, len(img_list) - 1)
        b = random.randint(0, len(img_list) - 1)
        img1 = cv2.imread(os.path.join(path, img_list[a]))
        img2 = cv2.imread(os.path.join(path, img_list[b]))
        bo, mask, imgDouble = make_Mask_And_Double(img1, img2)
        if not bo:
            continue
        cv2.imshow('doublePic', imgDouble)
        cv2.imshow('mask1', mask[:, :, 0])
        cv2.imshow('mask2', mask[:, :, 1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
