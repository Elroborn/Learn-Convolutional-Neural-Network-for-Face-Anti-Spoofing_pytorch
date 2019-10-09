#!/usr/bin/env python

import cv2, sys, os
from math import ceil
from glob import glob
import os
interpolation = cv2.INTER_CUBIC
borderMode = cv2.BORDER_REPLICATE

def crop_face(img, bbox, crop_sz, bbox_ext, extra_pad=0):
    shape = img.shape # [height, width, channels]
    x, y, w, h = bbox

    jitt_pad = int(ceil(float(extra_pad) * min(w, h) / crop_sz))

    pad = 0
    if x < w * bbox_ext + jitt_pad:
        pad = max(pad, w * bbox_ext + jitt_pad - x)
    if x + w * (1 + bbox_ext) + jitt_pad > shape[1]:
        pad = max(pad, x + w * (1 + bbox_ext) + jitt_pad - shape[1])
    if y < h * bbox_ext + jitt_pad:
        pad = max(pad, h * bbox_ext + jitt_pad - y)
    if y + h * (1 + bbox_ext) + jitt_pad > shape[0]:
        pad = max(pad, y + h * (1 + bbox_ext) + jitt_pad - shape[0])
    pad = int(pad)

    if pad > 0:
        pad = pad + 3
        replicate = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderMode)
    else:
        replicate = img
    cropped = replicate[int(pad + y - h * bbox_ext - jitt_pad) : int(pad + y + h * (1 + bbox_ext) + jitt_pad), 
                        int(pad + x - w * bbox_ext - jitt_pad) : int(pad + x + w * (1 + bbox_ext) + jitt_pad)]
    resized = cv2.resize(cropped, (crop_sz + 2*extra_pad, crop_sz + 2*extra_pad), interpolation=interpolation)
    return resized


def process_db_casia(db_dir, save_dir, scale, crop_sz):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_list = open(save_dir + "/file_list.txt", "w") # save info
    for file in open(db_dir+"/file_list.txt","r"):
        print("processing(scale %f): %s" % (scale, file))
        file_info = file.strip("\n").split(" ")
        file_name = file_info[0]
        bbox = map(int, file_info[1:5])
        label = file_info[5]

        cur_save_dir = os.path.join(save_dir, *file_name.split("/")[-4:-1])
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        frame = cv2.imread(file_name)
        bbox_ext = (scale - 1.0) / 2
        cropped = crop_face(frame, bbox, crop_sz, bbox_ext)

        save_fname = os.path.join(cur_save_dir,file_name.split("/")[-1])
        file_list.writelines("%s %s\n" % (save_fname,label))
        cv2.imwrite(save_fname, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])






if __name__ == "__main__":
    db_dir = '/home/userwyh/code/dataset/CASIA_frames'
    save_dir = '/home/userwyh/code/dataset/CASIA_scale'

    crop_sz = 128
    scales = [1.0, 1.4, 1.8, 2.2, 2.6]
    for scale in scales:
        cur_save_dir = save_dir + '/scale_' + str(scale)
        process_db_casia(db_dir, cur_save_dir, scale, crop_sz)