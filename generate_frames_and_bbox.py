"""
Created on 19.10.8 16:35
@File:generate_frames_and_bbox.py
@author: coderwangson
"""
"#codeing=utf-8"
# TODO using pip install
from mtcnn.mtcnn import MTCNN
import cv2
import os
from glob import glob
detector = MTCNN()
true_img_start = ('1', '2', 'HR_1')
def generate_frames_and_bbox(db_dir,save_dir,skip_num):
    file_list = open(save_dir+"/file_list.txt","w")
    for file in glob("%s/*/*/*.avi"%db_dir):
        print("Processing video %s"%file)
        dir_name = os.path.join(save_dir, *file.replace(".avi", "").split("/")[-3:])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        frame_num = 0
        count = 0
        vidcap = cv2.VideoCapture(file)
        success, frame = vidcap.read()
        while success:

            # 只保存有人脸的帧
            detect_res = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(detect_res)>0 and count%skip_num==0:

                file_name = os.path.join(dir_name,"frame_%d.jpg" % frame_num)
                # bbox = (x,y,w,h)
                bbox = (detect_res[0]['box'][0],detect_res[0]['box'][1],detect_res[0]['box'][2],detect_res[0]['box'][3])

                label_txt = file.replace(".avi", "").split("/")[-1]

                label = 1 if label_txt in true_img_start else 0
                # file_name x y w h label
                file_list.writelines("%s %d %d %d %d %d\n"%(file_name,bbox[0],bbox[1],bbox[2],bbox[3],label))

                cv2.imwrite(file_name,frame)
                frame_num+=1
            count+=1
            success, frame = vidcap.read()  # 获取下一帧

        vidcap.release()

    file_list.close()
def read():
    file = open("/home/userwyh/code/dataset/CASIA_frames/file_list.txt")  # 打开文件
    for line in file:
        print(line.strip("\n").split(" "))


if __name__ == '__main__':
    db_dir = "/home/userwyh/code/dataset/CASIA"
    save_dir = "/home/userwyh/code/dataset/CASIA_frames"
    generate_frames_and_bbox(db_dir,save_dir,3)


    # read()