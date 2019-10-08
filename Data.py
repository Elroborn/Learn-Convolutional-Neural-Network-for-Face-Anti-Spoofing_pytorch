"""
Created on 19.10.8 20:13
@File:Data.py
@author: coderwangson
"""
"#codeing=utf-8"
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
class Data(data.Dataset):
    def __init__(self,db_dir,is_train):
        self.is_train = is_train
        self.file_list,self.label = self.get_file_list(db_dir)
        if self.is_train:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.transforms(Image.open(self.file_list[index]).convert('RGB'))

        label = self.label[index]
        return img,label

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self,db_dir):
        file_list = []
        label_list = []
        for file in open(db_dir + "/file_list.txt", "r"):
            file_info = file.strip("\n").split(" ")
            file_name = file_info[0]
            label = file_info[1]
            if self.is_train and file_name.split("/")[-4].startswith("train"):
                file_list.append(file_name)
                label_list.append(int(label))
            if not self.is_train and file_name.split("/")[-4].startswith("test"):
                file_list.append(file_name)
                label_list.append(int(label))
        return file_list,label_list

if __name__ == '__main__':
    data = Data("/home/userwyh/code/dataset/CASIA_scale/scale_1.0/",True)



