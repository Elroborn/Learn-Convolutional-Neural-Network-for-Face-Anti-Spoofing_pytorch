# Learn Convolutional Neural Network for Face Anti-Spoofing using pytorch  

## requirements  

* pytorch
* cv2
* tensorflow
* [mtcnn][1]

## Step 1  

run `generate_frames_and_bbox.py`,video is sampled as a frame,also generate a file_list containing the list of files_name and the bbox of the face   

**like this:**  

file_name  x y w h label  

/home/CASIA_frames/test_release/27/1/frame_42.jpg 233 122 170 215 1   

## Step 2  

run `crop_image.py`,generate face photos at different scales,like this

![image](https://tva2.sinaimg.cn/large/005Dd0fOly1g7s9se2dv3j30aq037wg2.jpg)   

To facilitate training, generate a file_list for each scale.  

## Step 3  

run `train.py`,a network will be trained and tested every n epochs


  [1]: https://github.com/ipazc/mtcnn