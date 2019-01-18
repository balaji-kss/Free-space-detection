# **Free space detection**

### Objective
In the case of the autonomous driving, given a front camera view, the car needs to know where
to move on the road. For that we need to detect free spaces on the road. This assignment was
done as a part of our college Self Driving Car project.


#### Dependencies & my environment


* Python3.5
* OS: Ubuntu 14.04
* Tensorflow-gpu
* CUDA 9.0
* Numpy, Scipy

#### How to run the code

(1) Load pre-trained VGG

Download pre-trained ​ VGG​ [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip). Extract the files and place them inside the ​ data
folder.

(2) Download model

The trained model can be downloaded from [here](https://www.dropbox.com/s/uv1xkd6y4nzxq2n/model.zip?dl=0). Extract the files and place them inside the ​ model​ folder.

(3) Run the code:
```sh
python main.py
```
This code will take the input images from ​ input_images​ folder which is inside
the data folder and detects free space/ drivable area in these images and saves them in
the ​ outputs​ folder. The output videos are in the ​ output_videos​ folder.
