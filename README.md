# objects_picker

ROS package to read a stream of images, detect mechanical parts of cars using YOLOv4 and view the detection results.
___
## Launch Requirements
* ROS Noetic 1.15.9 (*Recommended for Ubuntu 20.04*)
* Python 3.4+
* OpenCV 4.4.0+
* NumPy 1.17.4+
* Download [yolo-obj.weights](https://drive.google.com/file/d/1sYhDkknxBJHRL8S8kOegkU3ktBOuNa8p/view?usp=sharing) and place it in the [yolo_model](yolo_model) folder
* For gray images: download [yolov4_objs_gray_last.weights](https://drive.google.com/file/d/1ENkEmb-9xs9ygOG7WrOuLw0wvZBNWXMl/view?usp=sharing) and place it in the [yolo_model](yolo_model) folder

## Installation
In order to install detect_mechanical_parts, clone this repository in your catkin workspace and compile the package using ROS.

```shell
$ cd catkin_workspace/src
$ git clone https://github.com/leonard0guerra/detect_mechanical_parts.git
$ cd ..
$ catkin build detect_mechanical_parts
```

## Detection :robot:
1. To launch detect_mechanical_parts run the command:
    ```shell
    $ roslaunch detect_mechanical_parts detect_mechanical_parts.launch
    ```
2. You can download [this bag](https://drive.google.com/file/d/1CngH1nSqTF9j4RZHsccH1meC1ZSXYaKp/view?usp=sharing) and run the command:
    ```shell
    $ rosbag play 20210113_161956.bag --topics /device_0/sensor_1/Color_0/image/data
    ```
You can change the parameters in the [launch file](launch/detect_mechanical_parts.launch) (e.g. topics, confidence threshold value...) and launch it.
___
## References
* [Computer Vision and Machine Perception ](http://web.unibas.it/bloisi/corsi/visione-e-percezione.html) - University of Basilicata (Italy)
* [AlexeyAB](https://github.com/AlexeyAB/darknet) darknet

__
## Note 27 January 2021
The package works also with ROS kinetic and python 2.7, but I had to substitute the default OpenCV with version 4.4.0.40 following the instructions described here: https://programmersought.com/article/11351781035/ 
I also changed the CMakeLists.txt file, adding lines 7-to-11 to indicates the new OpenCV path.
To install OpenCV 4.4.0.40, I followed this instructions: https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

## Activate the virtual environment 
$ source ros-venv/bin/activate

## To execute the detection with realtime images from Intel Realsense D435 camera
Open a terminal and launch the camera node for unorganized point cloud
```shell
$ cd catkin_ws/
$ source devel/setup.bash
$ roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud
```

To launch the camera node for organized point cloud
```shell
$ cd catkin_ws/
$ source devel/setup.bash
$ roslaunch realsense2_camera rs_rgbd.launch 
```

In another terminal launch 
```shell
$ cd catkin_ws/
$ source devel/setup.bash
$ roslaunch detect_mechanical_parts controller_icosaf.launch
```

In the last terminal launch 
```shell
$ cd catkin_ws/
$ source devel/setup.bash
$ roslaunch detect_mechanical_parts detect_mechanical_parts_icosaf.launch conf_threshold:=0.75
```


