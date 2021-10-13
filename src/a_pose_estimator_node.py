#!/usr/bin/env python

import sys
import rospy
import message_filters
from detector import Detector
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError

from detect_mechanical_parts.msg import DetectionMsg, DetectionArrayMsg
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
from numpy.linalg import inv

#import pcl
import open3d as o3d

from pyquaternion import Quaternion #[w,x,y,z]
from utils import Utils
from main_registration import start_comparison
import pyransac3d as pyrsc



class EstimatorNodePC:

    def __init__(self):
        rospy.init_node('detector_mechanical_parts', anonymous=True)
        self.node_name = rospy.get_name()

        bbox_topic = rospy.get_param(self.node_name + '/bbox_topic')
        bboxes_topic = rospy.get_param(self.node_name + '/detections_image_topic')
        grasp_pose_topic = rospy.get_param(self.node_name + '/grasp_pose_topic')
        depth_source_topic = rospy.get_param(self.node_name + '/depth_source_topic')
        pointcloud_source_topic = rospy.get_param(self.node_name + '/pointcloud_source_topic')

        self.__bridge = CvBridge()
        self.pcd_data = o3d.geometry.PointCloud()
        self.pcd_single_object = o3d.geometry.PointCloud()
        #self.rif = o3d.geometry.PointCloud()

        self.pose_publisher = rospy.Publisher(grasp_pose_topic, Pose, queue_size=1)  

        pc_sub = message_filters.Subscriber(pointcloud_source_topic, PointCloud2, queue_size=10)
        bbox_sub = message_filters.Subscriber(bbox_topic, DetectionMsg, queue_size=10)
        #bboxes_sub = message_filters.Subscriber(bboxes_topic, DetectionArrayMsg)
        depth_sub = message_filters.Subscriber(depth_source_topic, Image, queue_size=10)

        ts = message_filters.ApproximateTimeSynchronizer([pc_sub, bbox_sub, depth_sub], 20, 0.1, allow_headerless=True) 
        ts.registerCallback(self.my_callback)

        self.K = np.array([[613.6060180664062, 0.0, 324.6341247558594], 
                           [0.0, 613.75537109375, 235.69447326660156], 
                           [0.0, 0.0, 1.0]])

        

    def convert_depth(self, data):
        try:
            depth_image = self.__bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
 	        print(e)

        #Convert the depth image to a Numpy array
        self.depth_array = np.array(depth_image, dtype=np.float32)

    def ros_to_open3d(self, data):
        '''
        Converts a ROS PointCloud2 message to a pcl PointXYZRGB
        Args: data (PointCloud2): ROS PointCloud2 message
        Returns: pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
        '''
        points_list = []

        for data in pc2.read_points(data, skip_nans=True):
            if(data[2] < 1.400):
                points_list.append([data[0], data[1], data[2]])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
        self.pcd_data.clear()
        self.pcd_data.points = o3d.utility.Vector3dVector(points_list)
    
    def deproject_pixel_to_point(self, ppixel):
        u = ppixel[1] #x
        v = ppixel[0] #y
        d1 = self.depth_array[v,u] 
        x_ = (u - self.K[0,2])/self.K[0,0]
        y_ = (v - self.K[1,2])/self.K[1,1]
        z = (float(np.array(d1))) / 1000
        x = x_ * z
        y = y_ * z
        return np.array([x, y, z])

    def find_center(self, detection):
        xr = detection.x + detection.w
        yb = detection.y + detection.h
        a = (detection.x + xr)/2
        b = (detection.y + yb)/2
        center = [a, b]
        return center


    def my_callback(self, sensor_pc, detection, sensor_depth):
        print("--------------------")
        print(sensor_pc.header.stamp)
        print(sensor_depth.header.stamp)
        #print(detection.header.stamp)
        self.convert_depth(sensor_depth) #converto il messaggio depth per avere i dati in self.depth_array
        self.ros_to_open3d(sensor_pc)    #converto il messaggio pointCloud per avere i dati in self.pcd_data
        #self.ros_to_pcl(sensor_pc)    #converto il messaggio pointCloud per avere i dati in self.pcd_data

        #print("DRAW ORIGINAL POINTCLOUD")
        #o3d.visualization.draw_geometries([self.pcd_data])

        # Prendo la bb dell'oggetto intero (sta nel topic target, va lasciato perche' con l'array non funziona la sincronizzazione)
        #print(detection)

        # Taglio la point cloud
        #trovo il centro della bb del target
        center = self.find_center(detection) #centro in pixel
        #print("Centro in pixel")
        #print(center)
        #trasformo il centro in pixel nelle coordinate x,y,z terna camera
        center_dep = self.deproject_pixel_to_point([int(center[1]), int(center[0])])
        #print("centro in m - terna camera")
        #print(center_dep)

        #calcolo gli estremi per estrarre il cubetto di pointcloud
        min_bound = [center_dep[0]-(0.06), center_dep[1]-(0.06), 0.05] 
        max_bound = [center_dep[0]+(0.05), center_dep[1]+(0.05), 0.78]

        points_list_filtered = []
        for i in range(len(self.pcd_data.points)):
            p = self.pcd_data.points[i]
            #print(p)
            #if (min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1]) and (minimo/1000 <= p[2] and p[2] <= (minimo/1000+0.02)):
            if (min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1]) and (min_bound[2] <= p[2] and p[2] <= max_bound[2]):
                #print(p)
                points_list_filtered.append(p)

        self.pcd_single_object.clear()
        self.pcd_single_object.points = o3d.utility.Vector3dVector(points_list_filtered)

        # Salvo la point cloud
        o3d.io.write_point_cloud('/home/monica/ros_catkin_ws_mine/src/detect_mechanical_parts/data/cloud_obj.pcd', self.pcd_single_object, write_ascii=True)

        #global_registration
        print("Start comparison")
        T_to_rob = start_comparison()
        quaternion = Utils.r2quat(T_to_rob[:3,:3])

        pose_msg = Pose()
        pose_msg.position.x = T_to_rob[0,3]
        pose_msg.position.y = T_to_rob[1,3]
        pose_msg.position.z = T_to_rob[2,3]
        pose_msg.orientation.w = quaternion[0]
        pose_msg.orientation.x = quaternion[1]
        pose_msg.orientation.y = quaternion[2]
        pose_msg.orientation.z = quaternion[3]

        self.pose_publisher.publish(pose_msg)
       


def main(args):

    estimator_node_pc = EstimatorNodePC()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)
