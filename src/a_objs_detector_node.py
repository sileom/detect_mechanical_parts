#!/usr/bin/env python

import sys
import os
import rospy
import message_filters
from detector import Detector
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from detect_mechanical_parts.msg import DetectionMsg, DetectionArrayMsg
from utils import Utils
import numpy as np

import cv2 as cv

class DetectorNode:

    def __init__(self):
        rospy.init_node('detector_mechanical_parts', anonymous=True)
        self.node_name = rospy.get_name()
        yolo_cfg = rospy.get_param(self.node_name + '/yolo_cfg')
        yolo_weights = rospy.get_param(self.node_name + '/yolo_weights')
        obj_names = rospy.get_param(self.node_name + '/obj_names')

        conf_threshold = float(rospy.get_param(self.node_name + '/conf_threshold'))
        nms_threshold = float(rospy.get_param(self.node_name + '/nms_threshold'))

        name_obj = rospy.get_param(self.node_name + '/name_obj')
        name_obj_pt = rospy.get_param(self.node_name + '/name_obj_pt')

        color_match_file = rospy.get_param(self.node_name + '/color_match_file')
        Utils.load_color_match(color_match_file)
        self.qrDecoder = cv.QRCodeDetector() # oggetto per la detection del QR, uno per tutta l'app

        

        self.detector = Detector(yolo_cfg=yolo_cfg, \
                                 yolo_weights=yolo_weights,\
                                 obj_names=obj_names,\
                                 conf_threshold=conf_threshold,\
                                 nms_threshold=nms_threshold, \
                                 obj=name_obj,\
                                 obj_pt=name_obj_pt)
        
        self.__bridge = CvBridge()
        '''
        detections_image_topic = rospy.get_param(f'/{self.node_name}/detections_image_topic')
        image_source_topic = rospy.get_param(f'/{self.node_name}/image_source_topic')
        '''
        detections_image_topic = rospy.get_param(self.node_name + '/detections_image_topic')
        image_source_topic = rospy.get_param(self.node_name + '/image_source_topic')
        bbox_topic = rospy.get_param(self.node_name + '/bbox_topic')
        detections_bbox_topic = rospy.get_param(self.node_name + '/detections_bbox_topic')
        result_img_topic = rospy.get_param(self.node_name + '/result_img_topic')
        
        #buffer size = 2**24 | 480*640*3
        rospy.Subscriber(image_source_topic, Image, self.detect_image, queue_size=1, buff_size=2**24)
        self.__publisher = rospy.Publisher(detections_image_topic, Image, queue_size=1)

        #Questo nodo deve pubblicare le bbox sul topic per farle leggere all'altro nodo che creeremo
        self.__publisherBB = rospy.Publisher(bbox_topic, DetectionMsg, queue_size=1)
        #Pubblichiamo anche l'array dell bboxes che dovra' leggere l'altro nodo
        self.__publisher_DETECTION_BBS = rospy.Publisher(detections_bbox_topic, DetectionArrayMsg, queue_size=1)

        #Questo nodo pubblica l'immagine con la sola bb che viene presa in considerazione al fine del calcolo della posa
        self.__publisherResultImg = rospy.Publisher(result_img_topic, Image, queue_size=1)
        self.image_url_publisher = rospy.Publisher("/url_string", String, queue_size=1)
        self.errors_publisher = rospy.Publisher("/errors", String, queue_size=1)

        #self.detect_image([]) #------------------------------


        
    def get_qr_dim(self, bb):
        vertices = bb[0]
        p0 = vertices[0]
        p0 = [round(i,0) for i in p0]
        p1 = vertices[1]
        p1 = [round(i,0) for i in p1]
        p2 = vertices[2]
        p2 = [round(i,0) for i in p2]
        w = np.sqrt( (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 ) 
        h = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
        return [w, w]
        

    def detect_image(self, data):
        try:
            image_i = self.__bridge.imgmsg_to_cv2(data, 'bgr8')
            #cv.imwrite("../ros_catkin_ws_mine/src/detect_mechanical_parts/data/image_source.png", image_i)
            #print("inizio test")#------------------------------
            #image_i = cv.imread("/home/monica/Scaricati/test3_Color.png")#------------------------------
            # PER USARE L'RGB --> decommentare la riga seguente e commentare le due successive
            image = image_i.copy()
            #image_gray = cv.cvtColor(image_i, cv.COLOR_BGR2GRAY) # Convert in grayScale
            #image = cv.cvtColor(image_gray, cv.COLOR_GRAY2RGB)   # Convert in "grayScale with 3channel"
        except CvBridgeError as cve:
            rospy.logerr(str(cve))
            return

        image_copy = image.copy()
        
        outputs = self.detector.detect_objects(image)
        [bbox_part, idxs, boxes, classIDs] = self.detector.process_image(outputs, image)

        qr_data,qr_bbox,qr_rectifiedImage = self.qrDecoder.detectAndDecode(image_copy)

        try:
            detection_message = self.__bridge.cv2_to_imgmsg(image, "bgr8")
            self.__publisher.publish(detection_message)
            cv.imwrite("../ros_catkin_ws_mine/src/detect_mechanical_parts/data/image_detections.png", image)
            absolute_path = os.path.abspath("../ros_catkin_ws_mine/src/detect_mechanical_parts/data/image_detections.png")
            self.image_url_publisher.publish(absolute_path)

            detections_msg = DetectionArrayMsg()
            if len(idxs)>0: # the system finds any objects
                for i in idxs.flatten():
                    det_temp = DetectionMsg()
                    det_temp.type = "bb"
                    det_temp.classe = Utils.get_detection_class(classIDs[i])# str(classIDs[i])
                    det_temp.color = Utils.get_color(classIDs[i])
                    det_temp.x = boxes[i][0]
                    det_temp.y = boxes[i][1]
                    det_temp.w = boxes[i][2]
                    det_temp.h = boxes[i][3]
                    det_temp.header.stamp = data.header.stamp
                    detections_msg.detections.append(det_temp)
            else:
                self.errors_publisher.publish("no_objects")
                print("NO OBJECT FOUND")

            if len(qr_data)>0:
                #print("QRCODE FOUND")
                #cv.line(image_copy, (int(qr_bbox[0][0][0]), int(qr_bbox[0][0][1])), (int(qr_bbox[0][0][0]), int(qr_bbox[0][0][1])), (255,0,0), 3)
                #cv.line(image_copy, (int(qr_bbox[0][1][0]), int(qr_bbox[0][1][1])), (int(qr_bbox[0][1][0]), int(qr_bbox[0][1][1])), (0,255,0), 3)
                #cv.line(image_copy, (int(qr_bbox[0][2][0]), int(qr_bbox[0][2][1])), (int(qr_bbox[0][2][0]), int(qr_bbox[0][2][1])), (0,0,255), 3)

                qr_point_sx = qr_bbox[0][0]
                qr_dim = self.get_qr_dim(qr_bbox)
                det_temp = DetectionMsg()
                det_temp.type = "qrCode"
                det_temp.classe = "qr"
                det_temp.color = "#000000"
                det_temp.x = int(qr_point_sx[0])
                det_temp.y = int(qr_point_sx[1])
                det_temp.w = int(qr_dim[0])
                det_temp.h = int(qr_dim[1])
                detections_msg.detections.append(det_temp)
            
            self.__publisher_DETECTION_BBS.publish(detections_msg)


            if len(bbox_part) != 0:
                #cv.imwrite("/home/labarea/catkin_ws/src/detect_mechanical_parts/src/gray.jpg", image)
                det_msg = DetectionMsg()
                det_msg.type = "bb"
                det_msg.classe = Utils.get_detection_class(bbox_part[0])
                det_msg.color = Utils.get_color(bbox_part[0])
                det_msg.x = bbox_part[1]
                det_msg.y = bbox_part[2]
                det_msg.w = bbox_part[3]
                det_msg.h = bbox_part[4]
                det_msg.header.stamp = data.header.stamp
                self.detector.create_img_result(image_copy, det_msg)
                #cv.imwrite("/home/monica/ros_catkin_ws_mine/src/detect_mechanical_parts/src/rgb_target.jpg", image_copy)
                result_img_message = self.__bridge.cv2_to_imgmsg(image_copy, "bgr8")
                #self.__publisherResultImg.publish(result_img_message)
                self.__publisherBB.publish(det_msg)
        except CvBridgeError as cve:
            rospy.logerr(str(cve))
            return

def main(args):

    detector_node = DetectorNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)
