#!/usr/bin/env python3

import sys
import rospy
import message_filters
from detector import Detector
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from detect_mechanical_parts.msg import DetectionMsg, DetectionArrayMsg

import cv2 as cv


class DetectorNode:

    def __init__(self):
        rospy.init_node('detector_mechanical_parts', anonymous=True)
        self.node_name = rospy.get_name()
        '''
        yolo_cfg = rospy.get_param(f'/{self.node_name}/yolo_cfg')
        yolo_weights = rospy.get_param(f'/{self.node_name}/yolo_weights')
        obj_names = rospy.get_param(f'/{self.node_name}/obj_names')

        conf_threshold = float(rospy.get_param(f'/{self.node_name}/conf_threshold'))
        nms_threshold = float(rospy.get_param(f'/{self.node_name}/nms_threshold'))
        '''
        yolo_cfg = rospy.get_param(self.node_name + '/yolo_cfg')
        yolo_weights = rospy.get_param(self.node_name + '/yolo_weights')
        obj_names = rospy.get_param(self.node_name + '/obj_names')

        conf_threshold = float(rospy.get_param(self.node_name + '/conf_threshold'))
        nms_threshold = float(rospy.get_param(self.node_name + '/nms_threshold'))

        name_obj = rospy.get_param(self.node_name + '/name_obj')
        name_obj_pt = rospy.get_param(self.node_name + '/name_obj_pt')
        

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
        self.__publisherBB = rospy.Publisher(bbox_topic, Detection, queue_size=1)
        #Pubblichiamo anche l'array dell bboxes che dovra' leggere l'altro nodo
        self.__publisher_DETECTION_BBS = rospy.Publisher(detections_bbox_topic, DetectionArray, queue_size=1)

        #Questo nodo pubblica l'immagine con la sola bb che viene presa in considerazione al fine del calcolo della posa
        self.__publisherResultImg = rospy.Publisher(result_img_topic, Image, queue_size=1)


        

        

    def detect_image(self, data):
        try:
            image_i = self.__bridge.imgmsg_to_cv2(data, 'rgb8') #'bgr8')
            cv.imwrite("/home/user/catkin_ws/src/detect_mechanical_parts/src/origin.jpg", image_i)
            # PER USARE L'RGB --> decommentare la riga seguente e commentare le due successive
            #image = image_i.copy()
            #image_gray = cv.cvtColor(image_i, cv.COLOR_BGR2GRAY) # Convert in grayScale
            # GRAY PHOTONEO
            image_gray = image_i.copy()
            image = cv.cvtColor(image_gray, cv.COLOR_GRAY2RGB)   # Convert in "grayScale with 3channel"
        except CvBridgeError as cve:
            rospy.logerr(str(cve))
            return

        image_copy = image.copy()
        
        outputs = self.detector.detect_objects(image)
        [bbox_part, idxs, boxes, classIDs] = self.detector.process_image(outputs, image)

        try:
            detection_message = self.__bridge.cv2_to_imgmsg(image, "rgb8")
            self.__publisher.publish(detection_message)
            detections_msg = DetectionArray()
            if len(idxs)>0:
                for i in idxs.flatten():
                    det_temp = Detection()
                    det_temp.classe = classIDs[i]
                    det_temp.x = boxes[i][0]
                    det_temp.y = boxes[i][1]
                    det_temp.w = boxes[i][2]
                    det_temp.h = boxes[i][3]
                    detections_msg.detections.append(det_temp)
            
            self.__publisher_DETECTION_BBS.publish(detections_msg)


            if len(bbox_part) != 0:
                #cv.imwrite("/home/labarea/catkin_ws/src/detect_mechanical_parts/src/gray.jpg", image)
                det_msg = Detection()
                det_msg.classe = bbox_part[0]
                det_msg.x = bbox_part[1]
                det_msg.y = bbox_part[2]
                det_msg.w = bbox_part[3]
                det_msg.h = bbox_part[4]
                self.detector.create_img_result(image_copy, det_msg)
                cv.imwrite("/home/user/catkin_ws/src/detect_mechanical_parts/src/rgb.jpg", image)
                result_img_message = self.__bridge.cv2_to_imgmsg(image_copy, "bgr8")
                self.__publisherResultImg.publish(result_img_message)
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
