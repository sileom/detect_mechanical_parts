#!/usr/bin/env python

import sys
import rospy
import message_filters
from detector import Detector
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from detect_mechanical_parts.msg import DetectionMsg, DetectionArray

from utils import Utils

import cv2 as cv
import numpy as np

class QRDetectionNode:

    def __init__(self):
        rospy.init_node('detector_mechanical_parts', anonymous=True)
        self.node_name = rospy.get_name()

        
        self.__bridge = CvBridge()


        color_match_file = rospy.get_param(self.node_name + '/color_match_file')
        Utils.load_color_match(color_match_file)

        self.qrDecoder = cv.QRCodeDetector() # oggetto per la detection del QR, uno per tutta l'app

        detections_image_topic = rospy.get_param(self.node_name + '/detections_image_topic') # immagine risultato con bordi de QR
        image_source_topic = rospy.get_param(self.node_name + '/image_source_topic') # topic per immagine in input
        bbox_topic = rospy.get_param(self.node_name + '/bbox_topic') # topic per mandare le coordinate
        #detections_bbox_topic = rospy.get_param(self.node_name + '/detections_bbox_topic')
        #result_img_topic = rospy.get_param(self.node_name + '/result_img_topic') # topic per immagine con target DA ELIMINARE
        
        #buffer size = 2**24 | 480*640*3
        rospy.Subscriber(image_source_topic, Image, self.detect_qr, queue_size=1, buff_size=2**24)
        self.__publisher = rospy.Publisher(detections_image_topic, Image, queue_size=1)

        #Questo nodo deve pubblicare le bbox sul topic per farle leggere all'altro nodo che creeremo
        self.__publisherBB = rospy.Publisher(bbox_topic, DetectionMsg, queue_size=1)
        ## Qui non pubblichiamo l'array dell bboxes che dovra' leggere l'altro nodo perche' non ci interessano
        #self.__publisher_DETECTION_BBS = rospy.Publisher(detections_bbox_topic, DetectionArrayMsg, queue_size=1)

        #Questo nodo pubblica l'immagine con la sola bb che viene presa in considerazione al fine del calcolo della posa
        #self.__publisherResultImg = rospy.Publisher(result_img_topic, Image, queue_size=1)
    
    def get_point_sx(self, bb):
        vertices = bb[0]
        minimum = 1300.0
        idx = 5
        for i in range(4):
            if vertices[i][0] < minimum:
                minimum = vertices[i][0]
                idx = i
        return vertices[idx]

    def get_qr_dim(self, bb):
        vertices = bb[0]
        print("prima: {}".format(vertices))
        ind = np.argsort( vertices[:,1] ); 
        vertices = vertices[ind]
        print("dopo: {}".format(vertices))
    
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
        





    def detect_qr(self, data):
        try:
            image = self.__bridge.imgmsg_to_cv2(data, 'bgr8') # opencv MAT
        except CvBridgeError as cve:
            rospy.logerr(str(cve))
            return
        
        image_copy = image.copy()

        qr_data,qr_bbox,qr_rectifiedImage = self.qrDecoder.detectAndDecode(image_copy)
        if len(qr_data)>0:
            # draw QR
            print("BB: {}".format(qr_bbox))
            #print("N: {}".format(qr_bbox[0][1][0]))
            #print("N: {}".format(qr_bbox[0][1][1]))
            #cv.line(image_copy, (int(qr_bbox[0][0][0]), int(qr_bbox[0][0][1])), (int(qr_bbox[0][1][0]), int(qr_bbox[0][1][1])), (255,0,0), 3)
            #cv.line(image_copy, (int(qr_bbox[0][1][0]), int(qr_bbox[0][1][1])), (int(qr_bbox[0][2][0]), int(qr_bbox[0][2][1])), (255,0,0), 3)
            #cv.line(image_copy, (int(qr_bbox[0][2][0]), int(qr_bbox[0][2][1])), (int(qr_bbox[0][3][0]), int(qr_bbox[0][3][1])), (255,0,0), 3)
            #cv.line(image_copy, (int(qr_bbox[0][0][0]), int(qr_bbox[0][0][1])), (int(qr_bbox[0][3][0]), int(qr_bbox[0][3][1])), (255,0,0), 3)
            cv.line(image_copy, (int(qr_bbox[0][0][0]), int(qr_bbox[0][0][1])), (int(qr_bbox[0][0][0]), int(qr_bbox[0][0][1])), (255,0,0), 3)
            cv.line(image_copy, (int(qr_bbox[0][1][0]), int(qr_bbox[0][1][1])), (int(qr_bbox[0][1][0]), int(qr_bbox[0][1][1])), (0,255,0), 3)
            cv.line(image_copy, (int(qr_bbox[0][2][0]), int(qr_bbox[0][2][1])), (int(qr_bbox[0][2][0]), int(qr_bbox[0][2][1])), (0,0,255), 3)

            # CALCOLO DEL PUNTO IN ALTO A SINISTRA E LARGHEZZA E ALTEZZA 
            # NBBB !!!!!!!!!!!!!!!!!!!!!!! SE IL QR CODE E' MESSO DRITTO, IL PUNTO IN ALTO A SINISTRA E' IL PRIMO RESTITUITO DALLA FUNZIONE
            qr_point_sx = qr_bbox[0][0]
            qr_dim = self.get_qr_dim(qr_bbox)

            det_msg = DetectionMsg()
            det_msg.type = "qrCode"
            det_msg.classe = "qr"
            det_msg.color = "#000000"
            det_msg.x = 0
            det_msg.y = 0
            det_msg.w = 0
            det_msg.h = 0


            self.__publisherBB.publish(det_msg)
            
            detection_message = self.__bridge.cv2_to_imgmsg(image_copy, "bgr8")
            self.__publisher.publish(detection_message)
        else:
            print("QR Code not detected")
            

def main(args):

    qr_detection_node = QRDetectionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)



