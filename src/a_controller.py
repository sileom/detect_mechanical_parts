#!/usr/bin/env python3

import sys
import rospy
import os
import message_filters
from detector import Detector
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose

from detect_mechanical_parts.msg import SubscribeToStreamMsg, AckMsg, ImageReadyMsg, BoxesMsg, DisplacementMsg, DetectionArrayMsg
from detect_mechanical_parts.msg import RichiestaCoordinateMsg, RichiestaCoordinateStatusMsg, CoordinateMsg, PrelievoComponenteMsg, ComponenteErrorMsg, DepositoComponenteMsg, ComponenteStatusMsg

import cv2 as cv
import numpy as np
import time

class Controller:

    def __init__(self):
        rospy.init_node('detector_mechanical_parts', anonymous=True)
        self.node_name = rospy.get_name()

        self.__bridge = CvBridge()

        self.currentComponent = ""
        self.currentTaskID = 0
        self.currentAction = ""


        # ------------------ HOLOLENS
        subscribe_to_stream_topic = rospy.get_param(self.node_name + '/subscribe_to_stream') # mi sottoscrivo per vedere quando mi arriva la richiesta
        unsubscribe_to_stream_topic = rospy.get_param(self.node_name + '/unsubscribe_to_stream') # mi sottoscrivo per vedere quando mi arriva la richiesta
        acks_ds_sv_topic = rospy.get_param(self.node_name + '/acks_ds_sv') # mi sottoscrivo per ricevere gli ack da ds
        acks_sv_ds_topic = rospy.get_param(self.node_name + '/acks_sv_ds') # pubblico gli ack per inviarli al ds
        image_url_topic = rospy.get_param(self.node_name + '/image_url') # pubblico l'url dell'immagine
        bounding_boxes_topic = rospy.get_param(self.node_name + '/bounding_boxes') # pubblico le bounding box degli oggetti e del qr
        detections_bb_topic = rospy.get_param(self.node_name + '/detections_bbox') # mi sottoscrivo per vedere quando arrivano le bbox
        displacements_topic = rospy.get_param(self.node_name + '/displacements') # mi sottoscrivo per ricevere i displacement

        rospy.Subscriber(subscribe_to_stream_topic, SubscribeToStreamMsg, self.subscribe_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(unsubscribe_to_stream_topic, SubscribeToStreamMsg, self.unsubscribe_callback, queue_size=1, buff_size=2**24)

        self.acks_publisher = rospy.Publisher(acks_sv_ds_topic, AckMsg, queue_size=1)
        rospy.Subscriber(acks_ds_sv_topic, AckMsg, self.acks_from_ds_callback, queue_size=1, buff_size=2**24)

        rospy.Subscriber("/url_string", String, self.publish_url, queue_size=1, buff_size=2**24)
        self.image_url_publisher = rospy.Publisher(image_url_topic, ImageReadyMsg, queue_size=1)

        rospy.Subscriber(detections_bb_topic, DetectionArrayMsg, self.publish_bboxes, queue_size=1, buff_size=2**24)
        self.bounding_boxes_publisher = rospy.Publisher(bounding_boxes_topic, BoxesMsg, queue_size=1)

        rospy.Subscriber(displacements_topic, DisplacementMsg, self.displacements_callback, queue_size=1, buff_size=2**24)

        # ------------------ HMI
        richiesta_coordinate_topic = rospy.get_param(self.node_name + '/richiesta_coordinate') # mi sottoscrivo per vedere quando mi arriva la richiesta
        richiesta_coordinate_ack_topic = rospy.get_param(self.node_name + '/richiesta_coordinate_ack') # mi sottoscrivo per ricevere gli ack da ds
        coordinate_topic = rospy.get_param(self.node_name + '/coordinate') # pubblico gli ack per inviarli al ds
        prelievo_componente_topic = rospy.get_param(self.node_name + '/prelievo_componente') # mi sottoscrivo per vedere quando mi arriva la richiesta
        deposito_componente_topic = rospy.get_param(self.node_name + '/deposito_componente') # mi sottoscrivo per ricevere gli ack da ds
        componente_error_topic = rospy.get_param(self.node_name + '/componente_error') # pubblico gli ack per inviarli al ds
        prelievo_ack_topic = rospy.get_param(self.node_name + '/prelievo_ack') # pubblico gli ack per inviarli al ds
        deposito_ack_topic = rospy.get_param(self.node_name + '/deposito_ack') # pubblico gli ack per inviarli al ds

        rospy.Subscriber(richiesta_coordinate_topic, RichiestaCoordinateMsg, self.receive_coordinate, queue_size=1, buff_size=2**24)
        self.richiesta_coordinate_ack_publisher = rospy.Publisher(richiesta_coordinate_ack_topic, RichiestaCoordinateStatusMsg, queue_size=1)
        self.coordinate_publisher = rospy.Publisher(coordinate_topic, CoordinateMsg, queue_size=1)

        rospy.Subscriber(prelievo_componente_topic, PrelievoComponenteMsg, self.prelievo, queue_size=1, buff_size=2**24)
        self.pre_dep_publisher = rospy.Publisher(componente_error_topic, ComponenteErrorMsg, queue_size=1)

        rospy.Subscriber(deposito_componente_topic, DepositoComponenteMsg, self.deposito, queue_size=1, buff_size=2**24)

        self.prelievo_ack_publisher = rospy.Publisher(prelievo_ack_topic, ComponenteStatusMsg, queue_size=1)
        self.deposito_ack_publisher = rospy.Publisher(deposito_ack_topic, ComponenteStatusMsg, queue_size=1)

        # ------------------ Robot UR
        to_robot_topic = rospy.get_param(self.node_name + '/topic_to_robot')
        from_robot_topic = rospy.get_param(self.node_name + '/topic_from_robot')
        rospy.Subscriber(from_robot_topic, String, self.callback_robot_response)     # mi sottoscrivo alle risposte del robot
        self.from_robot_publisher = rospy.Publisher(to_robot_topic, String, queue_size=10) # pubblico i comandi per il robot 
        self.gripper_to_use = 1

        # ------------------ Object detector
        object_topic = rospy.get_param(self.node_name + '/topic_obj')
        self.obj_to_detect_publisher = rospy.Publisher(object_topic, String, queue_size=10) # pubblico gli oggetti da riconoscere
        
        grasp_pose_topic = rospy.get_param(self.node_name + '/grasp_pose_topic')
        rospy.Subscriber(grasp_pose_topic, Pose, self.receiveObjectPose, queue_size=1, buff_size=2**24)
    
    def receiveObjectPose(self, data):
        msg = CoordinateMsg()
        msg.task_id = self.currentTaskID
        msg.action = "richiesta coordinate"
        msg.component = self.currentComponent
        ## TODO fare parsing di coordinate
        msg.coordinate = "X:0.1,Y:0.2,Z:0.8"
        self.coordinate_publisher.publish(msg)


    def matchObjAndCodes(self, code):
        if(self.currentComponent == '10132161F'): # COPERCHIO OLIO
            return "oil_separator_crankcase_plastic"
        elif(self.currentComponent == '41572014F'):  # FLEX PLATE
            return "flexplate"
        elif(self.currentComponent == '21632313G'):  # DISTANZIALE
            return "spacer"
        return "rail"

    def callback_robot_response(self, data):
        print("Robot send message.")
        print(data.data)
        print(type(self.currentTaskID))
        if(data.data == "ok" or data.data == "OK"):
            if(self.currentTaskID == 507 or self.currentAction == "adjust"): # il robot e' in posizione avvio il detector
                self.obj_to_detect_publisher.publish(self.matchObjAndCodes(self.currentComponent))
                time.sleep(2)
                print("Richiesta servizio")
                # avvio il servizio per prendere i dati dalla camera
                os.system("rosservice call /phoxi_camera/get_frame \"in_: -1\"")
                print("Servizio attivo")
            if(self.currentTaskID == 508): # prelievo componente
                self.send_status_componente("prelievo componente", "OK", "")
            if(self.currentTaskID == 509): # deposito componente
                self.send_status_componente("deposito componente", "OK", "")
            
        elif(data.data == "fail" or data.data == "FAIL"):
            if(self.currentTaskID == 508): # prelievo componente
                self.send_status_componente("prelievo componente", "NOK", "R_ROB")
            if(self.currentTaskID == 509): # deposito componente
                self.send_status_componente("deposito componente", "NOK", "R_ROB")
    
    def send_ack_hmi_richiesta(self, data, valueOfStatus):
        status_msg = RichiestaCoordinateStatusMsg()
        status_msg.task_id = data.task_id
        status_msg.action = data.action
        status_msg.status = valueOfStatus
        self.richiesta_coordinate_ack_publisher.publish(status_msg)

    def send_ack_hmi_pre_dep(self, data, valueOfStatus):
        status_msg = ComponenteStatusMsg()
        status_msg.task_id = data.task_id
        status_msg.action = data.action
        status_msg.status = valueOfStatus
        if data.action == "prelievoComponente":
            self.prelievo_ack_publisher.publish(status_msg)
        elif data.action == "depositoComponente":
            self.deposito_ack_publisher.publish(status_msg)

    def receive_coordinate(self, data):
        print("------ RICEVUTA RICHIESTA COORDINATE PER COMPONENTE:")
        print(data.component)
        print()
        self.currentComponent = data.component
        self.currentTaskID = data.task_id
        self.send_ack_hmi_richiesta(data, "ACK") 
        self.newRequest = True
        # POSIZIONI PER INQUADRARE - NB: il numero iniziale rappresenta il tipo di end-effector per la presa
        if(self.currentComponent == '35032008F'): # RAIL '10132161F'): # COPERCHIO OLIO 
            self.gripper_to_use = 1
            self.from_robot_publisher.publish("(1, -18.66, -425.92, -183.85, 2.849, -1.218, -0.311)") # mm rotvec
            #self.from_robot_publisher.publish("1, -1.68, -2.48, -2.42, 0.28, 1.45, 3.83)")
        elif(self.currentComponent == '41572014F'):  # FLEX PLATE
            self.gripper_to_use = 2
            #self.from_robot_publisher.publish("2, -0.065, -0.389, -0.332, 2.559, -1.253, -0.205)") # DA MODIFICARE
            self.from_robot_publisher.publish("2, -0.065, -0.389, -0.332, 2.559, -1.253, -0.205)") 
        elif(self.currentComponent == '21632313G'):  # DISTANZIALE
            self.gripper_to_use = 3
            #self.from_robot_publisher.publish("3, -0.065, -0.389, -0.332, 2.559, -1.253, -0.205)") # DA MODIFICARE
            self.from_robot_publisher.publish("3, -0.065, -0.389, -0.332, 2.559, -1.253, -0.205)")


    def send_status_componente(self, valueOfAction, valueOfStatus, valueOfError):
        msg = ComponenteErrorMsg()
        msg.task_id = self.currentTaskID
        msg.action = valueOfAction
        msg.status = valueOfStatus
        msg.error = valueOfError
        self.pre_dep_publisher.publish(msg)

    

    def prelievo(self, data):
        print("------ RICEVUTA RICHIESTA PRELIEVO COMPONENTE:")
        print(data.component)
        self.currentTaskID = data.task_id
        print("ALLE COORDINATE: ")
        print(data.coordinate)
        print()
        self.send_ack_hmi_pre_dep(data, "OK")
        # parsing coordinate X:0.12,Y:0.03,Z:0.56
        componenti_pos = data.coordinate.split(',')
        x = float(componenti_pos[0].split(':')[1])
        y = float(componenti_pos[1].split(':')[1])
        z = float(componenti_pos[2].split(':')[1])
        #a = float(componenti_pos[3].split(':')[1])
        ## TODO conversione di orientamento
        message_for_robot = d = '({}, {}, {}, {}, {}, {}, {})'.format(self.gripper_to_use, x,y,z, 2.559, -1.253, -0.205) # TODO AGGIUNGERE ORIENTAMENTO DEFAULT
        # mando il robot
        self.from_robot_publisher.publish(message_for_robot)

    
    def deposito(self, data):
        print("------ RICEVUTA RICHIESTA DEPOSITO PER COMPONENTE:")
        print(data.component)
        self.currentTaskID = data.task_id
        print()
        self.send_ack_hmi_pre_dep(data, "OK")
        # MANDARE IL ROBOT A DEPOSITARE
        # posizioni di deposito su AVG
        if(self.currentComponent == '10132161F'): # COPERCHIO OLIO
            self.from_robot_publisher.publish("xc yc zc thetaXc thetaYc thetaZc")
        elif(self.currentComponent == '41572014F'):  # FLEX PLATE
            self.from_robot_publisher.publish("xf yf zf thetaXf thetaYf thetaZf")
        elif(self.currentComponent == '21632313G'):  # DISTANZIALE
            self.from_robot_publisher.publish("xd yd zd thetaXd thetaYd thetaZd")
        

    def acks_from_ds_callback(self, data):
        print("------ ACK RECEIVED ------")
        print("------ START MESSAGE ------")
        print("Action: {}".format(data.action))
        print("Ack: {}".format(data.ACK))
        print("Timestamp: {}".format(data.timestamp))
        print("------ END MESSAGE ------")
        print()

    def send_ack(self, data):
        ack_msg = AckMsg()
        ack_msg.ACK = 1
        ack_msg.action = data.action
        ack_msg.timestamp = data.timestamp
        self.acks_publisher.publish(ack_msg)


    def subscribe_callback(self, data):
        action = data.action
        print(action)
        print("------ RICEVUTA RICHIESTA DI SOTTOSCRIZIONE ALLO STREAM ------")
        print()
        self.send_ack(data)

    def unsubscribe_callback(self, data):
        action = data.action
        print(action)
        print("------ RICEVUTA RICHIESTA DI CANCELLAZIONE SOTTOSCRIZIONE ALLO STREAM ------")
        print()
        self.send_ack(data)
        


    def displacements_callback(self, data): #test
        print("------ DISPLACEMENT MESSAGE RECEIVED ------")
        print("------ START MESSAGE ------")
        print("Action: {}".format(data.action))
        print("Target: {}".format(data.target))
        print("offset X: {}".format(data.offsetX))
        print("offset Y: {}".format(data.offsetY))
        print("Angle X: {}".format(data.angle))
        print("Timestamp: {}".format(data.timestamp))
        print("------ END MESSAGE ------")
        self.currentAction = data.action
        self.send_ack(data)
        # messaggio per il robot 
        message_for_robot = '({}, {}, {}, {})'.format(self.gripper_to_use, data.offsetX, data.offsetY, data.angle) 
        # mando il robot
        self.from_robot_publisher.publish(message_for_robot)
        


    def errors_handler(self, data):
        msg = data.data
        if (msg == "no_objects"): #ERR 01
            self.errors_to_ds_publisher.publish("E-VIS-01")
        elif (msg == "only_upsidedown_objects"): #ERR 03
            self.errors_to_ds_publisher.publish("E-VIS-03") # TO DO in objs_detector_node
        elif (msg == "no_camera_data"): #ERR 04
            self.errors_to_ds_publisher.publish("E-CAM-01") # TO DO in objs_detector_node


    def publish_url(self, data):
        url = data.data
        image_url_msg = ImageReadyMsg()
        image_url_msg.action = 'imageReady'
        image_url_msg.url = url
        image_url_msg.timestamp = int(rospy.get_rostime().to_sec())
        self.image_url_publisher.publish(image_url_msg)


    def publish_bboxes(self, data):
        bboxes = data.detections
        boxes_msg = BoxesMsg()
        boxes_msg.action = 'boundingBoxes'
        boxes_msg.timestamp = int(rospy.get_rostime().to_sec())
        for i in range(len(bboxes)):
            boxes_msg.boundingBoxes.append(bboxes[i])
        self.bounding_boxes_publisher.publish(boxes_msg)

    
def main(args):

    controller = Controller()
    #while(True):
    #    image_url_msg = ImageReadyMsg()
    #    image_url_msg.action = 'imageReady'
    #    image_url_msg.url = "/home/user/catkin_ws/src/detect_mechanical_parts/data/image_predictions.png"
    #    image_url_msg.timestamp = int(rospy.get_rostime().to_sec())
    #    controller.image_url_publisher.publish(image_url_msg)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)



