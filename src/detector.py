#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from utils import Utils

class Detector:

    #def __init__(self, yolo_cfg:str, yolo_weights:str, obj_names:str, conf_threshold:float = 0.3, nms_threshold:float = 0.4):
    def __init__(self, yolo_cfg, yolo_weights, obj_names, conf_threshold = 0.3, nms_threshold = 0.4, obj = "", obj_pt = ""):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.yolo_cfg = yolo_cfg
        self.yolo_weights = yolo_weights
        self.obj_names = obj_names

        self.obj = obj
        self.obj_pt = obj_pt

        self.__init_colors_for_classes()
        self.__init_network()

    
    def __init_colors_for_classes(self):
        self.labels = Utils.load_classes(self.obj_names)

        #np.random.seed(777)
        #self.bbox_colors = np.random.uniform(low=0, high=255, size=(len(self.labels), 3))

        self.bbox_colors = []
        for i in range(len(self.labels)):
            rgb = Utils.hex_to_rgb(Utils.get_color(i))
            self.bbox_colors.append(rgb)
        
    
    def __init_network(self):
        self.net = cv.dnn.readNetFromDarknet(Utils.absolute_path(self.yolo_cfg), darknetModel=Utils.absolute_path(self.yolo_weights))
        
        if Utils.is_cuda_cv():
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        layer_names = self.net.getLayerNames()
        #self.__output_layer = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.__output_layer = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image):
        #print(image.shape)
        cv.imwrite("../catkin_ws/src/detect_mechanical_parts/data/image_detections_pho_prima.png", image)
        #image = cv.resize(image, (416, 416), interpolation = cv.INTER_AREA)
        #cv.imwrite("../catkin_ws/src/detect_mechanical_parts/data/image_detections_pho.png", image)
        blob = cv.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.__output_layer)
        return outputs

    def create_img_result(self, image, detection):
        (x, y) = (detection.x, detection.y)
        (w, h) = (detection.w, detection.h)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255) , 2)


    
    def process_image(self, outputs, image):
        boxes, confidences, classIDs = [], [], []

        parents, children = [], []

        bbox_part = [-1,0,0,0,0]

        H, W = image.shape[:2]
                                    
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                #print(detection)
                classID = np.argmax(scores)
                confidence = scores[classID]
                                        
                if confidence > self.conf_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    if self.labels[classID] == self.obj: #self.obj_pt: #"oil_separator_crankcase_castiron_pt1":
                        bbox_part = [classID, x, y, int(width), int(height)]
                        children.append(bbox_part)
                        parents.append(self.findParent(outputs, bbox_part, H, W))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        iOm = self.indexOfMax(parents)

        if iOm != -1:
            bbox_part = parents[iOm] #children[iOm]

                                        
        idxs = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=self.conf_threshold, nms_threshold=self.nms_threshold)
                                        
        if len(idxs)>0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                                        
                clr = [int(c) for c in self.bbox_colors[classIDs[i]]]
                                        
                cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)
                cv.putText(image, "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        else:
            print("----------------------NESSUN OGGETTO--------------------")
        return bbox_part, idxs, boxes, classIDs


    def indexOfMax(self, parents):
        maxArea = -1
        idx = -1
        for i in range(len(parents)):
            parent = parents[i]
            area_i = parent[3] * parent[4]
            if area_i > maxArea:
                maxArea = area_i
                idx = i
        return idx


    def findParent(self, outputs, bb, H, W):
        boxes, confidences, classIDs = [], [], []

        bb_current = [-1,0,0,0,0]
                                    
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                                        
                if confidence > self.conf_threshold and self.labels[classID] == self.obj: #"oil_separator_crankcase_castiron":
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    bb_current = [classID, x, y, int(width), int(height)]
                    if self.isIn(bb, bb_current):
                        return bb_current
        return bb_current

    
    def isIn(self, bb, bb_c):
        c_x = bb[1] + bb[3]/2
        c_y = bb[2] + bb[4]/2
        if bb_c[1] <= c_x and c_x <= (bb_c[1] + bb_c[3]):
            if bb_c[2] <= c_y and c_y <= (bb_c[2] + bb_c[4]):
                return True
        return False
