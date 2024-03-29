#!/usr/bin/env python3

import sys
import rospy
import cv2 as cv
from detector_node import DetectorNode
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Visualizer:

  def __init__(self, rec_video=False):
    rospy.init_node('visualizer', anonymous=True)

    #view_image_topic = rospy.get_param(f'/{rospy.get_name()}/view_image_topic')
    view_image_topic = rospy.get_param(rospy.get_name() + '/view_image_topic')
    #view_image_topic = rospy.get_param(rospy.get_name() + '/result_img_topic')
    

    self.__bridge = CvBridge()
    rospy.Subscriber(view_image_topic, Image, self.view_image)

    self.__writer = None
    self.__rec_video = rec_video

  def view_image(self,data):
    try:
      image = self.__bridge.imgmsg_to_cv2(data, 'bgr8')
    except CvBridgeError as cve:
      rospy.logerr(str(cve))
      return
    dsize = (int(image.shape[1] * 50 / 100), int(image.shape[0] * 50 / 100))
    output = cv.resize(image, dsize)
    cv.imshow('Object Detection ', output)
    cv.waitKey(0)
   
def main(args):
  visualizer =  Visualizer()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    rospy.logerr('Shutting down')
  cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

