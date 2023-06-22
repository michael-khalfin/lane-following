#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import DetectRedConfig

vel_msg = Twist()
bridge = CvBridge() 
state = 'NO_RED'
global cf

def dyn_rcfg_cb(config, level):
  global cf
  cf = config
  return config # this is required

def image_callback(ros_image): #get contour image
    global bridge, cols, state

    try: #convert ros_image into an opencv-compatible image
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8") #get image from the camera, and turn it into a CV Accesible image
    except CvBridgeError as e:
        print(e)

    cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    (rows,cols,channels) = cv_image.shape
    cv2.imshow("RGB image", cv_image)
    hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
  
    tcolLower = (cf.hue_l, cf.sat_l, cf.val_l) # Lower bounds of H, S, V for the target color
    tcolUpper =  (cf.val_h, cf.sat_h, cf.val_h)                              # Upper bounds of H, S, V for the target color <====
    mask4y = cv2.inRange(hsv_img, tcolLower, tcolUpper) # <=====
    cv2.imshow('Red Mask', mask4y)
    num_white_pix = cv2.countNonZero(mask4y)                # <====
    white_pct = (100* num_white_pix) / (rows * cols)
    #rospy.loginfo(f"percent of red pixels: {white_pct}")

    msg = Bool()

    #rospy.loginfo(f"Red percentage: {white_pct}%")
    if(state == 'NO_RED' and white_pct >= 0.02): #.15
        rospy.loginfo("Red percentage condition met")
        msg.data = True
        red_pub.publish(msg)
        state = 'RED'
        rospy.loginfo(f"Red Detected, Percentage: {white_pct}%")
    elif(state == 'RED' and white_pct >= 0.02):
        pass
    elif(state == 'RED' and white_pct <= 0.01): # .01
        state = "NO_RED"
        msg.data = False
        red_pub.publish(msg)
    # else:
    #     msg.data = True
    #     red_pub.publish(msg)
        
    #publish message saying red is detected to follow_line
    cv2.waitKey(3)

if __name__ == '__main__':

    rospy.init_node('red_detect', anonymous=True)  #makes the rospy node, calls it follow_lane
    rospy.Subscriber('/camera/image_raw', Image, image_callback) #subscribes to the topic
    red_pub = rospy.Publisher('/red_topic', Bool, queue_size=1)
    srv = Server(DetectRedConfig, dyn_rcfg_cb)
    try:
        rospy.spin() #loop code until it is shutdown by the user
    except rospy.ROSInterruptException:
        pass