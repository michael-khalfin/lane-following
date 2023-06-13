#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneConfig
from geometry_msgs.msg import Twist

vel_msg = Twist()
bridge = CvBridge()

def dyn_rcfg_cb(config, level):
    global thresh, speed, drive
    thresh = config.thresh
    speed = config.speed
    drive = config.enable_drive
    return config

def image_callback(ros_image):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    global rows, cols, channels
    (rows,cols,channels) = cv_image.shape

    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    ret, bw_image = cv2.threshold(gray_image, # input image
                                    thresh,     # threshol_value,
                                    255,        # max value in image
                                    cv2.THRESH_BINARY) # threshold type

    contours,hierarchy = cv2.findContours(bw_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    
    max_area = 0
    cx,cy=0,0
    try:
        num = 0
        for c in contours:
            num += 1
            M = cv2.moments(c)
            cx += int(M['m10']/M['m00'])
            cy += int(M['m01']/M['m00'])
            cv2.drawContours(cv_image, c, -1, (0,0,255), 10)
            cv2.circle(cv_image, (cx,cy), 10, (0,0,0), -1)
        cx /= num
        cy /= num
        drive_2_follow_line(cx,cy)
    except:
        pass

    cv2.imshow("My Image Window", cv_image)
    cv2.waitKey(3)

def drive_2_follow_line(cx, cy):  
    mid = cols/2
    if drive:
        vel_msg.linear.x = speed  
        if cx > mid+10:
            vel_msg.angular.z = -0.1
            velocity_pub.publish(vel_msg)
        elif cx < mid-10:
            vel_msg.angular.z = 0.1
            velocity_pub.publish(vel_msg)
        else:
            vel_msg.angular.z = 0
            velocity_pub.publish(vel_msg)
    else:
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        velocity_pub.publish(vel_msg)

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    imgtopic = rospy.get_param("~imgtopic_name")
    rospy.Subscriber(imgtopic, Image, image_callback)
    velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    srv = Server(FollowLaneConfig, dyn_rcfg_cb)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass