#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneConfig
from geometry_msgs.msg import Twist

vel_msg = Twist()
empty = Empty()
bridge = CvBridge()

def dyn_rcfg_cb(config, level):
    global thresh, speed, drive
    thresh = config.thresh
    speed = config.speed
    drive = config.enable_drive
    return config

def camera_callback(ros_image):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

    #rate = rospy.Rate(20)
    
    #cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    cv_image = cv_image[504:]
    
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
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_c = c
    try:
        
    
        M = cv2.moments(max_c)
        cv2.drawContours(cv_image, max_c, -1, (255,255,0), 10)

    except Exception as e:
        rospy.loginfo(f"Couldnt find line\nDid not start driving\n{e}")
    cx=504
    try:
        cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        cv2.circle(cv_image, (cx,cy), 10, (0,0,255), -1)
        drive_2_follow_line(cx,cy)
    except Exception as e:
        rospy.loginfo(f"Couldnt find line\nDriving forward slowly\n{e}")
        global speed
        speed = 2

    drive_2_follow_line(cx,cy)
    cv2.imshow("My Image Window", cv_image)
    cv2.imshow("BW_Image",bw_image)
    cv2.waitKey(3)

def drive_2_follow_line(cx, cy):  
    mid = cols/2
    enable_car.publish(empty)

    if drive:
        rospy.loginfo(f"DRIVING {speed} m/s")
        vel_msg.linear.x = speed  
        if cx > mid+10:
            vel_msg.angular.z = .7 * (mid-cx)/mid
            velocity_pub.publish(vel_msg)
        elif cx < mid-10:
            vel_msg.angular.z = .7 * (mid-cx)/mid
            velocity_pub.publish(vel_msg)
        else:
            vel_msg.angular.z = 0
            velocity_pub.publish(vel_msg)
    else:
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        velocity_pub.publish(vel_msg)
    #velocity_pub.publish(vel_msg)

if __name__ == '__main__':
    rospy.loginfo("Follow line initialized")
    rospy.init_node('follow_line', anonymous=True)
    rospy.Subscriber('/camera/image_raw', Image, camera_callback)
    enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
    velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
    srv = Server(FollowLaneConfig, dyn_rcfg_cb)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass