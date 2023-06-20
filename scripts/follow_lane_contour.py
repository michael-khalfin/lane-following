#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneHoughConfig
from geometry_msgs.msg import Twist
import numpy as np
from math import abs,log


class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.empty = Empty()

        self.cols = 0 # set later
        self.rows = 0

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)

        self.config = None
        self.srv = Server(FollowLaneHoughConfig, self.dyn_rcfg_cb)

        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)

    def dyn_rcfg_cb(self, config, level):
        rospy.logwarn("Got config")
        self.config = config
        return config

    def camera_callback(self, msg: Image):
        #rospy.loginfo("Got image")
        if not self.config:
            rospy.logwarn("Waiting for config...")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        #REsize the image before preprocessing
        image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows
        image = image[self.rows//2:]
        #Process the image to allow for hough lines to be drawn
        proc_image = self.preprocess(image)
        left_image = proc_image[:self.rows//2][:self.cols//2]
        right_image = proc_image[:self.rows//2][self.cols//2:]

        left_contours,hierarchy = cv2.findContours(left_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        right_contours,hierarchy=cv2.findContours(right_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not left_contours or not right_contours:
            return None

        left_max_c=0
        left_max_area = 0
        for c in left_contours:
            #print(cv2.contourArea(c))
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            if area > left_max_area:
                if M['m00'] != 0:
                    lcx,lcy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                left_max_area = area
                left_max_c = c

        right_max_c=0
        right_max_area = 0
        for c in right_contours:
            #print(cv2.contourArea(c))
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            if area > right_max_area:
                if M['m00'] != 0:
                    rcx,rcy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                right_max_area = area
                right_max_c = c
        #draw the obtained contour lines(or the set of coordinates forming a line) on the original image
        cv2.drawContours(image, left_max_c, -1, (0,0,255), 10)
        cv2.drawContours(image, right_max_c, -1, (255,0,0), 10)
        
        cx=(lcx+rcx)/2
        cy=(lcy+rcy)/2
        cv2.circle(image, (cx,cy), 10, (0,255,0), -1) # -1 fill the circle
        self.drive_2_follow_line(cx,image)
                

        
        cv2.imshow("My Image Window", image)
        cv2.imshow("BW_Image", proc_image)
        cv2.waitKey(2)

    def preprocess(self, orig_image):
        """
        Inputs:
            orig_image: original bgr8 image before preprocessing
        Outputs:
            bw_image: black-white image after preprocessing
        """

        blur_image = cv2.medianBlur(orig_image,self.config.blur_kernal)
        

        (rows, cols, channels) = blur_image.shape
        self.cols = cols
        self.rows=rows
        blur_image=cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
        
        canny_image = cv2.Canny(blur_image,self.config.canny_thresh_l,self.config.canny_thresh_u,apertureSize=3)

        blob_size=self.config.dilation_base
        dilation_size=(2*blob_size+1,2*blob_size+1)
        dilation_anchor=(blob_size,blob_size)
        dilate_element=cv2.getStructuringElement(cv2.MORPH_RECT,dilation_size,dilation_anchor)
        bw_image=cv2.dilate(canny_image,dilate_element)
        return bw_image


    

    def drive_2_follow_line(self, cx,image):
        """
        Inputs:
            lines: list of Hough lines in form of [x1,y1,x2,y2]
        Outputs:
            Image for the purposes of labelling
        Description:
            Self drive algorithm to follow lane by rotating wheels to steer
            toward center of the lane
        """

        mid = self.cols / 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.enable_car.publish(self.empty)
        

        if self.config.enable_drive:
            self.config.vel_msg.linear.x=self.config.speed
            if self.config.speed<2:
                strength_ratio = 0.6*(mid-cx)/mid
            else:
            #turn harder at faster speeds
                strength_ratio = 0.8*(mid-cx)/mid
            if cx<mid-10:
                cv2.putText(image,f"Turn Left",(10,self.rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
                self.config.vel_msg.angular.z=strength_ratio
                self.velocity_pub.publish(self.vel_msg)
            elif cx>mid+10:
                cv2.putText(image,f"Turn Right",(10,self.rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
                self.vel_msg.angular.z=strength_ratio
                self.velocity_pub.publish(self.vel_msg)
            else: 
                cv2.putText(image,f"Go Staight",(10,self.rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
                self.vel_msg.angular.z=0
                self.velocity_pub.publish(self.vel_msg)

                

        else:
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        self.velocity_pub.publish(self.vel_msg)
        return image

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    FollowLine()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass