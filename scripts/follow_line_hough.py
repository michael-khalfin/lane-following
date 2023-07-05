#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneHoughConfig
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String 
import numpy as np
from math import log, sin, cos, atan, exp
import statistics
import time
import pandas as pd
import os
import sys
from reg_model import RegModel

class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        rospy.loginfo(os.path.abspath('follow_line_hough.py'))
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.vel_msg.angular.z = 0
        self.empty = Empty()
        self.twist = TwistStamped()
        self.rate = rospy.Rate(20)
        
        self.median_list=[0]*4

        self.cols = 0 # set later
        self.rows = 0
        
        # red detect
        self.drive_pub = rospy.Publisher('/drive_enabled', Bool, queue_size=1)
        rospy.Subscriber('/red_detect_topic', String, self.red_callback)

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
        self.median_pub = rospy.Publisher('median', Float64, queue_size=1)
        self.slope_pub = rospy.Publisher('slope', Float64, queue_size=1)
        self.wslope_pub = rospy.Publisher('wslope', Float64, queue_size=1)


        self.config = None
        self.srv = Server(FollowLaneHoughConfig, self.dyn_rcfg_cb)

        self.config.canny_thresh_l = 20
        self.config.canny_thresh_u = 120

        #self.model = RegModel('../actor_ws/src/follow_lane_pkg/scripts/2023-06-29-11-22-51.bag', model_name=2)

        #while( not rospy.is_shutdown() ):
            #rospy.Subscriber('/vehicle/twist', TwistStamped, self.vel_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)


        # Red Blob Detection Variables
        # self.initial_time = rospy.Time.now().to_sec()
        # self.n_laps = 0
        # self.tot_frames = 0
        # self.t0 = 0
        # #self.sum_steer_err = 0 Implement Later 
        # self.avg_steer_err = 0
        # self.is_Driving = False
        # self.perimeter = 106.97
        
        # self.data = {
        #     'time': [],
        #     'speed (m/s)': [],
        #     'speed (km/hr)': [],
        #     'speed (mi/hr)': [],
        #     'avg steering center': []
        # }

        self.rate.sleep()
        

    def dyn_rcfg_cb(self, config, level):
        #rospy.logwarn("Got config")
        self.config = config
        bool_val = Bool()
        bool_val.data = self.config.enable_drive
        
        self.drive_pub.publish(bool_val)
        return config


    def red_callback(self, msg):
        if len(msg.data) > 0:
            print(msg.data)
        return


    def vel_callback(self, msg: TwistStamped):
        self.twist = msg.twist
        #rospy.loginfo(self.twist.linear.x)
        #cv2.waitKey(2)

    def camera_callback(self, msg: Image):
        #rospy.loginfo("Got image")
        if not self.config:
            rospy.logwarn("Waiting for config...")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        #Resize the image before preprocessing
        image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows
        image = image[self.rows//2:]
        #Process the image to allow for hough lines to be drawn
        proc_image= self.preprocess(image)

        #Theta is set to 1 degree = pi/180 radians = 0.01745329251
        #threshold=100
        #rho=1
        #minLineLength=70
        #maxLineGap=4
        lines = cv2.HoughLinesP(proc_image, 
                               rho=self.config.lines_rho, 
                               theta=0.01745329251, 
                               threshold=self.config.lines_thresh,
                               minLineLength=self.config.minLineLength,
                               maxLineGap=self.config.maxLineGap
                               )
        
        
        if lines is not None:
            lines=[l[0] for l in lines]
            slopes=[]
            for l in lines:
                # Graph lines on proc_image
                # (l[0],l[1]),(l[2],l[3]) are start and end point respectively
                # (255,0,0) is color of line(blue)
                # 2 is thickness of line
                slope=0
                try:
                    slope=(l[1]-l[3])/(l[0]-l[2])
                except:
                    rospy.logwarn("Divided by zero in slopes")
                    continue
                if abs(slope)<0.25 or abs(slope)>100:
                    continue
                
                cv2.line(image,(l[0],l[1]),(l[2],l[3]),(255,0,0),2)
                if isinstance(slope, np.float64) and not np.isnan(slope):
                    slopes.append(slope)
            
            image=self.drive_2_follow_line(lines,image,slopes)
                



        cv2.imshow("My Image Window", image)
        cv2.imshow("BW Image", proc_image)
        # cv2.imshow("Canny BW_Image", proc_image)
        # cv2.imshow("New BW_Image1", new_proc)
        # cv2.imshow("New BW_Image2", new_proc2)
        #cv2.imwrite('../actor_ws/src/follow_lane_pkg/data/image.png', image)
        #cv2.imwrite('../actor_ws/src/follow_lane_pkg/data/image_bw.png', proc_image)
        #sys.exit()
        cv2.waitKey(1)

    def preprocess(self,orig_image):
        """
        Inputs:
            orig_image: original bgr8 image before preprocessing
        Outputs:
            bw_image: black-white image after preprocessing
        """

                #blur_image = cv2.medianBlur(orig_image,self.config.blur_kernal)
        #Make first black and white image
        blur_image = cv2.GaussianBlur(orig_image,(9,9),0)
        

        (rows, cols, channels) = blur_image.shape
        self.cols = cols
        self.rows=rows
        blur_image=cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
        
        lower_canny_thresh = 0
        upper_canny_thresh = 100
        max_white = 0
        count = 0
        for min_t in range(0,101,10):
            for max_t in range(min_t+50,min_t+121,10):
                temp_image = cv2.Canny(blur_image,min_t,max_t,apertureSize=3)
                p_white = cv2.countNonZero(temp_image)
                if p_white > max_white:
                    count += 1
                    max_white=p_white
                    lower_canny_thresh=min_t
                    upper_canny_thresh=max_t
                    if count >= 10:
                        break
                        pass
        
            
        canny_image = cv2.Canny(blur_image,lower_canny_thresh,upper_canny_thresh,apertureSize=3)
        #canny_image = cv2.Canny(blur_image,self.config.canny_thresh_l,self.config.canny_thresh_u,apertureSize=3)

        blob_size=self.config.blur_kernal
        dilation_size=(2*blob_size+1,2*blob_size+1)
        dilation_anchor=(blob_size,blob_size)
        dilate_element=cv2.getStructuringElement(cv2.MORPH_RECT,dilation_size,dilation_anchor)
        bw_image=cv2.dilate(canny_image,dilate_element)

        thresh=210
        #Make the second black and white image
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        ret, bw_image2 = cv2.threshold(gray_image, # input image
                                        thresh,     # threshold value
                                        255,        # max value in image
                                        cv2.THRESH_BINARY) # threshold type

        num_white_pix = cv2.countNonZero(bw_image2)
        total_pix = rows * cols
        percent_white = num_white_pix / total_pix * 100

        thresh_max = 248
        thresh_min = 0
        change = 64
        percent_white_min=2
        percent_white_max=3

        while (percent_white > percent_white_max) or (percent_white < percent_white_min):
            if percent_white > percent_white_max:
                thresh += change
                if thresh > thresh_max:
                    thresh = thresh_max
            elif percent_white < percent_white_min:
                thresh -= change
                if thresh < thresh_min:
                    thresh = thresh_min
            else:
                break
            ret, bw_image2 = cv2.threshold(gray_image, # input image
                                            thresh,     # threshold value,
                                            255,        # max value in image
                                            cv2.THRESH_BINARY) # threshold type
            num_white_pix = cv2.countNonZero(bw_image2)
            percent_white = num_white_pix / total_pix * 100
            change /= 2
            if change < 2:
                break
        bw_image2=cv2.dilate(bw_image2,dilate_element)

        dif_image=cv2.subtract(bw_image,bw_image2)
        dif_image2=cv2.subtract(bw_image2,bw_image)
        dif_image=cv2.add(dif_image,dif_image2)
        new_proc=cv2.add(bw_image,bw_image2)
        new_proc=cv2.subtract(new_proc,dif_image)


        return bw_image

    def sigmoid(self, x, L=0.18, k=-14.5, x0=0.405):
        return L / (1 + exp(k*(x-x0))) +0.01

    def drive_2_follow_line(self, lines,image,slopes):
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
        
        # if not self.is_Driving and self.config.enable_drive:
        #     self.is_Driving = True
        #     self.t0 = rospy.Time.now().to_sec()
        # elif self.is_Driving and not self.config.enable_drive:
        #     self.is_Driving = False

        if self.config.enable_drive:
            rospy.loginfo(len(slopes))
            self.vel_msg.linear.x = self.config.speed
            try:
                neg = [i for i in slopes if i < 0]
                pos = [i for i in slopes if i > 0]
                slope=sum(slopes)/len(slopes)
                wslope = (len(neg)/(len(pos) + len(neg)) * sum(neg) + len(pos)/(len(pos) + len(neg)) * sum(pos)) / len(slopes)
                self.slope_pub.publish(slope)
                self.wslope_pub.publish(wslope)

                # output a line to show the slope
                # theta = abs(atan(slope))
                # x = int(cos(theta) * 200)
                # y = int(sin(theta) * 200)
                # cv2.line(image,(int(mid),int(self.rows-1)),(int(mid - x),int(self.rows - 1 - y)),(255,0,0),2)
                # cv2.line(image,(0,0),(100,100),(255,0,0),2)

                self.median_list.pop(0)
                self.median_list.append(slope)
                
            except Exception as e:
                #rospy.logwarn(f"No slopes: {e}")
                self.vel_msg.angular.z = 0

            median=statistics.median(self.median_list)
            self.median_pub.publish(median)
            #rospy.loginfo(f'median: {median}')
            #rospy.logwarn(median)
            # if median > 0:
            #     self.vel_msg.angular.z = self.sigmoid(median, L=.3, k=-30, x0=.5)
            #     #self.vel_msg.angular.z = log(median) * self.config.turn_speed_const #* self.config.speed 
            # elif median < 0:
            #     self.vel_msg.angular.z = -1 * self.sigmoid(-1 * median, L=.3, k=-30, x0=.5)
            #     # self.config.turn_speed_const
            # else:
            #     self.vel_msg.angular.z = 0
            try:
                #self.vel_msg.angular.z = self.model.make_prediction(median)
                self.vel_msg.angular.z =self.sigmoid(median)
            except Exception as e:
                rospy.logwarn(f'{type(self.model.make_prediction(median))}, {e}')
            #rospy.logwarn(f'ang z: {self.vel_msg.angular.z}')
            
        else:
            #rospy.logwarn(f"else state")
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        
        #rospy.logwarn(f"publishing {self.vel_msg.linear.x}, {self.vel_msg.angular.z}")

        #self.enable_car.publish(Empty())
        self.velocity_pub.publish(self.vel_msg)

        # tot_frames is for red blob
        # self.tot_frames += 1

        return image
        # self.twist.linear.x >= .7:

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    FollowLine()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
