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
import tensorflow as tf
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
        
        self.median_list=[0]*10

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
        self.twist_pub = rospy.Publisher('twist',Float64,queue_size=1)


        self.config = None
        self.srv = Server(FollowLaneHoughConfig, self.dyn_rcfg_cb)

        self.config.canny_thresh_l = 20
        self.config.canny_thresh_u = 120

        # self.model = RegModel('../actor_ws/src/follow_lane_pkg/scripts/2023-07-05-15-23-41.bag', model_name=0)
        self.model = RegModel('/home/reu-actor/actor_ws/src/follow_lane_pkg/scripts/2023-07-06-13-55-27.bag', model_name=2)
        

        # while( not rospy.is_shutdown() ):
        rospy.Subscriber('/vehicle/twist', TwistStamped, self.vel_callback)
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
        #self.enable_car.publish(Empty())
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
        image = image[2*self.rows//3:]
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows
        #Process the image to allow for hough lines to be drawn
        proc_image= self.preprocess(image)

        image = cv2.resize(image, (500,500))

        #Theta is set to 1 degree = pi/180 radians = 0.01745329251
        #threshold=100
        #rho=1
        #minLineLength=70
        #maxLineGap=4
        lines=None
        lines = cv2.HoughLinesP(proc_image, 
                               rho=self.config.lines_rho, 
                               theta=0.01745329251, 
                               threshold=self.config.lines_thresh,
                               minLineLength=self.config.minLineLength,
                               maxLineGap=self.config.maxLineGap
                               )
        
        
        if lines is not None:
            rospy.logwarn(len(lines))
            lines=[l[0] for l in lines]
            slopes=[]
            lengths=[]
            for l in lines:
                # Graph lines on proc_image
                # (l[0],l[1]),(l[2],l[3]) are start and end point respectively
                # (255,0,0) is color of line(blue)
                # 2 is thickness of line
                slope=0
                try:
                    slope=(l[1]-l[3])/(l[0]-l[2])
                    length = ((l[1]-l[3])**2 + (l[0]-l[2])**2)**.5
                except:
                    rospy.logwarn("Divided by zero in slopes")
                    continue
                if abs(slope)<0.25 or abs(slope)>100:
                    continue
                
                cv2.line(image,(l[0],l[1]),(l[2],l[3]),(255,0,0),2)
                if isinstance(slope, np.float64) and not np.isnan(slope) \
                and isinstance(length, np.float64) and not np.isnan(length):
                    slopes.append(slope)
                    lengths.append(length)
            image=self.drive_2_follow_line(lines,image,slopes,lengths)
                



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

        thresh=210
        #Make the second black and white image
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        ret, bw_image2 = cv2.threshold(gray_image, # input image
                                        thresh,     # threshold value
                                        255,        # max value in image
                                        cv2.THRESH_BINARY) # threshold type

        num_white_pix = cv2.countNonZero(bw_image2)
        total_pix = self.rows * self.cols
        percent_white = num_white_pix / total_pix * 100

        thresh_max = 254
        thresh_min = 0
        change = 64
        percent_white_min=4
        percent_white_max=5

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
        #Jack's fails @ around 238 with 4-5% target
        rospy.loginfo(f"Threshold: {thresh}")

        tf_img = np.copy(orig_image)
        #tf_img = tf_img[700:, :]
        tf_img = cv2.resize(tf_img, (500,500))
        #tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2GRAY)
        tf_img = np.expand_dims(tf_img, axis=0)
        predicted_img = artist.predict(tf_img)[0]
        (rows, cols, channels) = predicted_img.shape
        self.cols = cols
        self.rows=rows
        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)
        predicted_img=tf.keras.preprocessing.image.img_to_array(predicted_img,dtype='uint8')
        return predicted_img


        blur_image = cv2.medianBlur(orig_image,9,0)
        

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

        # thresh=210
        # #Make the second black and white image
        # gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        # ret, bw_image2 = cv2.threshold(gray_image, # input image
        #                                 thresh,     # threshold value
        #                                 255,        # max value in image
        #                                 cv2.THRESH_BINARY) # threshold type

        # num_white_pix = cv2.countNonZero(bw_image2)
        # total_pix = rows * cols
        # percent_white = num_white_pix / total_pix * 100

        # thresh_max = 248
        # thresh_min = 0
        # change = 64
        # percent_white_min=2
        # percent_white_max=3

        # while (percent_white > percent_white_max) or (percent_white < percent_white_min):
        #     if percent_white > percent_white_max:
        #         thresh += change
        #         if thresh > thresh_max:
        #             thresh = thresh_max
        #     elif percent_white < percent_white_min:
        #         thresh -= change
        #         if thresh < thresh_min:
        #             thresh = thresh_min
        #     else:
        #         break
        #     ret, bw_image2 = cv2.threshold(gray_image, # input image
        #                                     thresh,     # threshold value,
        #                                     255,        # max value in image
        #                                     cv2.THRESH_BINARY) # threshold type
        #     num_white_pix = cv2.countNonZero(bw_image2)
        #     percent_white = num_white_pix / total_pix * 100
        #     change /= 2
        #     if change < 2:
        #         break
        bw_image2=cv2.dilate(bw_image2,dilate_element)

        dif_image=cv2.subtract(bw_image,bw_image2)
        dif_image2=cv2.subtract(bw_image2,bw_image)
        dif_image=cv2.add(dif_image,dif_image2)
        new_proc=cv2.add(bw_image,bw_image2)
        new_proc=cv2.subtract(new_proc,dif_image)


        return bw_image

    def sigmoid(self, x, L=0.18, k=-14.5, x0=0.405):
        return L / (1 + exp(k*(x-x0))) +0.01

    def drive_2_follow_line(self, lines,image,slopes,lengths):
        """
        Inputs:
            lines: list of Hough lines in form of [x1,y1,x2,y2]
        Outputs:
            Image for the purposes of labelling
        Description:
            Self drive algorithm to follow lane by rotating wheels to steer
            toward center of the lane
        """
        
        mid = self.cols // 2 
        
        # if not self.is_Driving and self.config.enable_drive:
        #     self.is_Driving = True
        #     self.t0 = rospy.Time.now().to_sec()
        # elif self.is_Driving and not self.config.enable_drive:
        #     self.is_Driving = False

        if self.config.enable_drive:
            self.vel_msg.linear.x = 1
            center = mid


            left=[(l[0]+l[2])/2 for l in lines if ((l[0]+l[2])/2<self.cols/2)]
            right=[(l[0]+l[2])/2 for l in lines if ((l[0]+l[2])/2>self.cols/2)]
        
            if len(left)==0 or len(right)==0:
                if len(left)==0 and len(right)==0:
                    center = mid
                elif len(left)==0:
                    center = (sum(right)/len(right)+self.cols/4)//2
                else:
                    center = (sum(left)/len(left)+3*self.cols/4)//2
            else:
                center = (sum(left)/len(left) +sum(right)/len(right))//2

            cv2.line(image,(int(center),1),(int(center),self.rows),(0,255,0),2)
            cv2.line(image,(mid,1),(mid,self.rows),(0,0,255),2)


            #l_slopes=[(l[1]-l[3])/(l[0]-l[2]) for l in left if (l[0]!=l[2])]
            #r_slopes=[(l[1]-l[3])/(l[0]-l[2]) for l in right if (l[0]!=l[2])]


            ratio=self.vel_msg.angular.z = 0.6*(mid-center)/(mid)
            if center<mid-10:
                rospy.loginfo("Turn left!")
                self.vel_msg.angular.z = ratio
            elif center>mid+10:
                rospy.loginfo("Turn right!")
                self.vel_msg.angular.z = ratio
            else:
                rospy.logwarn("Go straight!")
                self.vel_msg.angular.z=0
            
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
    os.chdir("/home/reu-actor/actor_ws/src/jacks_pkg/scripts")
    artist = tf.keras.models.load_model("DL/artist")
    FollowLine()
    #os.chdir("../../../actor_ws/src/follow_lane_pkg/scripts")
    # os.chdir("/home/reu-actor/actor_ws/src/follow_lane_pkg/scripts")
    # artist = tf.keras.models.load_model("DL/artist")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
