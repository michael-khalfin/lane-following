#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneHoughConfig
from geometry_msgs.msg import Twist, TwistStamped
import numpy as np
from math import log, sin, cos, atan, exp
import statistics
import time


class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.vel_msg.angular.z = 0
        self.empty = Empty()
        self.twist = TwistStamped()
        self.rate = rospy.Rate(20)
        
        self.median_list=[0]*4

        self.cols = 0 # set later
        self.rows = 0

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
        

        self.config = None
        self.srv = Server(FollowLaneHoughConfig, self.dyn_rcfg_cb)

        self.config.canny_thresh_l = 20
        self.config.canny_thresh_u = 120

        #while( not rospy.is_shutdown() ):
            #rospy.Subscriber('/vehicle/twist', TwistStamped, self.vel_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/red_topic', Bool, red_callback)

        #Red Blob Detection Variables
        self.n_laps = 0
        self.tot_frames = 0
        #self.sum_steer_err = 0 Implement Later 
        self.avg_steer_err = 0
        self.t0 = 0
        self.is_Driving = False
        self.perimeter = 106.97
        

        self.rate.sleep()
        

    def dyn_rcfg_cb(self, config, level):
        rospy.logwarn("Got config")
        self.config = config
        return config

    def red_callback(msg):
        global r_detected, n_laps, tot_frames, drive, t0
        if msg.data == False: # found STOP sign and now gone.
            self.n_laps += 1
            #avg_steer_err = sum_steer_err/tot_frames Implement Later
        
            t1 = rospy.Time.now().to_sec()
            dt = t1 - t0
            s = perimeter/dt # meter per second
            km_h = s*3.6 # 3,600m seconds / 1,000 meters (1km)
            miles_h = km_h * 0.62137119223733
            print(f"** Lap#{n_laps}, t taken: {dt:.2f} seconds")
            print(f"   Avg speed: {s:.2f} m/s, {km_h:.2f} km/h, {miles_h: .2f} miles/h")
            print(f"   Avg steer centering err = {0}") #implement later
            self.t0 = rospy.Time.now().to_sec()
            #sum_steer_err = 0
            self.tot_frames = 0
        
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
        image = image[504:]
        #Process the image to allow for hough lines to be drawn
        proc_image = self.preprocess(image)

        #Theta is set to 1 degree = pi/180 radians = 0.01745329251
        lines = cv2.HoughLinesP(proc_image, 
                               rho=self.config.lines_rho, 
                               theta=0.01745329251, 
                               threshold=self.config.lines_thresh,
                               minLineLength=70,
                               maxLineGap=4
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
                # if slope > 0:
                #     slope *= 5
                #     pass
                if isinstance(slope, np.float64) and not np.isnan(slope):
                    slopes.append(slope)
            
            image=self.drive_2_follow_line(lines,image,slopes)
                

        
        cv2.imshow("My Image Window", image)
        cv2.imshow("BW_Image", proc_image)
        cv2.waitKey(1)

    def preprocess(self, orig_image):
        """
        Inputs:
            orig_image: original bgr8 image before preprocessing
        Outputs:
            bw_image: black-white image after preprocessing
        """

        #blur_image = cv2.medianBlur(orig_image,self.config.blur_kernal)
        blur_image = cv2.GaussianBlur(orig_image,(9,9),0)
        

        (rows, cols, channels) = blur_image.shape
        self.cols = cols
        self.rows=rows
        blur_image=cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
        
        #min_t is minty
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

            
        canny_image = cv2.Canny(blur_image,lower_canny_thresh,upper_canny_thresh,apertureSize=3)

        blob_size=self.config.dilation_base
        dilation_size=(2*blob_size+1,2*blob_size+1)
        dilation_anchor=(blob_size,blob_size)
        dilate_element=cv2.getStructuringElement(cv2.MORPH_RECT,dilation_size,dilation_anchor)
        bw_image=cv2.dilate(canny_image,dilate_element)
        return bw_image

    def sigmoid(self, x, L=1, k=-1, x0=0):
        return L / (1 + exp(k*(x-x0)))

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
        
        if not self.is_Driving and self.config.enable_drive:
            self.is_Driving = True
            self.t0 = rospy.Time.now().to_sec()
        elif self.is_Driving and self.config.enable_drive = False
            self.is_Driving = False

        if self.config.enable_drive:
            
            self.vel_msg.linear.x = self.config.speed
            try:
                neg = [i for i in slopes if i < 0]
                pos = [i for i in slopes if i > 0]
                #slope=sum(slopes)/len(slopes)
                slope = (len(neg)/(len(pos) + len(neg)) * sum(neg) + len(pos)/(len(pos) + len(neg)) * sum(pos)) / len(slopes)

                # output a line to show the slope
                # theta = abs(atan(slope))
                # x = int(cos(theta) * 200)
                # y = int(sin(theta) * 200)
                # cv2.line(image,(int(mid),int(self.rows-1)),(int(mid - x),int(self.rows - 1 - y)),(255,0,0),2)
                # cv2.line(image,(0,0),(100,100),(255,0,0),2)

                self.median_list.pop(0)
                self.median_list.append(slope)
                
            except Exception as e:
                rospy.logwarn(f"No slopes: {e}")
                self.vel_msg.angular.z = 0

            median=statistics.median(self.median_list)
            #rospy.logwarn(median)
            if median > 0:
                self.vel_msg.angular.z = self.sigmoid(median, L=.3, k=-22, x0=.42)
                #self.vel_msg.angular.z = log(median) * self.config.turn_speed_const #* self.config.speed 
            elif median < 0:
                self.vel_msg.angular.z = -1 * self.sigmoid(-1 * median, L=.3, k=-22, x0=.42)
                # self.config.turn_speed_const
            else:
                self.vel_msg.angular.z = 0
            
        else:
            rospy.logwarn(f"else state")
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        
        rospy.logwarn(f"publishing {self.vel_msg.linear.x}, {self.vel_msg.angular.z}")

        #self.enable_car.publish(Empty())
        self.velocity_pub.publish(self.vel_msg)

        #tot_frames is for red blob
        tot_frames += 1

        return image
        #self.twist.linear.x >= .7:

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    FollowLine()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
