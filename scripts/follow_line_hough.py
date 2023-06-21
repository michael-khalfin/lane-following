#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneHoughConfig
from geometry_msgs.msg import Twist, TwistStamped
import numpy as np
from math import log, sin, cos, atan
import time


class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.vel_msg.angular.z = 0
        self.empty = Empty()
        self.twist = TwistStamped()

        self.cols = 0 # set later
        self.rows = 0

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
        

        self.config = None
        self.srv = Server(FollowLaneHoughConfig, self.dyn_rcfg_cb)

        self.config.canny_thresh_l = 20
        self.config.canny_thresh_u = 120

        rospy.Subscriber('/vehicle/twist', TwistStamped, self.vel_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        

    def dyn_rcfg_cb(self, config, level):
        rospy.logwarn("Got config")
        self.config = config
        return config

    def vel_callback(self, msg: TwistStamped):
        self.twist = msg.twist
        #rospy.loginfo(self.twist.linear.x)
        #cv2.waitKey(2)

    def camera_callback(self, msg: Image):
        rospy.loginfo("Got image")
        if not self.config:
            rospy.logwarn("Waiting for config...")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        #REsize the image before preprocessing
        image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        image = image[504:]
        #Process the image to allow for hough lines to be drawn
        proc_image = self.preprocess(image)

        #Theta is set to 1 degree = pi/180 radians = 0.01745329251
        lines = cv2.HoughLinesP(proc_image, 
                               rho=self.config.lines_rho, 
                               theta=0.01745329251, 
                               threshold=self.config.lines_thresh,
                               minLineLength=50,
                               maxLineGap=0
                               )
        
        
        if lines is not None:
            lines=[l[0] for l in lines]
            slopes=[]
            for l in lines:
                #Graph lines on proc_image
                #(l[0],l[1]),(l[2],l[3]) are start and end point respectively
                #(255,0,0) is color of line(blue)
                #2 is thickness of line
                slope=0
                try:
                    slope=(l[1]-l[3])/(l[0]-l[2])
                except:
                    rospy.logwarn("Divided by zero in slopes")
                    continue
                if abs(slope)<0.25 or abs(slope)>100:
                    #rospy.loginfo(f"Begone {slope}")
                    continue
                # if ((l[1]-l[3])+(l[0]-l[2]))**0.5 <50:
                #     continue
                
                cv2.line(image,(l[0],l[1]),(l[2],l[3]),(255,0,0),2)
                #rospy.logwarn(type(slope))
                if isinstance(slope, np.float64) and not np.isnan(slope):
                    slopes.append(slope)
                    #rospy.loginfo(f"Adding {slope}")

            #rospy.logwarn(slopes)
            
            image=self.drive_2_follow_line(lines,image,slopes)
                

        
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
        self.enable_car.publish(self.empty)
        

        if self.config.enable_drive:
            if self.twist.linear.x >= .7:
                rospy.loginfo(f"DRIVING {self.config.speed} m/s")
                self.vel_msg.linear.x = self.config.speed
                try:
                    slope=sum(slopes)/len(slopes)

                    # output a line to show the slope
                    theta = abs(atan(slope))
                    #rospy.loginfo(theta)
                    x = int(cos(theta) * 200)
                    y = int(sin(theta) * 200)
                    #rospy.logwarn(str(theta) + " " + str(x) + " " + str(y))
                    rospy.logwarn(slope)
                    cv2.line(image,(int(mid),int(self.rows-1)),(int(mid - x),int(self.rows - 1 - y)),(255,0,0),2)
                    cv2.line(image,(0,0),(100,100),(255,0,0),2)

                    self.vel_msg.angular.z = slope * self.config.speed * self.config.turn_speed_const
                except Exception as e:
                    rospy.logwarn(f"No slopes: {e}")
                    self.vel_msg.angular.z = 0
                

                #self.velocity_pub.publish(self.vel_msg)
                # try:
                #     avg_left_slope=sum(left_slopes)/len(left_slopes)
                # except Exception as e:
                #     avg_left_slope = 0.1

                # try:
                #     avg_right_slope=sum(right_slopes)/len(right_slopes)
                # except Exception as e:
                #     avg_right_slope = 0.1
                # if avg_left_slope == 0:
                #     avg_left_slope = 0.1
                # if avg_right_slope == 0:
                #     avg_right_slope = 0.1

                # right_line=1
                # left_line=1
                
                
                # if (avg_left_slope>-1*self.config.slopes_thresh) and (avg_left_slope<0):
                #     #L-
                #     left_line=2
                # elif (avg_left_slope<self.config.slopes_thresh) and (avg_left_slope>0):
                #     #L+
                #     left_line=0
                # else:
                #     #L0
                #     left_line=1
                
                # if (avg_right_slope>-1*self.config.slopes_thresh) and (avg_right_slope<0):
                #     #R-
                #     left_line=2
                # elif (avg_right_slope<self.config.slopes_thresh) and (avg_right_slope>0):
                #     #R+
                #     right_line=0
                # else:
                #     #R0
                #     right_line=1

                
                # if left_line+right_line>2:
                #     #turn_right
                #     cv2.putText(image,f"Turn Right",(10,self.rows-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(125,125,125),2,cv2.LINE_AA)
                #     # number=(-1/log(float(abs(avg_left_slope)+abs(avg_right_slope))))*self.config.speed*self.config.turn_speed_const
                #     self.vel_msg.angular.z -= self.config.turn_speed_const
                #     if abs(self.vel_msg.angular.z)>self.config.turn_max:
                #         self.vel_msg.angular.z=-1*self.config.turn_max
                #     # elif abs(self.vel_msg.angular.z)<self.config.turn_min:
                #     #     self.vel_msg.angular.z=-1*self.config.turn_min
                # elif left_line+right_line<2:
                #     #turn left
                #     cv2.putText(image,f"Turn Left",(10,self.rows-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(125,125,125),2,cv2.LINE_AA)
                #     # number=(1/log(float(abs(avg_left_slope)+abs(avg_right_slope))))*self.config.speed*self.config.turn_speed_const
                #     self.vel_msg.angular.z += self.config.turn_speed_const
                #     if abs(self.vel_msg.angular.z)>self.config.turn_max:
                #         self.vel_msg.angular.z=self.config.turn_max
                #     # elif abs(self.vel_msg.angular.z)<self.config.turn_min:
                #     #     self.vel_msg.angular.z=self.config.turn_min
                # else:
                #     #go staight
                #     cv2.putText(image,f"Go Straight",(10,self.rows-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(125,125,125),2,cv2.LINE_AA)
                #     if abs(self.vel_msg.angular.z)<self.config.turn_speed_const:
                #         self.vel_msg.angular.z=0
                #     elif self.vel_msg.angular.z<0:
                #         self.vel_msg.angular.z +=self.config.turn_speed_const
                #     else:
                #         self.vel_msg.angular.z -=self.config.turn_speed_const
            else:
                self.vel_msg.linear.x += .01
                self.vel_msg.linear.z = 0



        else:
            rospy.logwarn(f"else state")
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        
        rospy.logwarn(f"publishing {self.vel_msg.linear.x}, {self.vel_msg.angular.z}")
        self.velocity_pub.publish(self.vel_msg)
        return image

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    FollowLine()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
