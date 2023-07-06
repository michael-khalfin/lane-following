#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import String 
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import DetectRedConfig
import time

vel_msg = Twist()
red_msg = String()

bridge = CvBridge() 
state = 'NO_RED'
global cf

n_laps = 0
tot_frames = 0
sum_steer_err = 0
avg_steer_err = 0
t0 = 0

drive = False
cb_counter = 0


def dyn_rcfg_cb(config, level):
  global cf, perimeter
  cf = config

  perimeter = 71.43 if cf.inner_lane else 86.32 if cf.outer_lane else 71.43 
  print(f"Perimeter is {perimeter}")
  return config # this is required

def drive_cb(msg):
    print(msg.data)
    global drive, t0
    if drive == False and msg.data == True:
        drive = True
        t0 = rospy.Time.now().to_sec()
        print(f"current time in seconds is: {t0}")
    elif drive == True and msg.data == False: 
        drive = False


def image_callback(ros_image): #get contour image
    global cb_counter
    cb_counter+=1
    
    global bridge, cols, state, n_laps, tot_frames, t0, perimeter

    try: #convert ros_image into an opencv-compatible imageadi
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8") #get image from the camera, and turn it into a CV Accesible image
    except CvBridgeError as e:
        print(e)

    cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv_image = cv_image[504:,300:]

    (rows,cols,channels) = cv_image.shape
    cv2.imshow("RGB image", cv_image)
    hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
  
    tcolLower = (cf.hue_l, cf.sat_l, cf.val_l) # Lower bounds of H, S, V for the target color
    tcolUpper =  (cf.val_h, cf.sat_h, cf.val_h)                              # Upper bounds of H, S, V for the target color <====
    mask4y = cv2.inRange(hsv_img, tcolLower, tcolUpper) # <=====
    cv2.imshow("Mask Image", mask4y)
    num_white_pix = cv2.countNonZero(mask4y)                # <====
    white_pct = (100* num_white_pix) / (rows * cols)

    if cb_counter > 3:
        if(state == 'NO_RED' and white_pct >= 0.6):

            state = 'RED'
            rospy.loginfo(f"Red Detected, Percentage: {white_pct}%")

        elif(state == 'RED' and white_pct <= 0.2):
            state = "NO_RED"
            
            n_laps+=1
            avg_steer_err = 1 #sum_steer_err/tot_frames
            
            t1 = rospy.Time.now().to_sec()
            dt = t1 - t0
            s = perimeter/dt # meter per second
            km_h = s*3.6 # 3,600m seconds / 1,000 meters (1km)
            miles_h = km_h * 0.62137119223733
            red_msg.data = f"** Lap#{n_laps}, t taken: {dt:.2f} seconds\n    Avg speed: {s:.2f} m/s, {km_h:.2f} km/h, {miles_h: .2f} miles/h\n    Avg steer centering err = {avg_steer_err}"
                        
            red_detect_pub.publish(red_msg)
            red_msg.data = ''
            # print(f"** Lap#{n_laps}, t taken: {dt:.2f} seconds")
            # print(f"   Avg speed: {s:.2f} m/s, {km_h:.2f} km/h, {miles_h: .2f} miles/h")
            # print(f"   Avg steer centering err = {avg_steer_err}")
            t0 = rospy.Time.now().to_sec()
            # sum_steer_err = 0
            # tot_frames = 0
    cv2.waitKey(3)

    


if __name__ == '__main__':

    rospy.init_node('red_detect_n', anonymous=True)  #makes the rospy node, calls it follow_lane
    # imgtopic = rospy.get_param("~imgtopic_name") #makes a new topic with the image value

    rospy.Subscriber('/drive_enabled', Bool, drive_cb) #subscribe to topic to see if enable_drive is true
    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    red_detect_pub = rospy.Publisher('/red_detect_topic', String, queue_size=1) #publish red detect messages to the follow_lane node
    # red_pub = rospy.Publisher('/red_topic', Bool, queue_size=1)

    srv = Server(DetectRedConfig, dyn_rcfg_cb)

    try:
        rospy.spin() #loop code until it is shutdown by the user
    except rospy.ROSInterruptException:
        pass
