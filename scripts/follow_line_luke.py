#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneLukeConfig
from geometry_msgs.msg import Twist

vel_msg = Twist()
empty = Empty()
bridge = CvBridge()

thresh = 210

def dyn_rcfg_cb(config, level):
    global speed, drive, percent_white_max, percent_white_min
    #thresh = config.thresh
    speed = config.speed
    drive = config.enable_drive
    percent_white_max = config.percent_white_max
    percent_white_min = config.percent_white_min
    return config

def camera_callback(ros_image):
    global bridge, rows, cols, channels, thresh
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

    #rate = rospy.Rate(20)
    
    #cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    cv_image = cv_image[504:]
    cv_image = cv2.medianBlur(cv_image,9)
    
    (rows,cols,channels) = cv_image.shape

    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    ret, bw_image = cv2.threshold(gray_image, # input image
                                    thresh,     # threshol_value,
                                    255,        # max value in image
                                    cv2.THRESH_BINARY) # threshold type

    #Dynamic Thresholding
    num_white_pix = cv2.countNonZero(bw_image)
    total_pix = rows * cols

    percent_white = num_white_pix / total_pix * 100
    
    #percent_white_max = 13
    #percent_white_min = 5
    threshold_max = 248
    threshold_min=0

    #difference = percent_white - percent_white_max
    
   
    #New while loop to calculate thresh
    #change = 2 ** abs(percent_white - percent_white_max)
    
    change=32

    while(percent_white > percent_white_max) or (percent_white < percent_white_min):
        if percent_white > percent_white_max:
            thresh+=change
            if thresh>threshold_max:
                thresh=threshold_max
        elif percent_white < percent_white_min:
            thresh-=change
            if thresh<threshold_min:
                thresh=threshold_min
        else:
            break
        ret, bw_image = cv2.threshold(gray_image, # input image
                                    thresh,     # threshol_value,
                                    255,        # max value in image
                                    cv2.THRESH_BINARY) # threshold type
        num_white_pix = cv2.countNonZero(bw_image)
        percent_white = num_white_pix / total_pix * 100
        change/=2
        if change<2:
            break

    #Old while loop to calculate thresh
    # if percent_white > percent_white_max:
    #     while(percent_white > percent_white_max):
    #         thresh+=1
    #         ret, bw_image = cv2.threshold(gray_image, # input image
    #                                 thresh,     # threshol_value,
    #                                 255,        # max value in image
    #                                 cv2.THRESH_BINARY) # threshold type
    #         num_white_pix = cv2.countNonZero(bw_image)
    #         percent_white = num_white_pix / total_pix * 100
    #         change/=2
    #         if thresh > threshold_max:
    #             thresh = threshold_max
    #             break
    # elif percent_white < percent_white_min:
    #     while(percent_white < percent_white_min):
    #         thresh-=1
    #         ret, bw_image = cv2.threshold(gray_image, # input image
    #                                 thresh,     # threshol_value,
    #                                 255,        # max value in image
    #                                 cv2.THRESH_BINARY) # threshold type
    #         num_white_pix = cv2.countNonZero(bw_image)
    #         percent_white = num_white_pix / total_pix * 100
    #         if thresh < threshold_min:
    #             thresh = threshold_min
    #             break
    print(f"The percent white is: {percent_white}%")
    print(f"The Threshold is: {thresh}")
    
    #bw_image=cv2.medianBlur(bw_image,12)
    blob_size=2
    dilation_size=(2*blob_size+1,2*blob_size+1)
    dilation_anchor=(blob_size,blob_size)
    dilate_element=cv2.getStructuringElement(cv2.MORPH_RECT,dilation_size,dilation_anchor)
    #bw_image=cv2.dilate(bw_image,dilate_element)

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
        #speed = 2

    

    # if difference < 10 and difference > 0:
    #     thresh += 2 #arbitrary value
    # elif difference > 10:
    #     thresh += int(0.3 * difference)
    # elif percent_white < percent_white_min:
    #     thresh -= 2

    


    drive_2_follow_line(cx,cy)
    cv2.imshow("My Image Window", cv_image)
    cv2.imshow("BW_Image",bw_image)
    cv2.waitKey(3)


def drive_2_follow_line(cx, cy):  
    mid = cols/2 +100
    enable_car.publish(empty)

    if drive:
        rospy.loginfo(f"DRIVING {speed} m/s")
        vel_msg.linear.x = speed  
        if cx > mid+50:
            vel_msg.angular.z = .7 * (mid-cx)/mid
        elif cx < mid-50:
            vel_msg.angular.z = .7 * (mid-cx)/mid
        else:
            vel_msg.angular.z = 0
    else:
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0

    velocity_pub.publish(vel_msg)

if __name__ == '__main__':
    rospy.loginfo("Follow line initialized")
    rospy.init_node('follow_line', anonymous=True)
    rospy.Subscriber('/camera/image_raw', Image, camera_callback)
    enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
    velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
    srv = Server(FollowLaneLukeConfig, dyn_rcfg_cb)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass