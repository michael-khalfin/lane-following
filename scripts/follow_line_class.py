#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneLukeConfig
from geometry_msgs.msg import Twist

class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.empty = Empty()

        self.cols = 0 # set later

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)

        self.config = None
        self.srv = Server(FollowLaneLukeConfig, self.dyn_rcfg_cb)

        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)

    def dyn_rcfg_cb(self, config, level):
        rospy.logwarn("Got config")
        self.config = config
        return config

    def camera_callback(self, msg: Image):
        rospy.loginfo("Got image")
        if not self.config:
            rospy.logwarn("Waiting for config...")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        image = image[504:]
        proc_image = self.preprocess(image, 210)

        try:
            max_c, cx, cy = self.find_center_point(proc_image)
            cv2.drawContours(image, max_c, -1, (255,255,0), 10)
            cv2.circle(image, (cx,cy), 10, (0,0,255), -1)
        except Exception as e:
            rospy.logwarn("Could not find contours")
            cx = 504

        self.drive_2_follow_line(cx)
        cv2.imshow("My Image Window", image)
        cv2.imshow("BW_Image", proc_image)
        cv2.waitKey(3)

def preprocess(self, orig_image):
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

        blob_size=self.config.dilation_base
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
        rospy.logwarn(bw_image)
        rospy.loginfo(bw_image2)


        return bw_image

    def find_center_point(self, bw_image) -> 'contour, centroid coordinates x, y':
        """
        Inputs:
            bw_image: black and white image
        Outputs:
            contour: so we can draw it
            x: x-coord of centroid
            y: y-coord of centroid
        Description:
            Identifies contours to locate white lines, and centroid
        """

        contours,hierarchy = cv2.findContours(bw_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return # leads to an exception
        
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                max_c = c
        
        try:
            M = cv2.moments(max_c)
        except Exception as e:
            rospy.logwarn(f"Couldn't find line\n{e}")

        cx = 504

        try:
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        except Exception as e:
            rospy.logwarn(f"Couldn't find line\nDriving forward slowly\n{e}")

        return max_c, cx, cy

    def drive_2_follow_line(self, cx) -> 'None':
        """
        Inputs:
            cx: centroid x
        Outputs:
            None
        Description:
            Self drive algorithm to follow line by rotating wheels to steer
            toward center of the line
        """

        mid = self.cols / 2 +100
        self.enable_car.publish(self.empty)

        if self.config.enable_drive:
            rospy.loginfo(f"DRIVING {self.config.speed} m/s")
            self.vel_msg.linear.x = self.config.speed
            if cx > mid + 50:
                self.vel_msg.angular.z = .7 * (mid - cx)/mid*self.config.speed
            elif cx < mid - 50:
                self.vel_msg.angular.z = .7 * (mid - cx)/mid*self.config.speed
            else:
                self.vel_msg.angular.z = 0
        else:
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        self.velocity_pub.publish(self.vel_msg)

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    FollowLine()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
