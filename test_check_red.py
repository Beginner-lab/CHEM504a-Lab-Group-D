from robotiq.robotiq_gripper import RobotiqGripper
import sys
import os
import time
import argparse
import math
import numpy as np
import cv2

from utils.UR_Functions import URfunctions as URControl
from robotiq.robotiq_gripper import RobotiqGripper


robot_1 = URControl(ip="192.168.0.2", port=30003)
gripper = RobotiqGripper()
gripper.connect("192.168.0.2", 63352)

def detect_red_color():
    # Initialize webcam
    cam = cv2.VideoCapture(0)

    # Check if the camera is opened correctly
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Create a window
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range for red (lower and upper bound to account for both red regions in HSV)
        lower_red1 = np.array([90, 50, 50])
        upper_red1 = np.array([120, 255, 255])
        lower_red2 = np.array([120, 50, 50])
        upper_red2 = np.array([150, 255, 255])

        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        # Combine both masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of the red color areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_detected = False

        # Check if red color is detected
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Only consider large enough areas (to avoid noise)
                red_detected = True
                break

        # Display the result
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        # Return the result
        return red_detected

    # Release the camera and close the window
    cam.release()
    cv2.destroyAllWindows()


def main(robot, gripper):
    red_detected = detect_red_color()  # Check if red color is detected

    if red_detected:
        print("Blue color detected! Moving to position...")
        joint_state = [1.7440741062164307, -1.858786245385641, 2.3711000124560755, -2.31543030361318, -1.5049341360675257, 0.09916827827692032]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.744102954864502, -1.6180201969542445, 2.371833149586813, -2.3157054386534632, -1.504852596913473, 0.09917159378528595]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        gripper.move(255,75,125)

        joint_state = [1.6685433387756348, -1.914468904534811, 1.8774221579181116, -1.5508024015328665, -1.4946983496295374, 0.054107118397951126]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [0.9685537219047546, -1.740039964715475, 1.877197567616598, -1.725760122338766, -1.5944035688983362, 0.054102420806884766]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.1097872257232666, -1.1103881162456055, 1.6571067015277308, -2.1457120380797328, -1.5382736364947718, -0.5338419119464319]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)


        joint_state = [1.1097524166107178, -1.0942512613585968, 1.6702635923968714, -2.174934049645895, -1.538393799458639, -0.5338013807879847]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        gripper.move(0,125,125)

    else:
        print("No red color detected. Moving to alternative position...")

        joint_state = [1.7440741062164307, -1.858786245385641, 2.3711000124560755, -2.31543030361318, -1.5049341360675257, 0.09916827827692032]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.744102954864502, -1.6180201969542445, 2.371833149586813, -2.3157054386534632, -1.504852596913473, 0.09917159378528595]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        gripper.move(255,75,125)

        joint_state = [1.6685433387756348, -1.914468904534811, 1.8774221579181116, -1.5508024015328665, -1.4946983496295374, 0.054107118397951126]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [0.9685537219047546, -1.740039964715475, 1.877197567616598, -1.725760122338766, -1.5944035688983362, 0.054102420806884766]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.1097872257232666, -1.1103881162456055, 1.6571067015277308, -2.1457120380797328, -1.5382736364947718, -0.5338419119464319]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)


        joint_state = [1.1097524166107178, -1.0942512613585968, 1.6702635923968714, -2.174934049645895, -1.538393799458639, -0.5338013807879847]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        gripper.move(0,125,125)

if __name__ == "__main__":
    main(robot_1, gripper)
