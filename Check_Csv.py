from robotiq.robotiq_gripper import RobotiqGripper
import sys
import os
import time
import argparse
import math
import numpy as np
import cv2
from utils.UR_Functions import URfunctions as URControl
from datetime import datetime
import csv

# Initialize robot and gripper
robot_1 = URControl(ip="192.168.0.2", port=30003)
gripper = RobotiqGripper()
gripper.connect("192.168.0.2", 63352)

def degreestorad(list):
    for i in range(6):
        list[i] = list[i] * (math.pi / 180)
    return list

# Define the function to detect blue color in the solution
def detect_blue_color():
    # Initialize webcam
    cam = cv2.VideoCapture(0)

    # Check if the camera is opened correctly
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        exit()

    time.sleep(5)  # Allow camera to focus
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range for blue
        lower_blue1 = np.array([90, 50, 50])
        upper_blue1 = np.array([120, 255, 255])
        lower_blue2 = np.array([120, 50, 50])
        upper_blue2 = np.array([150, 255, 255])

        mask1 = cv2.inRange(hsv_frame, lower_blue1, upper_blue1)
        mask2 = cv2.inRange(hsv_frame, lower_blue2, upper_blue2)

        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blue_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                blue_detected = True
                break

        cv2.imshow("test", frame)

        if blue_detected:
            print("Blue color detected!")
            return True

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

    return False

# Function to record reaction data into a CSV file
def log_reaction_data(output_csv_file, color_state):
    with open(output_csv_file, mode='a', newline='') as csv_file:
        fieldnames = ['Timestamp', 'Colour State']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the data to the CSV
        writer.writerow({'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Colour State': color_state})


# Main Check Colour Script
def main_o1(robot, gripper):
    gripper.move(0, 125, 125)
    joint_state = [1.6871144771575928, -1.032527045612671, 1.5819867292987269, -2.1232530079283656, -1.5728023687945765, 0.037368759512901306]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

    joint_state = degreestorad([-5.61, -83.95, 112.70, -119.79, -90.07, -5.48])

def main_2(robot, gripper):
    gripper.move(255, 125, 125)
    joint_state = [1.687131643295288, -1.0587236446193238, 1.5592082182513636, -2.074252267877096, -1.5726125876056116, 0.037265609949827194]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    joint_state = degreestorad([-5.61, -83.95, 112.70, -119.79, -90.07, -5.48])

def main_3(robot, gripper):
    joint_state = degreestorad([93.77, -89.07, 89.97, -90.01, -90.04, 0.0])
    robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    gripper.move(255, 255, 255)

def main_4(robot, gripper):
    gripper.move(255, 125, 125)
    joint_state = [1.3439631462097168, -1.5233835850707074, 1.9184191862689417, -1.9740644894041957, -1.5370295683490198, 0.061187610030174255]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    joint_state = degreestorad([-5.61, -83.95, 112.70, -119.79, -90.07, -5.48])

def main_5(robot, gripper):
    joint_state = [1.3439477682113647, -1.5092243042639275, 1.942242447529928, -2.0119878254332484, -1.5371320883380335, 0.06127524375915527]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

    joint_state = [1.3438105583190918, -1.5019107994488259, 1.933467213307516, -2.010717054406637, -1.5371387640582483, 0.061162471771240234]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    gripper.move(0, 125, 125)

    joint_state = [1.3438812494277954, -1.5379605305245896, 1.8642199675189417, -1.905468603173727, -1.536783520375387, 0.0609099380671978]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    time.sleep(20)

def main_6(robot, gripper):
    joint_state = [1.3438105583190918, -1.5019107994488259, 1.933467213307516, -2.010717054406637, -1.5371387640582483, 0.061162471771240234]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    gripper.move(255, 125, 125)

    joint_state = [1.3439116477966309, -1.585302015344137, 1.6429570356952112, -1.636733194390768, -1.5360587278949183, 0.0600738525390625]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

    joint_state = [1.744102954864502, -1.6180201969542445, 2.371833149586813, -2.3157054386534632, -1.504852596913473, 0.09917159378528595]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    gripper.move(0, 75, 125)

    joint_state = [1.7440741062164307, -1.858786245385641, 2.3711000124560755, -2.31543030361318, -1.5049341360675257, 0.09916827827692032]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

    joint_state = [0.7161983251571655, -2.0699573955931605, 2.027865711842672, -1.5948759518065394, -1.3347838560687464, -0.9307001272784632]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)


# Main Robot Execution and Colour Check
def main(robot, gripper, output_csv_file):
    
    # Now perform the colour check and log the result
    blue_detected = detect_blue_color()
    color_state = 'Blue' if blue_detected else 'Colourless'
    log_reaction_data(output_csv_file, color_state)

    if blue_detected:
        print("Blue color detected! Performing robotic actions for blue...")
        joint_state = [1.7440741062164307, -1.858786245385641, 2.3711000124560755, -2.31543030361318, -1.5049341360675257, 0.09916827827692032]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.744102954864502, -1.6180201969542445, 2.371833149586813, -2.3157054386534632, -1.504852596913473, 0.09917159378528595]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        gripper.move(255, 75, 125)

        log_reaction_data(output_csv_file, 'Blue')

    else:
        print("No blue color detected. Moving to alternative position...")
        while not blue_detected:
            joint_state = [1.7440741062164307, -1.858786245385641, 2.3711000124560755, -2.31543030361318, -1.5049341360675257, 0.09916827827692032]
            robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

            joint_state = [1.744102954864502, -1.6180201969542445, 2.371833149586813, -2.3157054386534632, -1.504852596913473, 0.09917159378528595]
            robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
            gripper.move(255, 75, 125)

            blue_detected = detect_blue_color()
            color_state = 'Blue' if blue_detected else 'Colourless'
            log_reaction_data(output_csv_file, color_state)

            time.sleep(10)

if __name__ == "__main__":
    output_csv_file = 'reaction_data.csv'
    
    # Add header to CSV if it doesn't exist
    if not os.path.exists(output_csv_file):
        with open(output_csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Timestamp', 'Colour State'])
            writer.writeheader()

    main_o1(robot_1, gripper, output_csv_file)
    main_2(robot_1, gripper, output_csv_file)
    main_3(robot_1, gripper, output_csv_file)
    main_4(robot_1, gripper, output_csv_file)
    main_5(robot_1 ,gripper, output_csv_file)
    main_6(robot_1 ,gripper, output_csv_file)
    main(robot_1, gripper ,output_csv_file)
