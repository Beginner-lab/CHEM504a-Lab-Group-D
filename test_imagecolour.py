import cv2
import numpy as np
import time
import os
import csv
import math
from datetime import datetime
from robotiq.robotiq_gripper import RobotiqGripper
from utils.UR_Functions import URfunctions as URControl

# Initialize robot and gripper
robot_1 = URControl(ip="192.168.0.2", port=30003)
gripper = RobotiqGripper()
gripper.connect("192.168.0.2", 63352)

def degreestorad(list):
    for i in range(6):
        list[i] = list[i] * (math.pi / 180)
    return list

import matplotlib.pyplot as plt

def detect_blue_presence_with_graph(show_frame=False, duration=30):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return False

    print(f"Scanning for blue color for {duration} seconds...")
    start_time = time.time()
    blue_detected = False
    timestamps = []
    intensities = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Blue Intensity")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Blue Intensity")
    ax.set_title("Live Blue Detection Intensity")
    ax.legend()
    plt.show(block=False)  # <- Add this to ensure figure shows up

    while True:
        current_time = time.time() - start_time
        if current_time >= duration:
            break

        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([65, 100, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        mean_intensity = float(np.mean(mask)) / 255.0

        # Contour check
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                blue_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                print(f"Detected blue color at: ({cx}, {cy})")
                if show_frame:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Update graph
        timestamps.append(current_time)
        intensities.append(mean_intensity)
        line.set_data(timestamps, intensities)
        ax.set_xlim(0, max(duration, current_time + 1))
        ax.set_ylim(0, max(0.2, max(intensities) + 0.1))
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        plt.pause(0.001)  # <- Force update on screen

        if show_frame:
            cv2.imshow("Blue Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cam.release()
    if show_frame:
        cv2.destroyAllWindows()

    plt.ioff()
    plt.show()  # Final display at end

    return blue_detected

def log_reaction_data(output_csv_file, color_status):
    with open(output_csv_file, mode='a', newline='') as csv_file:
        fieldnames = ['Timestamp', 'Color Status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writerow({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Color Status': color_status
        })

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

def main(robot, gripper, output_csv_file):
    time.sleep(30)
    blue_found = detect_blue_presence_with_graph(show_frame=True)
    log_reaction_data(output_csv_file, "Blue" if blue_found else "Colourless")

    if blue_found:  # adjust threshold as needed
        print("Blue color detected! Moving to position...")
        joint_state = [1.3438105583190918, -1.5019107994488259, 1.933467213307516, -2.010717054406637, -1.5371387640582483, 0.061162471771240234]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.3439116477966309, -1.585302015344137, 1.6429570356952112, -1.636733194390768, -1.5360587278949183, 0.0600738525390625]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [0.9685537219047546, -1.740039964715475, 1.877197567616598, -1.725760122338766, -1.5944035688983362, 0.054102420806884766]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.1097872257232666, -1.1103881162456055, 1.6571067015277308, -2.1457120380797328, -1.5382736364947718, -0.5338419119464319]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

        joint_state = [1.1097524166107178, -1.0942512613585968, 1.6702635923968714, -2.174934049645895, -1.538393799458639, -0.5338013807879847]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        gripper.move(0,125,125)
        log_reaction_data(output_csv_file, 'Blue')
    else:
        print("No blue color detected. Executing alternative task...")
        joint_state = [1.7440, -1.8587, 2.3711, -2.3154, -1.5049, 0.0991]
        robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
        # Find a change to remove else or figure out how to keep continuous

if __name__ == "__main__":
    output_csv_file = 'reaction_data.csv'
    if not os.path.exists(output_csv_file):
        with open(output_csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Timestamp', 'Color Status'])
            writer.writeheader()

    main_o1(robot_1, gripper)
    main_2(robot_1, gripper)
    main_3(robot_1, gripper)
    main_4(robot_1, gripper)
    main_5(robot_1, gripper)
    main(robot_1, gripper, output_csv_file)
