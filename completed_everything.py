from robotiq.robotiq_gripper import RobotiqGripper
import sys
import os
import time
import math
import numpy as np
import cv2
from utils.UR_Functions import URfunctions as URControl
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

# === Initialization ===
robot_1 = URControl(ip="192.168.0.2", port=30003)
gripper = RobotiqGripper()
gripper.connect("192.168.0.2", 63352)

# Global variables
elapsed_seconds = []
coverages = []
avg_intensities = []
running = True
start_time = None
duration = 30  # seconds

# Create output CSV
output_csv_file = 'reaction_data.csv'
if not os.path.exists(output_csv_file):
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Seconds Elapsed', 'Blue Coverage', 'Avg Blue Intensity'])
        writer.writeheader()

# Create snapshots directory if not exist
if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

# Initialize camera once
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

# Setup Video Writer
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('camera_recording.avi', fourcc, 20.0, (frame_width, frame_height))

# === Functions ===

def degreestorad(lst):
    return [angle * (math.pi / 180) for angle in lst]

def detect_blue_intensity(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    blue_coverage = float(np.mean(mask)) / 255

    blue_channel = frame[:, :, 0]  # B channel in BGR
    avg_blue_intensity = float(np.mean(blue_channel)) / 255

    return blue_coverage, avg_blue_intensity

def log_reaction_data(output_csv_file, seconds_elapsed, blue_coverage, avg_blue_intensity):
    with open(output_csv_file, mode='a', newline='') as csv_file:
        fieldnames = ['Seconds Elapsed', 'Blue Coverage', 'Avg Blue Intensity']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'Seconds Elapsed': seconds_elapsed,
            'Blue Coverage': f"{blue_coverage:.4f}",
            'Avg Blue Intensity': f"{avg_blue_intensity:.4f}"
        })

def robot_initial_sequence(robot, gripper):
    robot.move_joint_list([1.6365958452224731, -1.5545825728080054, 1.570291821156637, -1.5709616146483363, -1.571491543446676, -1.2699757711231996e-05], 0.25, 0.5, 0.02)
    gripper.move(0, 125, 125)
    robot.move_joint_list([1.6871144771575928, -1.032527045612671, 1.5819867292987269, -2.1232530079283656, -1.5728023687945765, 0.037368759512901306], 0.25, 0.5, 0.02)
    gripper.move(255, 125, 125)
    robot.move_joint_list([1.687131643295288, -1.0587236446193238, 1.5592082182513636, -2.074252267877096, -1.5726125876056116, 0.037265609949827194], 0.25, 0.5, 0.02)
    robot.move_joint_list(degreestorad([93.77, -89.07, 89.97, -90.01, -90.04, 0.0]), 0.5, 0.5, 0.02)
    gripper.move(255, 255, 255)
    gripper.move(255, 125, 125)
    robot.move_joint_list([1.3439631462097168, -1.5233835850707074, 1.9184191862689417, -1.9740644894041957, -1.5370295683490198, 0.061187610030174255], 0.25, 0.5, 0.02)
    robot.move_joint_list([1.3439477682113647, -1.5092243042639275, 1.942242447529928, -2.0119878254332484, -1.5371320883380335, 0.06127524375915527], 0.25, 0.5, 0.02)

def camera_and_logging():
    global running, start_time
    start_time = time.time()
    last_logged_second = -1

    while running:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        out.write(frame)
        cv2.imshow('Camera View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

        seconds_elapsed = int(time.time() - start_time)

        if seconds_elapsed != last_logged_second:
            blue_coverage, avg_blue_intensity = detect_blue_intensity(frame)

            elapsed_seconds.append(seconds_elapsed)
            coverages.append(blue_coverage)
            avg_intensities.append(avg_blue_intensity)

            log_reaction_data(output_csv_file, seconds_elapsed, blue_coverage, avg_blue_intensity)

            snapshot_filename = os.path.join('snapshots', f'snapshot_{seconds_elapsed}s.jpg')
            cv2.imwrite(snapshot_filename, frame)
            print(f"Saved snapshot: {snapshot_filename}")

            last_logged_second = seconds_elapsed

        if seconds_elapsed >= duration:
            print(f"Reached {duration} seconds. Ending recording.")
            running = False
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()

def plot_reaction_live():
    fig, ax = plt.subplots(figsize=(10, 5))

    def update(frame):
        ax.clear()
        ax.plot(elapsed_seconds, coverages, label='Blue Coverage', marker='o')
        ax.plot(elapsed_seconds, avg_intensities, label='Avg Blue Intensity', marker='x')
        ax.set_xlabel('Seconds Elapsed')
        ax.set_ylabel('Normalized Value (0.0 - 1.0)')
        ax.set_title('Live Blue Metrics Over 30 Seconds')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, interval=1000)
    plt.show()

    # Save final plot
    plt.figure(figsize=(12, 6))
    plt.plot(elapsed_seconds, coverages, label='Blue Coverage', marker='o')
    plt.plot(elapsed_seconds, avg_intensities, label='Avg Blue Intensity', marker='x')
    plt.xlabel('Seconds Elapsed')
    plt.ylabel('Normalized Value (0.0 - 1.0)')
    plt.title('Final Blue Metrics Over 30 Seconds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('blue_metrics_plot.png')
    print("Final plot saved as 'blue_metrics_plot.png'.")

def robot_final_sequence(robot, gripper):
    robot.move_joint_list([1.3438105583190918, -1.5019107994488259, 1.933467213307516, -2.010717054406637, -1.5371387640582483, 0.061162471771240234], 0.25, 0.5, 0.02)
    robot.move_joint_list([1.3439116477966309, -1.585302015344137, 1.6429570356952112, -1.636733194390768, -1.5360587278949183, 0.0600738525390625], 0.25, 0.5, 0.02)
    robot.move_joint_list([0.9685537219047546, -1.740039964715475, 1.877197567616598, -1.725760122338766, -1.5944035688983362, 0.054102420806884766], 0.25, 0.5, 0.02)
    robot.move_joint_list([1.1116843223571777, -1.0967931312373658, 1.6086209456073206, -2.0743991337218226, -1.5374119917498987, -0.6345880667315882], 0.25, 0.5, 0.02)
    robot.move_joint_list([1.1116446256637573, -1.0614957374385376, 1.6291587988482874, -2.1298734150328578, -1.5371788183795374, -0.6345832983600062], 0.25, 0.5, 0.02)
    gripper.move(0, 125, 125)
    robot.move_joint_list([0.9685537219047546, -1.740039964715475, 1.877197567616598, -1.725760122338766, -1.5944035688983362, 0.054102420806884766], 0.25, 0.5, 0.02)

# === Main Execution ===
if __name__ == "__main__":
    robot_initial_sequence(robot_1, gripper)

    camera_thread = threading.Thread(target=camera_and_logging)
    camera_thread.start()

    plot_reaction_live()

    camera_thread.join()

    robot_final_sequence(robot_1, gripper)

    print("Experiment completed.")
