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
import matplotlib.image as mpimg
from PIL import Image
import threading

# === Initialization ===
robot_1 = URControl(ip="192.168.0.2", port=30003)
gripper = RobotiqGripper()
gripper.connect("192.168.0.2", 63352)

# === Global variables ===
elapsed_seconds = []
coverages = []
avg_intensities = []
roi_x, roi_y, roi_w, roi_h = 290, 63, 119, 120
camera_start_time = None
frame = None
frame_lock = threading.Lock()
running = True

# === Create folders and CSV ===
output_csv_file = 'reaction_data.csv'
if not os.path.exists(output_csv_file):
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Seconds Elapsed', 'Blue Coverage', 'Avg Blue Intensity'])
        writer.writeheader()

if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

# === Setup Camera ===
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('camera_recording.avi', fourcc, 20.0, (frame_width, frame_height))

# === Helper Functions ===
def detect_blue_intensity(frame):
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_coverage = float(np.mean(mask)) / 255

    h_chan, s_chan, v_chan = cv2.split(hsv)
    h_norm = h_chan.astype(np.float32) / 179.0
    s_norm = s_chan.astype(np.float32) / 255.0
    v_norm = v_chan.astype(np.float32) / 255.0

    hue_distance = np.abs(h_norm - 0.67)
    hue_score = 1.0 - np.minimum(hue_distance * 3, 1.0)
    weighted_score = hue_score * s_norm * v_norm
    avg_blue_intensity = np.mean(weighted_score)

    return blue_coverage, avg_blue_intensity

def log_reaction_data(seconds_elapsed, blue_coverage, avg_blue_intensity):
    with open(output_csv_file, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Seconds Elapsed', 'Blue Coverage', 'Avg Blue Intensity'])
        writer.writerow({
            'Seconds Elapsed': seconds_elapsed,
            'Blue Coverage': f"{blue_coverage:.4f}",
            'Avg Blue Intensity': f"{avg_blue_intensity:.4f}"
        })

def trigger_snapshot():
    with frame_lock:
        if frame is None:
            print("No frame available for snapshot.")
            return
        snapshot = frame.copy()

    current_second = int(time.time() - camera_start_time)

    # Draw ROI
    cv2.rectangle(snapshot, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

    # Process
    blue_coverage, avg_blue_intensity = detect_blue_intensity(snapshot)
    elapsed_seconds.append(current_second)
    coverages.append(blue_coverage)
    avg_intensities.append(avg_blue_intensity)
    log_reaction_data(current_second, blue_coverage, avg_blue_intensity)

    # Save snapshot
    snapshot_filename = os.path.join('snapshots', f'snapshot_{current_second}s.jpg')
    cv2.imwrite(snapshot_filename, snapshot)
    print(f"[{current_second}s] Snapshot taken and data logged.")

# === Robot Movement Functions ===
def robot_to_stir_position(robot):
    print(">> Lowering to stir position...")
    robot.move_joint_list([1.3682881593704224, -1.526927178060152, 1.952386204396383, -1.9967791042723597, -1.5470431486712855, -0.42640716234316045],
                          0.25, 0.5, 0.02)

def robot_to_measure_position(robot):
    print(">> Raising to measure position...")
    robot.move_joint_list([1.3683245182037354, -1.558522069161274, 1.8917859236346644,
                           -1.9045912228026332, -1.5467665831195276, -0.4266312758075159],
                          0.25, 0.5, 0.02)

def robot_initial_sequence(robot, gripper):
    gripper.move(0, 125, 125)
    robot.move_joint_list([1.7099213600158691, -1.803235193292135, 1.750559155141012,
                           -1.6404873333373011, -1.5056775251971644, -0.06382209459413701],
                          0.25, 0.5, 0.02)
    robot.move_joint_list([1.6499958038330078, -1.0374747079661866, 1.5701993147479456,
                           -2.0898853741087855, -1.4785235563861292, 0.03156324476003647],
                          0.25, 0.5, 0.02)
    gripper.move(255, 125, 125)
    robot.move_joint_list([1.6408220529556274, -1.64308561901235, 1.5115016142474573,
                           -1.40816184998069, -1.4801214377032679, 0.03160172700881958],
                          0.25, 0.5, 0.02)
    robot.move_joint_list([1.2401678562164307, -1.6424476108946742, 1.5115578810321253,
                           -1.4080355030349274, -1.5262039343463343, -0.4312809149371546],
                          0.25, 0.5, 0.02)

def robot_final_sequence(robot, gripper):
    print(">> Finalizing robot position...")
    robot.move_joint_list([0.9460409283638, -1.762350698510641, 1.7529242674456995, -1.5523029708168288, -1.5504415670977991, -0.4272378126727503], 0.25, 0.5, 0.02)
    robot.move_joint_list([1.1166839599609375, -1.1068567496589203, 1.5925772825824183, -2.0422846279540003, -1.5426672140704554, -0.4272106329547327], 0.25, 0.5, 0.02)
    robot.move_joint_list([1.1166908740997314, -1.056407706146576, 1.5918343702899378, -2.063462873498434, -1.5425637404071253, -0.42719871202577764], 0.25, 0.5, 0.02)
    gripper.move(0, 125, 125)

# === Camera Thread ===
def camera_loop():
    global frame, camera_start_time
    camera_start_time = time.time()
    while running:
        ret, new_frame = cam.read()
        if not ret:
            print("Camera error.")
            break
        with frame_lock:
            frame = new_frame.copy()

        out.write(new_frame)
        cv2.rectangle(new_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        cv2.imshow('Camera View', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    out.release()
    cv2.destroyAllWindows()

# === Run Experiment ===
def run_experiment(robot):
    for i in range(4):
        print(f"\n== Cycle {i + 1} ==")
        robot_to_stir_position(robot)
        time.sleep(10)
        robot_to_measure_position(robot)
        time.sleep(4)
        trigger_snapshot()
        time.sleep(1)

# === Plotting ===
def plot_reaction():
    # Line graph
    plt.figure(figsize=(12, 6))
    plt.plot(elapsed_seconds, coverages, label='Blue Coverage', marker='o')
    plt.plot(elapsed_seconds, avg_intensities, label='Avg Blue Intensity', marker='x')
    plt.xlabel('Seconds Elapsed')
    plt.ylabel('Normalized Value (0.0 - 1.0)')
    plt.title('Final Blue Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('blue_metrics_plot.png')
    print("Final plot saved as 'blue_metrics_plot.png'.")
    plt.show()

    # Snapshot strip
    snapshot_paths = [f'snapshots/snapshot_{t}s.jpg' for t in elapsed_seconds]
    fig, axes = plt.subplots(1, len(snapshot_paths), figsize=(4*len(snapshot_paths), 4))
    for i, path in enumerate(snapshot_paths):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].set_title(f"{elapsed_seconds[i]}s")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('snapshot_strip.png')
    print("Snapshot strip saved as 'snapshot_strip.png'.")
    plt.show()

    # ROI color bar
    colors = []
    for path in snapshot_paths:
        img = Image.open(path).convert('RGB').crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
        avg_color = np.array(img).mean(axis=(0, 1)) / 255.0
        colors.append(avg_color)

    fig, ax = plt.subplots(figsize=(10, 2))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, -0.3, f'{elapsed_seconds[i]}s', ha='center')
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Average ROI Color Over Time')
    plt.tight_layout()
    plt.savefig('roi_color_bar.png')
    print("ROI color bar saved as 'roi_color_bar.png'.")
    plt.show()

# === Main ===
if __name__ == "__main__":
    robot_initial_sequence(robot_1, gripper)
    cam_thread = threading.Thread(target=camera_loop)
    cam_thread.start()

    run_experiment(robot_1)
    running = False
    cam_thread.join()

    plot_reaction()
    robot_final_sequence(robot_1, gripper)
    print("Experiment completed.")
