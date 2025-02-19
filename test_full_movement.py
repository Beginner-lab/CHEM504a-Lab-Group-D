from robotiq.robotiq_gripper import RobotiqGripper
import sys
import os
import time
import argparse
import math
from PIL import ImageTk, Image
import numpy as np
import math
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'robotiq'))
from utils.UR_Functions import URfunctions as URControl
from robotiq.robotiq_gripper import RobotiqGripper

robot_1 = URControl(ip="192.168.0.2", port=30003)
gripper=RobotiqGripper()
gripper.connect("192.168.0.2", 63352)
def main(robot, gripper):
   gripper.move(0,125,125)
   joint_state = [1.6871144771575928, -1.032527045612671, 1.5819867292987269, -2.1232530079283656, -1.5728023687945765, 0.037368759512901306]
   robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
   joint_state = degreestorad([-5.61,-83.95,112.70,-119.79,-90.07,-5.48])
def degreestorad(list):
    for i in range(6):
         list[i]=list[i]*(math.pi/180)
    return(list)    
HOST = "192.168.0.2"
PORT = 30003
def main_2(robot, gripper):
   gripper.move(255, 255, 255)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'robotiq'))
from utils.UR_Functions import URfunctions as URControl
HOST = "192.168.0.2"
PORT = 30003
def main_3(robot, gripper):
   joint_state=degreestorad([93.77,-89.07,89.97,-90.01,-90.04,0.0])
   robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
   gripper.move(255, 255, 255)

def degreestorad(list):
    for i in range(6):
         list[i]=list[i]*(math.pi/180)
    return(list)
def main_4(robot, gripper):
   gripper.move(255,125,125)
   joint_state = [1.3636760711669922, -1.4639088225415726, 1.8849189917193812, -1.9444557629027308, -1.5683139006244105, -0.20309573808778936]
   robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
   joint_state = degreestorad([-5.61,-83.95,112.70,-119.79,-90.07,-5.48])

if __name__=="__main__":
    main(robot_1, gripper)
    main_2(robot_1, gripper)
    main_3(robot_1, gripper)
    main_4(robot_1, gripper)
