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

def main_2(robot, gripper):
    gripper.move(255,125,125)
    joint_state = [1.687131643295288, -1.0587236446193238, 1.5592082182513636, -2.074252267877096, -1.5726125876056116, 0.037265609949827194]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    joint_state = degreestorad([-5.61,-83.95,112.70,-119.79,-90.07,-5.48])

def degreestorad(list):
     for i in range(6):
          list[i]=list[i]*(math.pi/180)
     return(list)    
 
HOST = "192.168.0.2"
PORT = 30003

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
    joint_state = [1.343822717666626, -1.5149623540094872, 1.9113667646991175, -1.9754559002318324, -1.5370000044452112, 0.06107468530535698]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    joint_state = degreestorad([-5.61,-83.95,112.70,-119.79,-90.07,-5.48])

def main_5(robot, gripper):
    joint_state = [1.3438105583190918, -1.5019107994488259, 1.933467213307516, -2.010717054406637, -1.5371387640582483, 0.061162471771240234]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    gripper.move(0,125,125)

def main_6(robot, gripper):
    joint_state = [1.3438812494277954, -1.5379605305245896, 1.8642199675189417, -1.905468603173727, -1.536783520375387, 0.0609099380671978]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

def main_7(robot, gripper):
    joint_state = [1.3438105583190918, -1.5019107994488259, 1.933467213307516, -2.010717054406637, -1.5371387640582483, 0.061162471771240234]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    gripper.move(255,125,125)

def main_8(robot, gripper):
    joint_state = [1.3439116477966309, -1.585302015344137, 1.6429570356952112, -1.636733194390768, -1.5360587278949183, 0.0600738525390625]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

def main_9(robot, gripper):
    joint_state = [1.1097872257232666, -1.1103881162456055, 1.6571067015277308, -2.1457120380797328, -1.5382736364947718, -0.5338419119464319]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)

def main_10(robot, gripper):
    joint_state = [1.1097524166107178, -1.0942512613585968, 1.6702635923968714, -2.174934049645895, -1.538393799458639, -0.5338013807879847]
    robot.move_joint_list(joint_state, 0.25, 0.5, 0.02)
    gripper.move(0,125,125)

if __name__=="__main__":
     main(robot_1, gripper)
     main_2(robot_1, gripper)
     main_3(robot_1, gripper)
     main_4(robot_1, gripper)
     main_5(robot_1, gripper)
     main_6(robot_1, gripper)
     main_7(robot_1, gripper)
     main_8(robot_1, gripper)
     main_9(robot_1, gripper)
     main_10(robot_1, gripper)
