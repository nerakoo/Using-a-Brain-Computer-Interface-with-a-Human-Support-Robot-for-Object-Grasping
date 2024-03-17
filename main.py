#!/usr/bin/env python3

import controller_manager_msgs.srv
import rospy
import trajectory_msgs.msg
import sys
from flask import Flask, request
import time
from tqdm import tqdm
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

NODE_NAME = 'robots_after_dark'

MAX_BUILD = 50

JOINT_L_NAMES = ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']
JOINT_R_NAMES = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']
JOINT_NEUTRALS = [-1.10, 1.4, 2.70, 1.71, -1.57, 1.39, 0.0]
JOINT_VELOCITIES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
JOINT_ACTION = [1.5, 0.58, 0.0, 1.01, -1.70, -0.0, -0.0]

count = 0
prev_time = 0
start = True
lock = False

rospy.init_node(NODE_NAME, disable_signals=True)
print('Running as:', NODE_NAME)

arm_l_pub = rospy.Publisher('/arm_left_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)
arm_r_pub = rospy.Publisher('/arm_right_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)

while arm_l_pub.get_num_connections() == 0:
    print('Waiting for controller...')
    rospy.sleep(1.0)

while arm_r_pub.get_num_connections() == 0:
    print('Waiting for controller...')
    rospy.sleep(1.0)

def neutral():
    global arm_r_pub
    print('Moving to neutral:', JOINT_NEUTRALS)
    traj = trajectory_msgs.msg.JointTrajectory()
    traj.joint_names = JOINT_R_NAMES
    p = trajectory_msgs.msg.JointTrajectoryPoint()
    p.positions = JOINT_NEUTRALS
    p.velocities = JOINT_VELOCITIES
    p.time_from_start = rospy.Duration(5)
    traj.points = [p]

    arm_r_pub.publish(traj)

def action():
    global lock, arm_r_pub

    lock = True
    print('Moving to action:', JOINT_ACTION)
    traj = trajectory_msgs.msg.JointTrajectory()
    traj.joint_names = JOINT_R_NAMES
    p = trajectory_msgs.msg.JointTrajectoryPoint()
    p.positions = JOINT_ACTION
    p.velocities = JOINT_VELOCITIES
    p.time_from_start = rospy.Duration(5)
    traj.points = [p]

    arm_r_pub.publish(traj)

    rospy.sleep(15.0)
    neutral()
    lock = False

@app.route('/emotiv_input', methods=['POST'])
def emotiv_input():
    global count, prev_time, start, lock

    if lock:
        return 'LOCKED'

    pbar = tqdm(total=MAX_BUILD)

    time_now = time.time() * 1000
    time_diff = time_now - prev_time
    if time_diff > 2000:
        print('Reset.')
        count = 0

    count = count + 1
    pbar.update(count)

    prev_time = time_now

    pbar.close()

    if count == MAX_BUILD:
        count = 0
        print('[', time_now, ']', 'Action!')
        action()

    return 'OK'

@app.route('/emotiv_difficulty', methods=['POST'])
def emotiv_difficulty():
    action()
    return 'OK'

app.run(host='0.0.0.0', port=5014)