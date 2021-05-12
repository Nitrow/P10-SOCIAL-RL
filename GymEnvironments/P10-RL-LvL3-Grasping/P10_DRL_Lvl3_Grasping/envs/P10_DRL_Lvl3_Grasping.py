# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime
import torch

import os

from controller import Robot, Motor, Supervisor, Connector


class P10_DRL_Lvl3_Grasping(gym.Env):

    def __init__(self):

        random.seed(1)
        self.id = "SAC - " + str(datetime.now())[:-7].replace(':','_') + 'P10_DRL_Lvl3_Grasping' 
        #self.id = '2021-04-15 09_44_43_SAC_P10_MarkEnv_SingleJoint_' 
        self.path = "data/" + self.id + "/"
        os.makedirs(self.path, exist_ok=True)
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.robot_connector = supervisor.getDevice("connector")
        self.robot_connector.enablePresence(TIME_STEP)
        self.isPresence = 0
        self.pastPresence = 0
        self.pastLocked = False
        self.isLocked = False
        self.pastMoveState = 1
        self.goal_pos = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("translation")
        self.goal_rot = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("rotation")
        self.goal_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN")
        self.fingers = [self.supervisor.getDevice('right_finger'), self.supervisor.getDevice('left_finger')]
        self.sensor_fingers = [self.supervisor.getDevice('right_finger_sensor'), self.supervisor.getDevice('left_finger_sensor')]
        
        self.sensor_fingers[0].enable(TIME_STEP)
        self.sensor_fingers[1].enable(TIME_STEP)
        
        self.timeout = 300
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        #self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.touch_sensors = [0] * 7#(len(self.joint_names)+1)
     
        self.done = True
        self.prev_dist = 0
        
        self.plot_rewards = []
        self.total_rewards = 0

        self.movement_state = 0  # 1 for upper state, 0 for lower state
        self.movement_state = 1  # 0 for open fingers, 1 for closed fingers

        self.epOutcome = ""

        self.reward = 1
        self.rewardstr = "Get presence: 1, close fingers: 1, lift up: 1"
        self.figure_file = self.path + "{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
        self._setTarget()

        self.up_pose = [16.63, -111.19, -63.15, -96.24, 89.47, 10.81]
        self.down_pose = [16.62, -119.16, -92.69, -58.74, 89.52, 11.01]

        self.counter = 0

        self.actionScale = 3
        # Action: open/close finger, rotate joint, go up/down
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(33,), dtype=np.float32)
        
        self.documentation = "Action space: move_down, move_up, close_fingers, rotate+, rotate-, open_fingers"
        self.documentation += "{} - Rewards {} - Timeout at {}\n".format(self.id, self.rewardstr, str(self.timeout))
        self.saveEpisode(self.documentation)
        
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.counter = 0
        self._getMotors()
        self._getSensors() 
        self._setTarget()
        self.total_rewards = 0    
        self.done = False
        state = self.getState()
        return np.asarray(state)


    def step(self, action):
        #print(self.counter)
        self.goal_node.resetPhysics()
        self._getMotors()
        self._getSensors()
        # Set actions
        self._action_moveFingers(round(action[0]))  # Open or close fingers
        self._action_moveTCP(round(action[1]))  # Go up or down
        self.motors[-1].setVelocity(action[2])  # Set the rotation
        # Execute actions
        self.supervisor.step(self.TIME_STEP)   
        # Get new state
        state = self.getState()
        self.counter = self.counter + 1
        reward = self._getReward()    
        if self.counter >= self.timeout:
            self.epOutcome = "Timeout"
            print("Timeout")
            self.done = True
        if self.success == 3 and self.endPosition
            self.epOutcome = "Success"
            print("Success")
            self.done = True
        if self._isCollision():
            self.epOutcome = "Collision"
            print("Collision")
            self._setTarget()
            self.done = True
        #print(self.done)
        self.total_rewards += reward
        if self.done:
            self.saveEpisode(str(round(self.total_rewards)) + ";")
            self.plot_learning_curve()
        #print(reward)
        return [state, float(reward), self.done, {}]


    def _getSensors(self):
        """
        Initializes the touch sensors
        """
        for i in range(len(self.touch_sensors)):
            self.touch_sensors[i] = self.supervisor.getDevice("touch_sensor"+str(i+1))
            self.touch_sensors[i].enable(self.TIME_STEP)


    def _getReward(self):
        """
        First reward: Get orientation right
            +1 if connector senses a presence, -1 if it doesn't (one time reward)
        Second reward: Learn to pick it up
            +1 if locks and closes finger
        Third reward: Learn to lift it up
            +1 if lifts up while locked and presence
        """
        reward = 0
        self.isPresence = min(self.robot_connector.getPresence(),1)
        self.isLocked = self.robot_connector.isLocked()

        if self.isPresence != self.pastPresence:
            reward = 1 if (self.isPresence and not self.pastPresence) else -1
        if reward and self.isLocked != self.pastLocked:
            reward = reward + 1 if (self.isLocked and not self.pastLocked) else reward - 1
        if reward == 2 and self.movement_state != self.pastMoveState:
            reward = reward + 1 if self.movement_state == 1 else reward - 1
        if reward == 3:
            #TODO: position check
            pass
        self.pastPresence = self.isPresence
        self.pastLocked = self.isLocked
        self.pastMoveState = self.movement_state

        return reward


    def _getMotors(self):
        """
        Initializes the motors and their sensors
        """
        for i in range(len(self.joint_names)):
            # Get motors
            self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(0)
            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)


    def _setTarget(self):
        rotation = random.choice(rotations)
        x = random.uniform(-0.05, 0.03)
        translation = [x, 0.84, 0.4]
        self.goal_rot.setSFRotation(rotation)
        self.goal_pos.setSFVec3f(translation)

    def render(self, mode='human'):
        pass


    def saveEpisode(self, reward):
        with open(os.path.join(self.path, "documentation.txt"), 'a') as f:
            f.write(reward)


    def getState(self):
        """
        1. Joint angle - self.sensors[-1] (1)
        2. can orientation - absolute (change to relative to tool?) (4)
        3. can position - absolute (change to relative to tool?) (3)
        4. finger status (open, closed) - binary
        5. move status (up, down) - binary
        """
        state = []
        state.append(self.sensors[-1])  # Get joint angle
        state += self.goal_rot.getSFRotation()
        state += self.goal_pos.getSFVec3f()
        state.append(self.finger_state)
        state.append(self.movement_state)
        return state

    def _action_moveFingers(self, mode=0):
        # 0 for open state, 1 for closed state
        if mode == 0:
            self.finger_state = 0
            self.fingers[0].setPosition(0.04)
            self.fingers[1].setPosition(0.04)
            self.robot_connector.lock()
        elif mode == 1:
            self.finger_state = 1
            self.fingers[0].setPosition(0.015)
            self.fingers[1].setPosition(0.015)
            self.robot_connector.unlock()


    def _action_moveTCP(self, mode=0):
        # 1 for upper state, 0 for lower state
        if mode == 0:
            self.movement_state = 1
            [self.motors[i].setPosition(math.radians(self.down_pose[i])) for i in range(len(self.down_pose))]
        elif mode == 1:
            self.movement_state = 0
            [self.motors[i].setPosition(math.radians(self.down_pose[i])) for i in range(len(self.down_pose))]


    def _util_positionCheck(self, pos, sens, limit = 0.1):
        if len(pos):
            #print("Target position at: {}, position is at {}".format(pos, [math.degrees(sens[i].getValue()) for i in range(len(pos))]))
            if all([abs(self.sensors[i].getValue() - math.radians(pos[i])) < limit for i in range(len(pos))]):
                movementLock = False
                return True 
        else: 
            return False


    def plot_learning_curve(self):
        self.plot_rewards.append(self.total_rewards)
        x = [i+1 for i in range(len(self.plot_rewards))]
        running_avg = np.zeros(len(self.plot_rewards))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(self.plot_rewards[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(self.figure_file)
    
