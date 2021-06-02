# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
from datetime import datetime
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from dqn_network import DQN

import os, sys

from controller import Robot, Motor, Supervisor, Connector

class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_dir='tmp/dqn'):
        super(DQN, self).__init__()
        self.name = "DQN_Grasping"
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # * unpacks a list
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = os.getcwd().split("/P10-XRL/")[0] + "/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Primitives-Conveyor/P10_DRL_Lvl3_Grasping_Primitives_Conveyor/envs/" + chkpt_dir
        #self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class P10_DRL_Lvl3_Grasping_Primitives_Conveyor(gym.Env):

    def __init__(self, itername, vdist, plt_avg):

        random.seed(1)
        self.vdist = vdist/100
        self.plt_avg = plt_avg
        self.test = True
        #self.own_path = "/home/harumanager/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Primitives/P10_DRL_Lvl3_Grasping_Primitives/envs/P10_DRL_Lvl3_Grasping_Primitives.py"
        self.own_path = os.getcwd().split("/P10-XRL/")[0] + "/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Primitives/P10_DRL_Lvl3_Grasping_Primitives/envs/P10_DRL_Lvl3_Grasping_Primitives.py"
        self.id = "DQN - " + str(datetime.now())[:-7].replace(':','_') + 'P10_DRL_Lvl3_Grasping_Primitives_' + itername
        #self.id = '2021-04-15 09_44_43_SAC_P10_MarkEnv_SingleJoint_' 
        self.path = "data/" + self.id + "/" if not self.test else "test/" + self.id + "/" 
        os.makedirs(self.path, exist_ok=True)
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.finger1 = self.supervisor.getFromDef("FINGER1")
        self.finger2 = self.supervisor.getFromDef("FINGER2")
        self.goal_pos = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("translation")
        self.goal_rot = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("rotation")
        self.goal_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN")
        self.fingers = [self.supervisor.getDevice('right_finger'), self.supervisor.getDevice('left_finger')]
        self.sensor_fingers = [self.supervisor.getDevice('right_finger_sensor'), self.supervisor.getDevice('left_finger_sensor')]
        self.tcp = self.supervisor.getFromDef("TCP")
        self.tcp_pos = self.supervisor.getFromDef("TCP").getField("translation")
        self.tcp_can_total_dist = 0
        self.tcp_can_vertical_dist = 0
        self.tcp_can_horizontal_dist = 0
        # Preparing Grasping DQN
        self.grasping_DQN = DQN(0, n_actions=8, input_dims=[7], fc1_dims=128, fc2_dims=128, chkpt_dir='models')
        print(self.grasping_DQN.checkpoint_file)
        self.grasping_DQN.load_checkpoint()

        self.sensor_fingers[0].enable(self.TIME_STEP)
        self.sensor_fingers[1].enable(self.TIME_STEP)
        self.timeout = 100

        self.conveyor = self.supervisor.getFromDef("CONVEYOR")
        #self.conveyor_speed_field = self.conveyor.getField("speed")#getSFFloat(), setSFFloat(new_speed)
        #self.conveyor_speed = 0.5

        self.rotations = self._util_readRotationFile('rotations.txt')#[0.577, 0.577, 0.577, 2.094]

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        #self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.touch_sensors = [0] * 7#(len(self.joint_names)+1)
     
        self.done = True
        self.prev_dist = 0
        
        self.plot_rewards = []
        self.total_rewards = 0
        # Different reward components:
        self.reward = 0
        self.partialReward = 0

        self.movement_state = 1  # 1 for upper state, 0 for lower state
        self.finger_state = 1  # 0 for open fingers, 1 for closed fingers
        self.pastFingerState = self.finger_state

        self.rewardstr = "successfulgrasp-1"
        self.figure_file = self.path + "{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
        self.bugged = False

        self._setTarget()

        self.up_pose = [16.63, -111.19, -63.15, -96.24, 89.47]
        self.down_pose = [16.60, -121.69, -94.63, -54.27, 89.52]

        self.counter = 0

        self.actionScale = 3
        # Action: open/close finger, rotate joint, go up/down
        self.grasp_action_shape = 8
        self.grasp_actions = [m.radians(22.5*i) for i in range(self.grasp_action_shape)]
        self.action_shape = 200
        self.state_shape = 1
        #self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_shape,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_shape)
        
        self.observation_space = spaces.Box(low=-200, high=200, shape=(self.state_shape,), dtype=np.float32)

        self.documentation = "Action space: move_down, move_up, close_fingers, rotate+, rotate-, open_fingers"
        self.documentation += "{} - Rewards {} - Timeout at {}\n".format(self.id, self.rewardstr, str(self.timeout))
        self.saveEpisode(self.documentation)
        print("Robot initilized!")
        

    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.conveyor.restartController()
        self.supervisor.step(self.TIME_STEP)  
        self.counter = 0
        self.bugged = False
        self.stepCounter = 0
        self.reward = 0
        self._getMotors()
        self._getSensors() 
        self._setTarget()
        self.finger1.resetPhysics()
        self.finger2.resetPhysics()
        self.total_rewards = 0    
        self.done = False
        self.motors[-1].setPosition(float(1.6))
        state = self.getState()
        if self.test: print("Initial state: ", state)
        return state


    def step(self, action):

        if self.test:
            print("--------------------")
            print("STEP: ", self.counter)
        self.partialReward = 0#-0.25
        #self.goal_node.resetPhysics()
        self._getMotors()
        self._getSensors()
        self.finger1.resetPhysics()
        self.finger2.resetPhysics()
        self.goal_node.resetPhysics()
        # Set actions
        if self.test: print("Setting target...")
        self._setTarget()
        if self.test: print("... target set!")
        for i in range(action):
            #graspState = self.getState(mode="grasp")
            #action = self.grasp(graspState)
            self.supervisor.step(self.TIME_STEP)
        if self.test: print("Waited for {} timesteps!\n setting graping...".format(action))

        #graspState = self.getState(mode="grasp")
        #action = self.grasp(graspState)
        if self.test: print("...grasping set for {}".format(action)) 
        if self.test: print("Motion primitive initilized!\n... Opening fingers!")
        self._action_moveFingers(0)  # Open fingers
        for i in range(5): self.supervisor.step(self.TIME_STEP)
        if self.test: print("... Moving down!")
        self._action_moveTCP(0)  # Go down
        for i in range(5): self.supervisor.step(self.TIME_STEP)
        if self.test: print("... Closing fingers!")
        self._action_moveFingers(1)  # close fingers
        self.reward = -np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100
        if self.test: print("... Reward acquired: {}".format(self.reward))
        for i in range(5): self.supervisor.step(self.TIME_STEP)
        if self.test: print("... Moving up!")
        self._action_moveTCP(1)  # Go up
        if self.test: print("... Waiting for can to slip out!")
        for i in range(100):
            #self.goal_node.resetPhysics()
            self.supervisor.step(self.TIME_STEP)
        self.counter = self.counter + 1
        self.reward = 100 if np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100 < 10 else self.reward
        #self.reward = -np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100
        #self.reward = 1 if np.linalg.norm(np.array(self.goal_pos.getSFVec3f())-np.array(self.tcp.getPosition()))*100 < 10 else self.reward
        if self.test: print("Get new state")
        state = self.getState()
        
        if self.test: print("REWARD: ", self.reward)
        if self.reward == 1:
            print("\nSuccess\n")
        if (self.counter >= self.timeout) or self.bugged:
            self.done = True

        if self.test: print("STATE: ", state)
        self.total_rewards = self.total_rewards + 1 if self.reward >= 0 else 0
        if self.done:
            self.saveEpisode(str(round(self.total_rewards)) + ";")
            self.plot_learning_curve()
        #self.supervisor.step(self.TIME_STEP)
        return [state, float(self.reward), self.done, {}]


    def grasp(self, state):
        stateGrasping = T.tensor([state]).to(self.grasping_DQN.device)
        actions = self.grasping_DQN.forward(stateGrasping)
        action = T.argmax(actions).item()  # Take the action with the highest Q-score
        self.motors[-1].setPosition(float(self.grasp_actions[action]))
        while not (self.sensors[-1].getValue() - m.radians(float(self.grasp_actions[action]))) < 0.01: self.supervisor.step(self.TIME_STEP)
        #print(m.radians(float(self.grasp_actions[action])), self.sensors[-1].getValue())
        return action

    def gotStuck(self):
        finger1_pos = self.finger1.getField("translation").getSFVec3f()
        finger2_pos = self.finger2.getField("translation").getSFVec3f()
        #print("FINGERS: " ,np.linalg.norm(np.array(finger1_pos)-np.array(finger2_pos)))
        #return False
        if np.linalg.norm(np.array(finger1_pos)-np.array(finger2_pos)) > 1:
            print("\nBugged out!\n")
            self.bugged = True
            return True


    def _getSensors(self):
        for i in range(len(self.touch_sensors)):
            self.touch_sensors[i] = self.supervisor.getDevice("touch_sensor"+str(i+1))
            self.touch_sensors[i].enable(self.TIME_STEP)


    def _getMotors(self):
        for i in range(len(self.joint_names)):
            self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)


    def _setTarget(self):
        #rotation = random.choice(self.rotations)
        z = random.uniform(0.4 - self.vdist/2, 0.4 - self.vdist/3) if random.randint(0, 1) else random.uniform(0.4 + self.vdist/2, 0.4 + self.vdist/3)

        #translation = [-0.01, 0.84, 0.4]
        x = random.uniform(0.5, 1.5)
        #translation = [x, 0.84, z]
        translation = [x, 0.84, 0.4]
        
        #self.goal_rot.setSFRotation(rotation)
        self.goal_rot.setSFRotation([2.7661199999894182e-06, -2.397449999990828e-09, 0.9999999999961744, 1.5708])
        
        self.goal_pos.setSFVec3f(translation)
        self.supervisor.step(self.TIME_STEP)
        self.goal_node.resetPhysics()

        #self.conveyor_speed = random.choice([0.3, 0.4, 0.5])
        #self.conveyor_speed_field.setSFFloat(self.conveyor_speed)
        

    def render(self, mode='human'):
        pass


    def saveEpisode(self, reward):
        with open(os.path.join(self.path, "documentation.txt"), 'a') as f:
            f.write(reward)


    def getState(self, mode="default"):
        x_dist = abs(self.goal_pos.getSFVec3f()[0] - self.tcp.getPosition()[0])*100
        y_dist = abs(self.goal_pos.getSFVec3f()[2] - self.tcp.getPosition()[2])*100
        state = []
        state.append(x_dist)
        if mode == "grasp":
            state.append(y_dist)
            state.append(self.sensors[-1].getValue())  # Get joint angle
            state += self.goal_rot.getSFRotation()
        #elif mode == "default":
        #    state.append(self.conveyor_speed)
        state = [float(s) for s in state]
        return state

    def _action_moveFingers(self, mode=0):
        # 0 for open state, 1 for closed state
        if mode == 0:
            self.finger_state = 0
            pos = [0.04, 0.04]
        elif mode == 1:
            self.finger_state = 1
            pos = [0.015, 0.015]
        self.fingers[0].setPosition(pos[0])
        self.fingers[1].setPosition(pos[1])
        self._util_positionCheck(pos, self.sensor_fingers, 0.02)

    def _action_moveTCP(self, mode):
        pose = self.down_pose
        if mode == 0:
            pose = self.down_pose
            self.movement_state = 0
        elif mode == 1:
            self.movement_state = 1
            pose = self.up_pose
        [self.motors[i].setPosition(m.radians(pose[i])) for i in range(len(pose))]
        self._util_positionCheck(pose, self.sensors, 0.05)

    def _util_positionCheck(self, pos, sensors, limit = 0.1):
        if len(pos):
            counter = 0
            prev_pose = 0
            pose = 0
            while not all([abs(sensors[i].getValue() - m.radians(pos[i])) < limit for i in range(len(pos))]):
                pose = round(sum([abs(sensors[i].getValue() - m.radians(pos[i])) for i in range(len(pos))]),2)
                counter = counter + 1 if pose == prev_pose else 0
                if counter >= 1: break
                self.supervisor.step(self.TIME_STEP)
                prev_pose = pose
                if self.gotStuck(): break
        return False


    def _util_readRotationFile(self, file):
        rotationFile = open(file, 'r')
        rotationFileLines = rotationFile.readlines()
        rotations = []
        # Strips the newline character
        for line in rotationFileLines:
            l = line.strip()
            numbers = l[1:-1].split(', ')
            tempList = [float(num) for num in numbers]
            rotations.append(tempList) 
        return rotations


    def plot_learning_curve(self):
        self.plot_rewards.append(self.total_rewards)
        x = [i+1 for i in range(len(self.plot_rewards))]
        running_avg = np.zeros(len(self.plot_rewards))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(self.plot_rewards[max(0, i-self.plt_avg):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous '+ str(self.plt_avg) + ' scores')
        plt.savefig(self.figure_file)

