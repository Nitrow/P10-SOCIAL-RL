#!/usr/bin/env python3.8

import gym
import numpy as np
import random
#from agent_dqn import Agent
#from utils import plot_learning_curve
import numpy as np
from datetime import datetime
import math as m
import shutil
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.spatial.transform import Rotation as R

from controller import Robot, Motor, Supervisor, Connector

class DQN(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/dqn'):
		super(DQN, self).__init__()
		self.name = name
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
		self.checkpoint_dir = "models/"#os.getcwd().split("/P10-XRL/")[0] + "/P10-XRL/GymEnvironments/P10-RL-LvL3-Grasping-Combined/P10_DRL_Lvl3_Grasping_Combined/envs/" + chkpt_dir
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


class Environment():
	def __init__(self):
		self.supervisor = Supervisor()
		self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
		self.robot_node = self.supervisor.getFromDef("UR3")
		self.conveyor = self.supervisor.getFromDef("CONVEYOR")
		self.finger1 = self.supervisor.getFromDef("FINGER1")
		self.finger2 = self.supervisor.getFromDef("FINGER2")
		self.goal_pos = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("translation")
		self.goal_rot = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("rotation")
		self.goal_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN")
		self.fingers = [self.supervisor.getDevice('right_finger'), self.supervisor.getDevice('left_finger')]
		self.sensor_fingers = [self.supervisor.getDevice('right_finger_sensor'), self.supervisor.getDevice('left_finger_sensor')]
		self.sensor_fingers[0].enable(self.TIME_STEP)
		self.sensor_fingers[1].enable(self.TIME_STEP)
		self.tcp = self.supervisor.getFromDef("TCP")
		self.tcp_pos = self.supervisor.getFromDef("TCP").getField("translation")
		self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
		self.motors = [0] * len(self.joint_names)
		self.sensors = [0] * len(self.joint_names)
		self.waitingTs = 3
		self.graspActions = [m.radians(10*i) for i in range(18)]
		self.successes = 0
		self.counter = 0
		self.timeout = 100
		self.done = False
		self.bugged = False
		self.poses =   {"up-close-left" : [-14.22, -115.34, -56.86, -97.83, 89.28],
						"up-close-mid" : [18.62, -100.45, -76.74, -93.45, 89.47],
						"up-close-right" : [46.16, -115.86, -55.7, -99.42, 89.92],
						"up-far-left" : [-12.4, -134.6, -24.56, -110.89, 89.31],
						"up-far-mid" : [15.87, -116.57, -54.78, -99.24, 89.47],
						"up-far-right" : [40.6, -135.69, -22.17, -113.07, 89.84],
						"down-close-left" : [-14.23, -124.83, -89.6, -55.58, 89.34],
						"down-close-mid" : [18.58, -113.75, -108.23, -48.63, 89.51],
						"down-close-right" : [46.16, -124.96, -88.91, -57.09, 89.97],
						"down-far-left" : [-12.39, -136.25, -68.44, -65.37, 89.38],
						"down-far-mid" : [15.86, -125.56, -88.06, -56.98, 89.53],
						"down-far-right" : [40.64, -136.44, -67.7, -66.78, 89.91]}
		self.up_pose = [16.63, -111.19, -63.15, -96.24, 89.47]
		self.down_pose = [16.60, -121.69, -94.63, -54.27, 89.52]
		self.rotations = self._util_readRotationFile('rotations.txt')

	def get_state(self):
		state = {"Grasping" : [], "Timing" : []}
		x_dist = abs(self.goal_pos.getSFVec3f()[0] - self.tcp.getPosition()[0])*100

		y_dist = abs(self.goal_pos.getSFVec3f()[2] - self.tcp.getPosition()[2])*100
		#state["Grasping"] += self.tcp.getOrientation()
		#state["Grasping"] += self.goal_node.getOrientation()
		state["Grasping"] += [round(x) for x in R.from_matrix(np.array(self.tcp.getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]  # Get joint angle
		state["Grasping"] += [round(x) for x in R.from_matrix(np.array(self.goal_node.getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]

		#state["Grasping"].append(x_dist)
		#state["Grasping"].append(y_dist)
		state["Grasping"] = [float(s) for s in state["Grasping"]]

		state["Timing"].append(float(x_dist))
			#state["Timing"] = [float(s) for s in state["Timing"]]

		return state

	def reset(self):
		self.supervisor.simulationReset()
		self.conveyor.restartController()
		self.supervisor.step(self.TIME_STEP)  
		self.successes = 0
		self.done = False
		self.bugged = False
		self.counter = 0
		self._getMotors()
		self.finger1.resetPhysics()
		self.finger2.resetPhysics()
		self.goal_node.resetPhysics()
		self._setTarget()
		#self.supervisor.step(self.TIME_STEP)  
		return self.get_state()

	def step(self, actions):
		graspAngle, waitingTime = actions
		# Resetting-utility function
		self._getMotors()
		self.finger1.resetPhysics()
		self.finger2.resetPhysics()
		self.motors[-1].setPosition(float(self.graspActions[graspAngle]))
		# Actual actions
		for i in range(waitingTime):
			self.supervisor.step(self.TIME_STEP)
		self._action_moveFingers(0)
		for i in range(self.waitingTs): self.supervisor.step(self.TIME_STEP)
		self._action_moveTCP(0)
		for i in range(self.waitingTs): self.supervisor.step(self.TIME_STEP)
		self._action_moveFingers(1)
		for i in range(self.waitingTs): self.supervisor.step(self.TIME_STEP)
		self._action_moveTCP(1)
		for i in range(100):
			self.supervisor.step(self.TIME_STEP)
		point = 1 if np.linalg.norm(np.array(self.goal_node.getField("translation").getSFVec3f())-np.array(self.tcp.getPosition()))*100 < 10 else 0
		self._setTarget()
		self.successes += point
		self.counter += + 1
		if self.counter >= self.timeout or self.bugged:
			if self.bugged: self.successes = self.successes/self.counter * 100
			self.done = True
			print(self.successes)
		return self.get_state()

	def _getMotors(self):
		for i in range(len(self.joint_names)):
			self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
			self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
			self.sensors[i].enable(self.TIME_STEP)

	def _action_moveFingers(self, mode=0):
		# 0 for open state, 1 for closed state
		if mode == 0: pos = [0.04, 0.04]
		elif mode == 1: pos = [0.015, 0.015]
		self.fingers[0].setPosition(pos[0])
		self.fingers[1].setPosition(pos[1])
		self._util_positionCheck(pos, self.sensor_fingers, 0.02)

	def _action_moveTCP(self, mode):
		pose = self.poses["down-close-left"]
		if mode == 0:
			pose = self.poses["down-close-right"]
			#pose = self.down_pose
			self.movement_state = 0
		elif mode == 1:
			self.movement_state = 1
			pose = self.poses["up-close-right"]
			#pose = self.up_pose
		[self.motors[i].setPosition(m.radians(pose[i])) for i in range(len(pose))]
		self._util_positionCheck(pose, self.sensors, 0.05)


	def gotStuck(self):
		finger1_pos = self.finger1.getField("translation").getSFVec3f()
		finger2_pos = self.finger2.getField("translation").getSFVec3f()
		#print("FINGERS: " ,np.linalg.norm(np.array(finger1_pos)-np.array(finger2_pos)))
		#return False
		if np.linalg.norm(np.array(finger1_pos)-np.array(finger2_pos)) > 1:
			print("\nBugged out!\n")
			self.bugged = True
			return True


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
			rotations.append([float(num) for num in numbers]) 
		return rotations

	def _setTarget(self):
		rotation = random.choice(self.rotations)
		#translation = [random.uniform(0.2, 1.2), 0.84, 0.4]
		#translation = [-0.21, 0.84, 0.36]
		#translation = [-0.01, 0.84, 0.4]
		translation = [0.21, 0.84, 0.42]
		
		self.goal_node.getField("rotation").setSFRotation(rotation)
		#self.goal_node.getField("rotation").setSFRotation([0, 0, 1, 1.5708])
		self.goal_node.getField("translation").setSFVec3f(translation)
		self.supervisor.step(self.TIME_STEP)
		self.goal_node.resetPhysics()


class Agent():
	def __init__(self):
		self.grasping_DQN = DQN(0, n_actions=18, input_dims=[6], fc1_dims=128, fc2_dims=128, name="DQN_Grasping", chkpt_dir='models')
		self.grasping_DQN.load_checkpoint()
		self.timing_DQN = DQN(0, n_actions=100, input_dims=[1], fc1_dims=1024, fc2_dims=1024, name="DQN_Timing", chkpt_dir='models')
		self.timing_DQN.load_checkpoint()

	def choose_action(self, state):
		# Get grasping
		stateGrasping = T.tensor([state["Grasping"]]).to(self.grasping_DQN.device)
		graspActions = self.grasping_DQN.forward(stateGrasping)
		graspAction = T.argmax(graspActions).item()
		# Get timing
		stateTiming = T.tensor([state["Timing"]]).to(self.timing_DQN.device)
		timingActions = self.timing_DQN.forward(stateTiming)
		timingAction = T.argmax(timingActions).item()

		return [graspAction, timingAction]



if __name__ == '__main__':
	# Initialize game variables
	n_games = 100

	env = Environment()
	agent = Agent()
	#shutil.copy(env.own_path, env.path)
	total_points = 0
	bugs = 0
	for i in range(n_games):
	# Start a new game
		observation = env.reset()

		while not env.done:
		# Take an action
			actions = agent.choose_action(observation)
			observation = env.step(actions)
			#print(env.get_state())
		total_points += env.successes
		if env.bugged and env.successes == 0: bugs += 1
	print("Avg successes out of hundred games: {}%".format(float(total_points/(n_games-bugs))))
	#print("Action: {}\t Observation: {}\t Reward: {}".format(action, observation, reward))