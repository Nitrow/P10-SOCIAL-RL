#!/usr/bin/env python3.8

import gym
import numpy as np
import random
import cv2

from agent_dqn import DQN_Agent
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

from controller import Robot, Motor, Supervisor, Connector, Camera, Display

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
		self.root_children = self.supervisor.getRoot().getField("children")
		self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
		self.robot_node = self.supervisor.getFromDef("UR3")
		self.conveyor = self.supervisor.getFromDef("CONVEYOR")
		self.finger1 = self.supervisor.getFromDef("FINGER1")
		self.finger2 = self.supervisor.getFromDef("FINGER2")
		#self.goal_pos = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("translation")
		#self.goal_rot = self.supervisor.getFromDef("GREEN_ROTATED_CAN").getField("rotation")
		#self.goal_node = self.supervisor.getFromDef("GREEN_ROTATED_CAN")
		self.fingers = [self.supervisor.getDevice('right_finger'), self.supervisor.getDevice('left_finger')]
		self.sensor_fingers = [self.supervisor.getDevice('right_finger_sensor'), self.supervisor.getDevice('left_finger_sensor')]
		self.sensor_fingers[0].enable(self.TIME_STEP)
		self.sensor_fingers[1].enable(self.TIME_STEP)
		self.tcp = self.supervisor.getFromDef("TCP")
		self.tcp_pos = self.supervisor.getFromDef("TCP").getField("translation")
		self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
		# Display information
		self.camera = Camera("camera")
		self.camera.enable(self.TIME_STEP)
		self.camera.recognitionEnable(self.TIME_STEP)
		self.cam_width = self.camera.getWidth()
		self.cam_height = self.camera.getHeight()
		
		self.display_explanation = self.supervisor.getDevice("display_explanation")

		self.colors =  {"yellow" : [0.309804, 0.913725, 1.0],
						"red" : [0.0, 0.0, 1.0],
						"green" : [0.0, 1.0, 0.0]}
		# Motors
		self.color_codes = {"red" : 0, "green": 1, "yellow" : 2}
		self.motors = [0] * len(self.joint_names)
		self.sensors = [0] * len(self.joint_names)
		self.waitingTs = 3
		self.graspActions = [m.radians(10*i) for i in range(18)]
		self.successes = 0
		self.counter = 0
		self.timesteps = 0
		self.timeout = 10
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
						"down-far-right" : [40.64, -136.44, -67.7, -66.78, 89.91],
						"green-bin" : [120, -125, -30, -115, 90],
						"red-bin"   : [-107.77, -119, -46.48, -102.99, 91]}
		self.pose_code = {"left" : -0.21, "mid" : 0, "right" : 0.21, "far" : 0.42, "close" : 0.36}
		self.up_pose = [16.63, -111.19, -63.15, -96.24, 89.47]
		self.down_pose = [16.60, -121.69, -94.63, -54.27, 89.52]
		self.rotations = self._util_readRotationFile('rotations.txt')
		self.timing_dist = 0.25
		# Can registry
		self.cans = {}
		self.select_can_n = 5
		self.actions = self.select_can_n + 1
		self.selected_cans = []
		self.candidates = []
		self.delete_cans = []
		self.chosen_can = 0
		self.crate_pos_img = {"RED_ROBOT_CRATE" : [], "GREEN_ROBOT_CRATE" : []}
		self.chosen_id = 0
		self.choose_new_can = False
		self.success = False
		self.tryGetCratePos()
		# DRL related variables
		self.state = {"Grasping" : [], "Timing" : [], "Selecting" : []}
		self.path = "training"
		self.lvl3_agent = Agent()
		self.state_shape = self.select_can_n * 3
		self.r_index = -1
		self.r_dist = -1
		self.r_color = -1
		self.r_success = 1
		self.fullState = False


	def reset(self):
		self.cleanup()
		self.supervisor.simulationReset()
		self.conveyor.restartController()
		self.selected_cans = []
		self.candidates = []
		self.delete_cans = []
		self.chosen_can = 0
		self.cans = {}
		self.successes = 0
		self.done = False
		self.bugged = False
		self.success = False
		self.counter = 0
		self.fullState = False
		self.timesteps = 0
		self.chosen_id = 0
		self._getMotors()
		self.finger1.resetPhysics()
		self.finger2.resetPhysics()
		self.crate_pos_img = {"RED_ROBOT_CRATE" : [], "GREEN_ROBOT_CRATE" : []}
		self.tryGetCratePos()
		while len(self.selected_cans) == 0: self.move_time()
		return self.get_state()


	def get_reward():
		pass


	def move_time(self):
		self.timesteps += 1
		#print(self.chosen_can)
		self._drawImage()
		self.canManager()
		self.get_state()
		#print(self.chosen_can)
		#sel = self.get_state()["Selecting"]
		#print(self.selected_cans)
		#print(sel, len(sel))
		self.supervisor.step(self.TIME_STEP)


	def get_state_low(self):
		self.state["Grasping"] = []
		self.state["Timing"] = []
		tcp_rot = [round(x) for x in R.from_matrix(np.array(self.tcp.getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]  # Get joint angle
		can_rot = [round(x) for x in R.from_matrix(np.array(self.chosen_can.getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]
		self.state["Grasping"] += tcp_rot
		self.state["Grasping"] += can_rot
		self.state["Grasping"] = [float(s) for s in self.state["Grasping"]]
		x_dist = abs(self.chosen_can.getPosition()[0] - self.tcp.getPosition()[0])*100
		self.state["Timing"].append(float(x_dist))

	def get_state(self):
		self.state["Selecting"]  = []
		n_candidates = 0
		self.selected_cans = []
		for candidate in self.candidates:
			n_candidates += 1
			if n_candidates > self.select_can_n: break
			can_rot = [round(x) for x in R.from_matrix(np.array(self.supervisor.getFromId(candidate).getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]
			self.state["Selecting"].append(self.color_codes[self.cans[candidate]["color"]])
			self.state["Selecting"].append(self.cans[candidate]["position"][0])
			self.state["Selecting"].append(can_rot[2])
			self.selected_cans.append(candidate)
		
		self.state["Selecting"] += ((self.select_can_n * 3) - len(self.state["Selecting"])) * [0]
		self.state["Selecting"] = [float(s) for s in self.state["Selecting"]]
		return self.state


	def step(self, action):
		self.fullState = False
		try:
			self.chosen_can = self.supervisor.getFromId(self.selected_cans[action])
			self.fullState = True
		except IndexError:
			self.chosen_can = 0
			self.fullState = False
			return [self.get_state(), self.r_index]
		#print(self.chosen_can)
		self.chosen_id = self.chosen_can.getId()
		self.choose_new_can = False

		self.get_state_low()
		graspAngle, waitingTime = self.lvl3_agent.choose_action(self.state)
		self.motors[-1].setPosition(float(self.graspActions[graspAngle]))
		reward = 0
		
		# Move to pose
		self.move_time()
		if self.choose_new_can: return [self.get_state(), reward]
		chosen_color = self.cans[self.chosen_id]["color"]
		if chosen_color == "yellow": return [self.get_state(), self.r_color]
		x = round(self.chosen_can.getPosition()[0],2)
		y = round(self.chosen_can.getPosition()[2],2)
		if self.pose_code["right"] + self.timing_dist < x : x_pos = "right"
		elif self.pose_code["mid"] + self.timing_dist < x : x_pos = "mid"
		elif self.pose_code["left"] + self.timing_dist < x : x_pos = "left"
		else: return [self.get_state(), self.r_dist]
		
		
		
		
		y_pos = "far" if y > (self.pose_code["close"]+self.pose_code["far"])/2 else "close"
		self.down_pose = self.poses["down-" + y_pos + "-" + x_pos]
		self.up_pose = self.poses["up-" + y_pos + "-" + x_pos]
		self._action_moveTCP(1)
		self._action_moveFingers(0)

		#0.0128 + 8
		for i in range(max(round(abs(x-self.tcp.getPosition()[0])/0.0128 - 25),0)): self.move_time()
		# Move down
		self._action_moveTCP(0)
		for i in range(self.waitingTs): self.move_time()
		# Grasp
		self._action_moveFingers(1)
		for i in range(self.waitingTs): self.move_time()
		self._action_moveTCP(1)
		# Put to bin
		self.up_pose = self.poses[chosen_color + "-bin"]
		for i in range(self.waitingTs): self.move_time()
		self._action_moveTCP(1)
		for i in range(self.waitingTs): self.move_time()
		self._action_moveFingers(0)
		for i in range(self.waitingTs): self.move_time()
		self.up_pose = self.poses["up-close-mid"]
		self._action_moveTCP(1)
		for i in range(self.waitingTs): self.move_time()
		

		# Get reward
		if self.success:
			print("Success!")
			self.success = False
			reward = self.r_success
		self.counter += 1
		#print(self.counter)
		if self.counter >= self.timeout or self.bugged:
			if self.bugged: self.successes = self.successes/self.counter * 100
			self.done = True
			#print(self.successes)
		return [self.get_state(), reward]

	def _drawImage(self):
		cameraData = self.camera.getImage()
		if not cameraData: return
		image = np.frombuffer(cameraData, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
		chosen_id = self.chosen_can.getId() if type(self.chosen_can) != type(1) else 0
		for obj in self.camera.getRecognitionObjects():
			obj_id = obj.get_id()
			if obj_id not in self.cans: continue
			if chosen_id == obj_id:
				size = np.array(obj.get_size_on_image()) + np.array([10, 10])
				start_point = np.array(obj.get_position_on_image()) - (size / 2) 
				start_point =  np.array([int(n) for n in start_point])
				end_point = start_point + size
				thickness = 2
				col = self.cans[obj_id]["color"]
				color = self.colors[self.cans[obj_id]["color"]]
				color = np.rint(np.array(color)*255)
				color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
				font = cv2.FONT_HERSHEY_SIMPLEX
				image = cv2.rectangle(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
				# Draw arrows:
				if col != "yellow":
					end_point = self.crate_pos_img[col.upper() + "_ROBOT_CRATE"]
					start_point = np.array([int(n) for n in obj.get_position_on_image()])
					image = cv2.arrowedLine(image, tuple(start_point), tuple(end_point), tuple(color), thickness)
		cv2.imwrite('resources/tmp.jpg', image)
		ir = self.display_explanation.imageLoad('resources/tmp.jpg')
		self.display_explanation.imagePaste(ir, 0, 0, False)
		self.display_explanation.imageDelete(ir)


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
		pose = self.up_pose
		if mode == 0: pose = self.down_pose
		elif mode == 1:	pose = self.up_pose
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
				self.move_time()
				prev_pose = pose
				if self.gotStuck(): break
		return False

	def _util_readRotationFile(self, file):
		rotationFile = open('resources/' + file, 'r')
		rotationFileLines = rotationFile.readlines()
		rotations = []
		# Strips the newline character
		for line in rotationFileLines:
			l = line.strip()
			numbers = l[1:-1].split(', ')
			rotations.append([float(num) for num in numbers]) 
		return rotations

	# def _setTarget(self):
	# 	rotation = random.choice(self.rotations)
	# 	#translation = [random.uniform(0.2, 1.2), 0.84, 0.4]
	# 	#translation = [-0.21, 0.84, 0.36]
	# 	#translation = [-0.01, 0.84, 0.4]
	# 	translation = [0.21, 0.84, 0.42]
		
	# 	self.goal_node.getField("rotation").setSFRotation(rotation)
	# 	#self.goal_node.getField("rotation").setSFRotation([0, 0, 1, 1.5708])
	# 	self.goal_node.getField("translation").setSFVec3f(translation)
	# 	self.move_time()
	# 	self.goal_node.resetPhysics()


	def tryGetCratePos(self):
		for obj in self.camera.getRecognitionObjects():
			obj_node = self.supervisor.getFromId(obj.get_id())
			if obj_node:
				obj_name = obj_node.getDef()
				if obj_name == "RED_ROBOT_CRATE":
					self.crate_pos_img["RED_ROBOT_CRATE"] = obj.get_position_on_image()
				elif obj_name == "GREEN_ROBOT_CRATE":
					self.crate_pos_img["GREEN_ROBOT_CRATE"] = obj.get_position_on_image()



	def generateCans(self):
		can_color = random.choice(["green", "yellow", "red"])
		self.root_children.importMFNode(-1, 'resources/' + can_color + ".wbo")
		self.root_children.getMFNode(-1).getField("translation").setSFVec3f([2.7, 0.87, random.uniform(0.36,0.42)])
		self.root_children.getMFNode(-1).getField("rotation").setSFRotation(random.choice(self.rotations))
		self.cans[self.root_children.getMFNode(-1).getId()] = {"color" : can_color}


	def removeCans(self):
		for can_id in self.delete_cans:
			self.supervisor.getFromId(can_id).remove()
			del self.cans[can_id]
			self.candidates.remove(can_id)


	def maintainCans(self):
		self.candidates = []
		self.delete_cans = []
		for can_id, can_details in self.cans.items():
			can_node = self.supervisor.getFromId(can_id)
			#pos = self.supervisor.getFromId(can_id).getPosition()
			#rot = self.supervisor.getFromId(can_id).getOrientation()
			self.cans[can_id]["position"] = [round(x,2) for x in can_node.getPosition()]
			self.cans[can_id]["orientation"] = [round(x,2) for x in can_node.getOrientation()]

			if self.cans[can_id]["position"][0] < 1 : self.candidates.append(can_id)
			
			can_pos = self.supervisor.getFromId(can_id).getPosition()
			if can_pos[0] < -0.6 or (can_pos[2] < 0.34 and can_pos[1] < 0.7):
				if can_pos[2] < 0.34 and can_pos[1] < 0.7:
					if can_id == self.chosen_id: self.success  = True
				if can_id == self.chosen_id: self.choose_new_can = True
				self.delete_cans.append(can_id)


	def canManager(self):
		# Can birth
		if self.timesteps%10 == 0 and random.choice([0,1,2]) == 2: self.generateCans()
		# 
		self.maintainCans()
		#print(self.candidates)
		# Can death
		self.removeCans()


	def cleanup(self):
		root_children_n = self.root_children.getCount()
		cans = []
		for n in range(root_children_n):
			if "CAN" in self.root_children.getMFNode(n).getDef():
				cans.append(self.root_children.getMFNode(n))
		for can in cans: can.remove()


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
	batchsize = 128
	lr = 0.03
	neurons = 128
	env = Environment()

	agent = DQN_Agent(gamma=0, epsilon=1.0, batch_size=batchsize, n_actions=env.actions, eps_end=0.01, input_dims=[env.state_shape], lr=lr, chkpt_dir=env.path, fc1_dims=neurons, fc2_dims=neurons) 
	#shutil.copy(env.own_path, env.path)
	total_points = 0
	bugs = 0

	reward_history = []
	best_score = -999
	for i in range(n_games):
	# Start a new game
		observation = env.reset()
		score = 0
		while not env.done:
		# Take an action
			action = agent.choose_action(observation["Selecting"])
			observation_, reward = env.step(action)
			#print(action)
			agent.remember(observation["Selecting"], action, reward, observation_["Selecting"], env.done)
			#print(observation["Selecting"])
			agent.learn()
			#print(len(observation["Selecting"]))
			#print(env.get_state())
			score += reward
		reward_history.append(score)
		avg_score = np.mean(reward_history[-100:])
		total_points += env.successes
		if env.bugged and env.successes == 0: bugs += 1
		if avg_score >= best_score:
			agent.save_models()
		print('Episode {}: score {} trailing 100 games avg {}'.format(i, score, avg_score))
	print("Avg successes out of hundred games: {}%".format(float(total_points/(n_games-bugs))))
	#print("Action: {}\t Observation: {}\t Reward: {}".format(action, observation, reward))