#!/usr/bin/env python3.8
import numpy as np
from controller import Robot, Supervisor, Lidar
import ikpy
from ikpy.chain import Chain

class Environment():
    """
    Similarly to a standard RL setting, the environment provides the information to the agent.
    The environment can consist of but is not limited to:
    - Joint angles
    - Joint positions
    - Tool Center Point (TCP) position and orientation
    - Target Position
    """
    def __init__(self):
        """
        Initilializing the environment
        """
        # Define the time step
        self.TIME_STEP = 8
        # Initiate the supervisor and the robot
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getFromDef('UR3')
        self.tcp = self.supervisor.getFromId(855) # print(supervisor.getSelected().getId())
        #print(self.supervisor.getSelected().getId())
        # Get the robot position vector and rotation matrix
        self.robot_pos = np.array(self.robot.getPosition())
        self.robot_rot = np.transpose(np.array(self.robot.getOrientation()).reshape(3,3))
        
        # Variable for kinematics calculation
        self.ur3_chain = Chain.from_urdf_file("../../resources/robot2.urdf")
        
        # Set target position
        self.target_pos = []
        
        self.episode_over = False
        # Position of the TCP both from WeBots and from the kinematics equations
        
        self.pos_tcp_wb = [] # TCP position from WeBots
        self.pos_tcp_fk = [] # TCP position from kinematics
        
        self.step_iteration = 0
        self.joint_names = [ 'shoulder_pan_joint',
                            'shoulder_lift_joint',
                            'elbow_joint',
                            'wrist_1_joint',
                            'wrist_2_joint',
                            'wrist_3_joint']
                            
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.conveyor_node = self.supervisor.getFromDef('CONVEYOR')
        self.tv_node = self.supervisor.getFromDef('TV')
        
        self._getTarget()
        self._getmotors()
        self._getTCP()
                            
        
    def reset(self):
        """
        Resetting the environment
        """
        self.supervisor.simulationReset()
        self.conveyor_node.restartController()
        self.tv_node.restartController()
        
        
    def play_step(self, action):
        """
        Playing the action
        """
        self.step_iteration += 1
        self._execute(action)
        #self.supervisor.step(1)
        reward = self.calculate_reward()
        game_over = reward > 0.05
        return reward, game_over


    def calculate_reward(self):
        """
        Calculates the reward
        """
        if self.robot.getNumberOfContactPoints(True) > 0:
            return -500
        else:
            self._getTCP()
            self._getTarget()
            dist2target = np.linalg.norm(self.target_pos - self.pos_tcp_wb)
        
        return -dist2target


    def getState(self):
        # Get joint angles
        positions = np.array(self.ur3_chain.inverse_kinematics(self.pos_tcp_wb)[1:])
        # Get joint velocities
        velocities = np.array([self.motors[i].getVelocity() for i in range(len(self.motors))])
        # Get tcp position
        tcp_pos = np.array(self.pos_tcp_wb)
        # Get target position
        target_pos = np.array(self.target_pos)
        # Return assembled state
        return np.hstack((positions, velocities, tcp_pos, target_pos))


    def _getTarget(self):
        """
        Generates a target, or finds one by itself
        """
        #self.target_pos = self.supervisor.getFromDef("BEER").getPosition()
        self.target_pos = [0, 0, 0]
      
        
    def _execute(self, action):
        """
        Setting the motors
            action: a list of joint velocity commands
        """
        for i in range(len(action)):
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(float(action[i]))
    
    
    def _world2robotFrame(self, position):
        """
        Changes the position from the world frame to the robot frame
        """
        # Get the relative translation between the robot and the target
        target_position = np.subtract(np.array(position), self.robot_pos)
        # Matrix multiplication to get the target position relative to the robot
        target_position = np.dot(self.robot_rot, target_position)
        return target_position
        
        
    def _getmotors(self):
        """
        Initializes the motors and their sensors
        """
        for i in range(len(self.joint_names)):
            # Get motors
            self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
            self.motors[i].setPosition(float('inf'))
            
            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)


    def _getTCP(self):
        pass
        """
        Gets the TCP position both from Webots and from the kinematics
        """
        # Get the position of the TCP
        self.pos_tcp_wb = self.tcp.getPosition()
        # Transform it into the robot's frame
        self.pos_tcp_wb = self._world2robotFrame(self.pos_tcp_wb)