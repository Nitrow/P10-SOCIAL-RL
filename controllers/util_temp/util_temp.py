from controller import Robot, Supervisor
from ikpy.chain import Chain
import itertools
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R
import random

rotationFile = open("rotations.txt", 'r')
rotationFileLines = rotationFile.readlines()
rotations = []
# Strips the newline character
for line in rotationFileLines:
    l = line.strip()
    numbers = l[1:-1].split(', ')
    tempList = [float(num) for num in numbers]
    rotations.append(tempList) 

supervisor = Supervisor()

IKPY_MAX_ITERATIONS = 4
IKchain = Chain.from_urdf_file("UR3.urdf")


robot = supervisor.getFromDef("UR3")

can = supervisor.getFromDef("GREEN_ROTATED_CAN")
conveyor = supervisor.getFromDef("CONVEYOR")
tcp = supervisor.getFromDef("TCP")
#tcp = supervisor.getFromDef("UR_END")

timestep = int(supervisor.getBasicTimeStep())

# for i in [0, 6]:
#     IKchain.active_links_mask[0] = False

motors = []
sensors = []
for link in IKchain.links:
    if 'joint' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(3.14)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        sensors.append(position_sensor)
        motors.append(motor)

#initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]

# up: xyz: 0.4, 0, 0.32
x,y,z = 0.01, 0.15, 0.41
ikResults = IKchain.inverse_kinematics([x, y, z], max_iter=IKPY_MAX_ITERATIONS)#, initial_position=initial_position)
# down: xyz: 0.4, 0, 0.15

conv_speed = 0
conveyor.getField("speed").setSFFloat(conv_speed)
ts = 0
startpos = can.getField("translation").getSFVec3f()
prev_moves = 0
k = []
#motors[-1].setVelocity()
while supervisor.step(timestep) != -1:
    ts += 1
    # moves = startpos[0]-can.getField("translation").getSFVec3f()[0]
    # can_diff = round((prev_moves - moves)*100,4)
    # print("In {} timesteps the can moved {} cm\t speed {} cm/ts".format(ts, round(moves*100,2), can_diff))
    pos = [round(x,2) for x in tcp.getPosition()]
    pos[1] -= 0.85
	#print(pos)
    if ts%100 == 0:
        can.getField("rotation").setSFRotation(random.choice(rotations))
        can_rot = [round(x) for x in R.from_matrix(np.array(can.getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]
        tcp_rot = [round(x) for x in R.from_matrix(np.array(tcp.getOrientation()).reshape(3,3)).as_euler('ZYX', degrees=True)]
    	#r = R.from_matrix(rot)
    	#rot = [round(x) for x in R.from_matrix(rot).as_euler('zyx', degrees=True)]
    	#rot = r.as_rotvec()
        can_angle = can_rot[2] if can_rot[2] > 0 else can_rot[2] + 180
        tcp_angle = tcp_rot[2] if tcp_rot[2] > 0 else tcp_rot[2] + 180
        #print(x)
        p = sensors[-1].getValue()
        #print(p, tcp_angle)
        diff = m.radians(can_angle - tcp_angle)
        print(can_angle - tcp_angle)
        motors[-1].setPosition(p - diff)
	# print(can.getOrientation())
    # prev_moves = moves

   

