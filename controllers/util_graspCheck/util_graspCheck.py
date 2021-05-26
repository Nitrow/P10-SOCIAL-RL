"""util_goToPos controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Supervisor, Node
import math as m
import random
from scipy.spatial.transform import Rotation as R

def util_positionCheck(pos, sensors, limit = 0.1):
    if len(pos):
        counter = 0
        prev_pose = 0
        pose = 0
        #print("Target position at: {}, position is at {}".format(pos, [m.degrees(sens[i].getValue()) for i in range(len(pos))]))
        while not all([abs(sensors[i].getValue() - m.radians(pos[i])) < limit for i in range(len(pos))]):
            pose = round(sum([abs(sensors[i].getValue() - m.radians(pos[i])) for i in range(len(pos))]),2)
            counter = counter + 1 if pose == prev_pose else 0
            if counter >= 1: break
            supervisor.step(timestep)
            prev_pose = pose
    return False


def axisangle2euler(rotation):
	# YZX
	x,y,z,angle = rotation
	s = m.sin(angle)
	c = m.cos(angle)
	t = 1-c
	if ((x*y*t + z*s) > 0.998): # north pole singularity
		yaw = round(m.degrees(2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))
		pitch = round(m.degrees(m.pi/2))
		roll = round(m.degrees(0))
		#return [roll, pitch, yaw]

	elif ((x*y*t + z*s) < -0.998):
		yaw = round(m.degrees(-2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))
		pitch = round(m.degrees(-m.pi/2))
		roll = round(m.degrees(0))
		#return [roll, pitch, yaw]
	else:
		yaw = round(m.degrees(m.atan2(y*s - x*z, 1 - (y*y + z*z) * t)))
		pitch = round(m.degrees(m.asin(x * y * t + z * s)))
		roll = round(m.degrees(m.atan2(x * s - y * z * t, 1 - (x*x + z*z) * t)))
	yaw = yaw + 180 if yaw <= 0 else yaw
	pitch = pitch + 180 if pitch <= 0 else pitch
	roll = roll + 180 if roll <= 0 else roll
	return [roll, pitch, yaw]

# create the Robot instance.
supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")
f1 = supervisor.getFromDef("FINGER1")
f2 = supervisor.getFromDef("FINGER2")
random.seed(1)

rotationFile = open('rotations.txt', 'r')
rotationFileLines = rotationFile.readlines()
 
rotations = []
# Strips the newline character
for line in rotationFileLines:
	l = line.strip()
	numbers = l[1:-1].split(', ')
	tempList = [float(num) for num in numbers]
	rotations.append(tempList)

count = 0

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
joint_names = ['shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint']
motors = [0]*len(joint_names)
sensors = [0]*len(joint_names)

fingers = [supervisor.getDevice('right_finger'), supervisor.getDevice('left_finger')]
sensor_fingers = [supervisor.getDevice('right_finger_sensor'), supervisor.getDevice('left_finger_sensor')]
[sensor_fingers[i].enable(timestep) for i in range(len(sensor_fingers))]


can = supervisor.getFromDef("GREEN_ROTATED_CAN")
can_pos = can.getField("translation")
tcp = supervisor.getFromDef("TCP")


def moveFingers(fingers, mode="open"):

    if mode == "open": pose = [0.04, 0.04]
    elif mode == "close": pose = [0.015, 0.015]
    fingers[0].setPosition(pose[0])
    fingers[1].setPosition(pose[1])
    util_positionCheck(pose, sensor_fingers, 0.02)

def rotateCan():
	rotation = random.choice(rotations)
	translation = [-0.01, 0.84, 0.4]
	can.getField("rotation").setSFRotation(rotation)
	#print(rotation)
	#print("can rotation in radians: ",rotation,"   can rotation in angles: ", axisangle2euler(rotation))
	can_pos.setSFVec3f(translation)
	supervisor.step(timestep)
	can.resetPhysics()

for i in range(len(joint_names)):  
    motors[i] = supervisor.getDevice(joint_names[i])   
    
    sensors[i] = supervisor.getDevice(joint_names[i] + '_sensor')
    sensors[i].enable(timestep)
    #motors[i].setPosition(float('inf'))

up_pose = [16.63, -111.19, -63.15, -96.24, 89.47]
down_pose = [16.60, -121.69, -94.63, -54.27, 89.52]

#moveFingers(fingers, "open")
robot_connector = supervisor.getDevice("connector")
robot_connector.enablePresence(timestep)
#motors[-1].setPosition(float('inf'))
done = False

def stepTime():
	print("CAN angle: ", axisangle2euler(can.getField("rotation").getSFRotation()))
	supervisor.step(timestep)

while not done:
	#r = R.from_mrp(can.getField("rotation").getSFRotation())
	#print(r.as_mrp())
	#print(r.as_euler())
	moveFingers(fingers, "open")
	for i in range(5): stepTime()
	[motors[i].setPosition(m.radians(down_pose[i])) for i in range(len(down_pose))]
	util_positionCheck(down_pose, sensors, 0.05)
	for i in range(5): stepTime()
	moveFingers(fingers, "close")
	for i in range(5): stepTime()
	[motors[i].setPosition(m.radians(up_pose[i])) for i in range(len(up_pose))]
	util_positionCheck(up_pose, sensors, 0.05)
	#print("CAN: ", round(can_pos.getSFVec3f()[1],6)*100, "\tTCP: ", round(tcp.getPosition()[1], 6)*100)
	#print("CAN angle: ", axisangle2euler(can.getField("rotation").getSFRotation()))
	#print(axisangle2euler())
	for i in range(100): stepTime()
	rotateCan()
	can.resetPhysics()
	#[motors[i].setPosition(m.radians(up_pose[i])) for i in range(len(up_pose))]

# Enter here exit cleanup code.

