"""can_pose_recorder controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Supervisor

# create the Robot instance.
supervisor = Supervisor()
robot = supervisor.getFromDef("UR3")
can = supervisor.getFromDef("GREEN_ROTATED_CAN")

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
rotations = []
while supervisor.step(timestep) != -1:
    rotation = [round(x,3) for x in can.getField("rotation").getSFRotation()]
    if rotation not in rotations:
        rotations.append(rotation)
        f = open("rotations2.txt", "a")
        f.write(str(rotation)+'\n')
        f.close()
# Enter here exit cleanup code.
