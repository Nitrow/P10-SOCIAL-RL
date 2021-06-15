# from controller import Robot, Supervisor

# supervisor = Supervisor()



# robot = supervisor.getFromDef("UR3")

# can = supervisor.getFromDef("GREEN_ROTATED_CAN")
# conveyor = supervisor.getFromDef("CONVEYOR")

# timestep = int(supervisor.getBasicTimeStep())


# conv_speed = 0.5
# conveyor.getField("speed").setSFFloat(conv_speed)
# ts = 0
# startpos = can.getField("translation").getSFVec3f()
# prev_moves = 0
# while supervisor.step(timestep) != -1:
    # moves = startpos[0]-can.getField("translation").getSFVec3f()[0]
    # can_diff = round((prev_moves - moves)*100,4)
    # print("In {} timesteps the can moved {} cm\t speed {} cm/ts".format(ts, round(moves*100,2), can_diff))

    # ts += 1
    # prev_moves = moves

    # 0.32 cm with 0.1 m/s
    
    
import pyikfastur3
target_translation = [0.5, 0.5, 0.5]
target_rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
# Calculate inverse kinematics
positions = pyikfast.inverse(target_translation, target_rotation)
print(positions)

