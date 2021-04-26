from controller import Robot, Display


robot = Robot()
        
timeStep = 32
display1 = robot.getDevice('display')
print(display)
display.drawLine(0, 50, 10, 50)
while robot.step(timeStep) != -1:
    pass    