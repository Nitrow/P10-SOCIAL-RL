#!/usr/bin/env python3

from controller import Robot, Camera, Supervisor, Display, Connector, Keyboard
import numpy as np
import cv2
import random
import os
import math

#viewpoint:

# POS: 1.6931910127331369 1.781756528011428 5.350749519514143
# ROT: -0.9975109062294765 -0.05050270631532978 0.049208420093273746 0.06363818669844469
#condition = ["control", "all", "visual", "written"]
condition = "written"

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
display_explanation = supervisor.getDevice("display_explanation")
screen = 0
key = 0
setScreen = True
keyboard = Keyboard()
keyboard.enable(timestep)
root_children = supervisor.getRoot().getField("children")
importCans = True
deleteCans = True
canSelection = None
prevCan = None
can_ID = None
gravityField = root_children.getMFNode(0).getField("gravity")
viewNode = root_children.getMFNode(1)
gravityField.setSFFloat(0)
while supervisor.step(timestep) != -1:
	prevKey = key
	key=keyboard.getKey()
	if (key == 316 or key == 32) and prevKey != key: screen += 1
	#if key == 314 and prevKey != key: screen -= 1

	if screen == 0:
		display_explanation.setOpacity(1)
		display_explanation.setAlpha(1)
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		display_explanation.setFont("Lucida Console", 64, True)
		display_explanation.drawText("Welcome!", 300, 50)
		display_explanation.setFont("Lucida Console", 32, True)
		display_explanation.drawText("In this experiment, your task is to", 20, 160)
		display_explanation.drawText("sort different aluminium cans in col-", 20, 214)
		display_explanation.drawText("laboriation with an UR-3 manipulator.", 20, 266)
		display_explanation.drawText("There are three types of cans:", 20, 366)
		display_explanation.drawText("Yellow", 200, 466)
		display_explanation.drawText("Red", 500, 466)
		display_explanation.drawText("Green", 700, 466)
		if importCans:
			root_children.importMFNode(-1, "Red.wbo")
			root_children.importMFNode(-1, "Yellow.wbo")
			root_children.importMFNode(-1, "Green.wbo")
			importCans = False
			deleteCans = True

	if screen == 1:
		importCans = True
		if deleteCans:
			supervisor.getFromDef("INTRO-CAN-RED").remove()
			supervisor.getFromDef("INTRO-CAN-YELLOW").remove()
			#supervisor.getFromDef("INTRO-CAN-GREEN").remove()
			deleteCans = False
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		display_explanation.drawText("Your task is to sort the yellow and ", 20, 160)
		display_explanation.drawText("green cans. You can do it by clicking", 20, 214)
		display_explanation.drawText("first on the can, then on the crate", 20, 266)
		display_explanation.drawText("you wish to put it into.", 20, 316)
		display_explanation.drawText("Try it out with this can:", 20, 416)
		selection = supervisor.getSelected()
		selectionName = selection.getDef()

		if "CAN" in selectionName:
			canSelection = selection

		elif ("CRATE" in selectionName) and canSelection:
			new_position = selection.getField("translation").getSFVec3f()
			new_position[1] = 1.5
			canSelection.getField("translation").setSFVec3f(new_position)
			gravityField.setSFFloat(9.81)
			prevCan = canSelection
			canSelection = None
		if prevCan:
			if prevCan.getField("translation").getSFVec3f()[1] < 0.5:
				prevCan = None
				screen += 1

	if screen == 2:
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		display_explanation.drawText("The robot will also sort cans, that ", 20, 164)
		display_explanation.drawText("are either red or green.", 20, 214)
		display_explanation.drawText("The cans will arrive from the conve-", 20, 336)
		display_explanation.drawText("yor belt, and you'll gain scores for", 20, 386)
		display_explanation.drawText("sorting them correctly.", 20, 436)
		display_explanation.drawText("You get 1 point for each correct sort", 20, 526)
		display_explanation.drawText("and -1 point for each mistake or ", 20, 576)
		display_explanation.drawText("missed can.", 20, 626)

	# if screen == 3:
	# 	display_explanation.setColor(0xFFFFFF)
	# 	display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
	# 	display_explanation.setColor(0x000000)
	# 	x = 100
	# 	display_explanation.drawText("The robot is controlled by a specific", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("type of machine learning algorithm ", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("called reinforcement learning, where", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("the robot tries to reinforce positive", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("behavior, which in this case is", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("sorting the cans correctly.", 20, x)
	# 	x += 100
	# 	display_explanation.drawText("It is therefore preferred to let the ", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("robot sort as many cans correctly as ", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("possible, while preventing it from", 20, x)
	# 	x += 50
	# 	display_explanation.drawText("making errors.", 20, x)

	if screen == 3:
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		x = 100
		display_explanation.drawText("Each correctly sorted can", 20, x)
		x += 50
		display_explanation.drawText("that is sorted by the robot is worth", 20, x)
		x += 50
		display_explanation.drawText("double points, while you also lose", 20, x)
		x += 50
		display_explanation.drawText("two points if the robot makes a", 20, x)
		x += 50
		display_explanation.drawText("mistake.", 20, x)



	if screen == 4:
		#rot: -0.9975109062294765 -0.05050270631532978 0.049208420093273746 0.06363818669844469
		#pos: 1.690983429298919 1.8240480779136725 6.016126696620077
		viewNode.getField("position").setSFVec3f([1.690983429298919, 1.8240480779136725, 6.016126696620077])
		viewNode.getField("orientation").setSFRotation([-0.9975109062294765, -0.05050270631532978, 0.049208420093273746, 0.06363818669844469])
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		x = 100
		display_explanation.drawText("This screen will be used to display", 20, x)
		x += 50
		display_explanation.drawText("the view of the camera you see above,", 20, x)
		x += 50
		display_explanation.drawText("which the robot uses as an input", 20, x)
		x += 50
		display_explanation.drawText("to make decisions.", 20, x)
		x += 100
		display_explanation.drawText("The following slide will show you an", 20, x)
		x += 50
		display_explanation.drawText("example of the camera view.", 20, x)

	if screen == 5:
		if importCans:
			root_children.importMFNode(-1, "can1_example.wbo")
			root_children.importMFNode(-1, "can2_example.wbo")
			root_children.importMFNode(-1, "can3_example.wbo")
			importCans = False
		viewNode.getField("position").setSFVec3f([1.6935306498046143, 1.7752499661599694, 5.24838113227376])
		viewNode.getField("orientation").setSFRotation([-0.9975109062294765, -0.05050270631532978, 0.049208420093273746, 0.06363818669844469])
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		ir = display_explanation.imageLoad('tmp-' + condition + '.jpg')
		display_explanation.imagePaste(ir, 0, 0, False)
		display_explanation.imageDelete(ir)


	# if screen == 7 and condition in ["all", "written"]:
	# 	# display_explanation.setColor(0xFFFFFF)
	# 	# display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
	# 	# display_explanation.setColor(0x000000)
	# 	x = 500
	# 	y = 20
	# 	display_explanation.setColor(0xFFFFFF)
	# 	display_explanation.drawText("The robot is currently", y, x)
	# 	x += 50
	# 	display_explanation.drawText("unable to grasp cans that", y, x)
	# 	x += 50
	# 	display_explanation.drawText("are not standing upright.", y, x)
	# 	x = 500
	# 	y = 450
	# 	# display_explanation.drawText("It also recognises", y, x)
	# 	# x += 50
	# 	# display_explanation.drawText("if the color doesn't", y, x)
	# 	# x += 50
	# 	# display_explanation.drawText("match", y, x)

	# if screen == 8 and condition in ["all", "written"]:
	# 	ir = display_explanation.imageLoad('tmp-' + condition + '.jpg')
	# 	display_explanation.imagePaste(ir, 0, 0, False)
	# 	display_explanation.imageDelete(ir)

	# 	display_explanation.setColor(0xFFFFFF)
	# 	x = 500
	# 	y = 350
	# 	display_explanation.drawText("It also recognises", y, x)
	# 	x += 50
	# 	display_explanation.drawText("if the color is not", y, x)
	# 	x += 50
	# 	display_explanation.drawText("green or red.", y, x)


	# elif (screen == 9 and condition in ["all", "written"]) or (screen == 7 and condition == "visual"):
	# 	ir = display_explanation.imageLoad('tmp-' + condition + '.jpg')
	# 	display_explanation.imagePaste(ir, 0, 0, False)
	# 	display_explanation.imageDelete(ir)
	# 	x = 50
	# 	y = 450
	# 	display_explanation.setColor(0xFFFFFF)
	# 	display_explanation.drawText("As you see, the", y, x)
	# 	x += 50
	# 	display_explanation.drawText("robot's perception", y, x)
	# 	x += 50
	# 	display_explanation.drawText("is not 100% accurate,", y, x)
	# 	x += 50
	# 	display_explanation.drawText("and sometimes it can", y, x)
	# 	x += 50
	# 	display_explanation.drawText("make mistakes when ", y, x)
	# 	x += 50
	# 	display_explanation.drawText("recognising colors.", y, x)

	# 	x = 450
	# 	y = 450
	# 	if condition in ["all", "written"]:
	# 		display_explanation.drawText("The number reprents", y, x)
	# 		x += 50
	# 		display_explanation.drawText("the rank the robot", y, x)
	# 		x += 50
	# 		display_explanation.drawText("gives to the can.", y, x)
	# 		x += 50
	# 		display_explanation.drawText("It always tries to", y, x)
	# 		x += 50
	# 		display_explanation.drawText("take the one labelled", y, x)
	# 		x += 50
	# 		display_explanation.drawText("with 1 first", y, x)

	if screen == 6:
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		x = 100
		display_explanation.drawText("Your task is to prevent the mistakes,", 20, x)
		x += 50
		display_explanation.drawText("while performing the sorting task.", 20, x)
		x += 100
		display_explanation.drawText("There will be 3 runs of increasing", 20, x)
		x += 50
		display_explanation.drawText("difficulty. Each run takes around", 20, x)
		x += 50
		display_explanation.drawText("2 minutes to complete.", 20, x)
		x += 100
		display_explanation.drawText("As it is a collaborative work, you", 20, x)
		x += 50
		display_explanation.drawText("will collect scores together.", 20, x)
		x += 100
		display_explanation.drawText("Your collective score will be", 20, x)
		x += 50
		display_explanation.drawText("displayed in the end of each run.", 20, x)
	if screen == 7:
		display_explanation.setColor(0xFFFFFF)
		display_explanation.fillRectangle(0, 0, display_explanation.getWidth(), display_explanation.getHeight())
		display_explanation.setColor(0x000000)
		x = 350
		y = 250
		display_explanation.setFont("Lucida Console", 64, True)
		display_explanation.drawText("Good luck!", y, x)