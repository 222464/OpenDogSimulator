import os
import pybullet
import pybullet_data
import numpy as np
import time
from random import randint
import eogmaneo

pybullet.connect(pybullet.GUI)

########################### Create Agent ###########################

# Create hierarchy
cs = eogmaneo.ComputeSystem(4)

lds = []

layerSize = 3

for i in range(3):
    ld = eogmaneo.LayerDesc()

    ld._width = layerSize
    ld._height = layerSize
    ld._columnSize = 32
    ld._forwardRadius = 2
    ld._backwardRadius = 2
    ld._temporalHorizon = 2
    ld._ticksPerUpdate = 2

    lds.append(ld)

inputWidth = 4
inputHeight = 3
inputColumnSize = 16

actionWidth = 4
actionHeight = 3
actionColumnSize = 12

h = eogmaneo.Hierarchy()
h.create([ (inputWidth, inputHeight), (actionWidth, actionHeight) ], [ inputColumnSize, actionColumnSize ], [ False, True ], lds, 123)

# Set parameters
for i in range(len(lds)):
    l = h.getLayer(i)
    l._alpha = 0.01
    l._beta = 0.0001
    l._gamma = 0.95
    l._maxRepaySamples = 32

########################### Simulate ###########################

# Motor ranges
lowValues = 12 * [ -1.0 ]
highValues = 12 * [ 1.0 ]

# Reduce range on coxa joints
offset = 0

for i in range(4):
	lowValues[i * 3 + offset] = -0.5
	highValues[i * 3 + offset] = 0.5

maxMotorForce = 50
smoothSpeed = 0.25

for ep in range(10000):
	pybullet.resetSimulation()

	floor = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)

	pybullet.changeDynamics(floor, 0, lateralFriction=1.0)

	bot = pybullet.loadURDF("opendog.urdf", [0, 0, 0.36], globalScaling=0.1, flags=pybullet.URDF_USE_SELF_COLLISION)

	numJoints = pybullet.getNumJoints(bot)

	for i in range(12):
		pybullet.changeDynamics(bot, i, lateralFriction=1.0)

	pybullet.setGravity(0, 0, -9.81)

	totalReward = 0.0

	actionSDR = actionWidth * actionHeight * [ 0 ]

	# Targets with smoothing
	currentTargets = 12 * [ 0.0 ]

	for t in range(2000):
		pybullet.stepSimulation()

		# Obtain observation
		observation = 12 * [ 0.0 ]
		
		for i in range(12):
			observation[i] = min(highValues[i], max(lowValues[i], pybullet.getJointState(bot, i)[0]))

		pos, rot = pybullet.getBasePositionAndOrientation(bot)

		lVel, rVel = pybullet.getBaseVelocity(bot)

		reward = lVel[0] * 0.1 # Move along positive X axis

		# Find upright vector
		up = pybullet.getMatrixFromQuaternion(rot)[3:6]
		
		# Normalization factor
		normFactor = 1.0 / max(0.0001, np.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]))

		reset = False

		# If fell over
		if up[1] * normFactor < 0.5:
			reset = True
			reward = -5.0
			print("Fell over!")

		totalReward += reward

		# Create the input SDR
		inputSDR = inputWidth * inputHeight * [ 0 ]

		for i in range(12):
			value = observation[i]

			index = int((value - lowValues[i]) / (highValues[i] - lowValues[i]) * (inputColumnSize - 1) + 0.5)

			inputSDR[i] = index

		h.step(cs, [ inputSDR, actionSDR ], reward, True)

		actionSDR = list(h.getPredictions(1))

		action = actionWidth * actionHeight * [ 0.0 ]

		for i in range(12):
			if np.random.rand() < 0.01: # Exploration
				actionSDR[i] = np.random.randint(0, actionColumnSize)
			
			# Rescale
			action[i] = actionSDR[i] / (actionColumnSize - 1) * (highValues[i] - lowValues[i]) + lowValues[i]

			currentTargets[i] += smoothSpeed * (action[i] - currentTargets[i])

		# Apply motor commands
		for i in range(12):
			pybullet.setJointMotorControl2(bodyUniqueId=bot, 
				jointIndex=i, 
				controlMode=pybullet.POSITION_CONTROL,
				targetPosition=currentTargets[i],
				force=maxMotorForce)

		if reset:
			break

	print("Completed episode " + str(ep + 1) + " with a total reward of " + str(totalReward))
	
pybullet.resetSimulation()
pybullet.disconnect()