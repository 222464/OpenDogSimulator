import gym
import numpy as np
from EnvOpenDogRun import EnvOpenDogRun
import time
import eogmaneo

env = EnvOpenDogRun()
env.seed(0)

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

inputWidth = 6
inputHeight = 3
inputColumnSize = 16

actionWidth = 4
actionHeight = 3
actionColumnSize = 16

h = eogmaneo.Hierarchy()
h.create([ (inputWidth, inputHeight), (actionWidth, actionHeight) ], [ inputColumnSize, actionColumnSize ], [ False, True ], lds, 123)

# Set parameters
for i in range(len(lds)):
    l = h.getLayer(i)
    l._alpha = 0.01
    l._beta = 0.0001
    l._gamma = 0.98
    l._maxRepaySamples = 32

actionSDR = list(h.getPredictions(1))

########################### Simulate ###########################

episodeCount = 1000
reward = 0

realTime = True

for i in range(episodeCount):
	print("Episode: " + str(i))

	observation = env.reset()

	totalReward = 0.0

	while True:
		timeStart = time.clock()

		# Create the input SDR
		inputSDR = inputWidth * inputHeight * [ 0 ]

		for i in range(env.observation_space.shape[0]):
			value = observation[i]

			index = int((value - env.observation_space.low[i]) / (env.observation_space.high[i] - env.observation_space.low[i]) * (inputColumnSize - 1) + 0.5)

			inputSDR[i] = index

		h.step(cs, [ inputSDR, actionSDR ], reward, True)

		actionSDR = list(h.getPredictions(1))

		action = env.action_space.shape[0] * [ 0.0 ]

		for i in range(env.action_space.shape[0]):
			# Exploration
			if np.random.rand() < 0.04:
				actionSDR[i] = np.random.randint(0, actionColumnSize)
			
			# Rescale
			action[i] = actionSDR[i] / (actionColumnSize - 1) * (env.action_space.high[i] - env.action_space.low[i]) + env.action_space.low[i]

		observation, reward, done, _ = env.step(np.array(action))

		totalReward += reward

		########################

		timeEnd = time.clock()

		if realTime:
			time.sleep(max(0.0, env.timeStep - (timeEnd - timeStart)))

		if done:
			print("Received " + str(totalReward) + " reward.")
			break

env.close()