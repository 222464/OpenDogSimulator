"""
OpenDog simulator running behavior environment.
"""
import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p
import pybullet_data
from pkg_resources import parse_version

logger = logging.getLogger(__name__)

class EnvOpenDogRun(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    numJoints = 12

    def __init__(self, renders=True):
        self._renders = renders
        if (renders):
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # Motor ranges
        motorLowValues = np.array(self.numJoints * [ -1.0 ])
        motorHighValues = np.array(self.numJoints * [ 1.0 ])

        # Reduce range on coxa joints
        offset = 0

        for i in range(4):
            motorLowValues[i * 3 + offset] = -0.5
            motorHighValues[i * 3 + offset] = 0.5

        # Observations have additional information - 6 components for up vector and forward vector
        additionalLow = np.array(6 * [-1])
        additionalHigh = np.array(6 * [1])

        observationLowValues = np.concatenate((motorLowValues, additionalLow), axis=0)
        observationHighValues = np.concatenate((motorHighValues, additionalHigh), axis=0)

        self.observation_space = spaces.Box(observationLowValues, observationHighValues)
        self.action_space = spaces.Box(motorLowValues, motorHighValues)

        # Constants
        self.tipThresh = 0.6 # Threshold for detecting falling over (based on cosine of angle to upright vector)
        self.maxTime = 5.0 # 5 Seconds per episode
        self.timeStep = 0.02 # 50 FPS
        self.motorForce = 50 # 50 Newtons

        self._seed()

        self.viewer = None
        self._configure()

        self.currentSimTime = 0.0

        self.done = False

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getBasis(self):
        _, rot = p.getBasePositionAndOrientation(self.dog)

        # Find upright vector
        forward = p.getMatrixFromQuaternion(rot)[0:3]
        up = p.getMatrixFromQuaternion(rot)[3:6]
        right = p.getMatrixFromQuaternion(rot)[6:9]
       
        # Normalization factors
        normFactorF = 1.0 / max(0.0001, np.sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]))
        normFactorU = 1.0 / max(0.0001, np.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]))
        normFactorR = 1.0 / max(0.0001, np.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]))
        
        return [ [ forward[0] * normFactorF, forward[1] * normFactorF, forward[2] * normFactorF ],
            [ up[0] * normFactorU, up[1] * normFactorU, up[2] * normFactorU ],
            [ right[0] * normFactorR, right[1] * normFactorR, right[2] * normFactorR ] ]

    def _getObservation(self):
        basis = self._getBasis()

        # Obtain observation
        observation = self.numJoints * [ 0.0 ]

        for i in range(self.numJoints):
            observation[i] = min(self.observation_space.high[i], max(self.observation_space.low[i], p.getJointState(self.dog, i)[0]))

        # Additional information (2 components of basis)
        observation += basis[0] + basis[1]

        return np.array(observation)

    def _step(self, action):
        p.stepSimulation()

        self.currentSimTime += self.timeStep

        lVel, _ = p.getBaseVelocity(self.dog)

        reward = lVel[0] * 0.1 # Move along positive X axis

        basis = self._getBasis()

        up = basis[1]

        # If fell over
        if up[1] < self.tipThresh:
            reward = -5.0 # Punish for tipping over

            self.done = True

        # Apply action
        for i in range(self.numJoints):
            p.setJointMotorControl2(self.dog, i, p.POSITION_CONTROL, targetPosition=action[i], force=self.motorForce)

        if self.currentSimTime > self.maxTime:
            self.done = True

        return self._getObservation(), reward, self.done, {}

    def _reset(self):
        self.done = False
        self.currentSimTime = 0.0

        p.resetSimulation()

        floor = p.loadURDF("myplane.urdf", [0, 0, 0])

        p.changeDynamics(floor, 0, lateralFriction=1.0)

        self.dog = p.loadURDF("opendog.urdf", [0, 0, 0.36], globalScaling=0.1, flags=p.URDF_USE_SELF_COLLISION)

        for i in range(self.numJoints):
            p.changeDynamics(self.dog, i, lateralFriction=1.0)

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timeStep)

        p.setRealTimeSimulation(0)

        totalReward = 0.0

        return self._getObservation()

    def _render(self, mode='human', close=False):
        return

    # For compatibility with older gym versions
    if parse_version(gym.__version__)>=parse_version('0.9.6'):
        render = _render
        reset = _reset
        seed = _seed
        step = _step