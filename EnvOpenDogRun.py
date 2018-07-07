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

# Helpers
def cross(v0, v1):
    return [ v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0] ]

def rotateVec(q, v):
    uv = cross(q[0:3], v)
    uuv = cross(q[0:3], uv)

    scaleUv = 2.0 * q[3]

    uv[0] *= scaleUv
    uv[1] *= scaleUv
    uv[2] *= scaleUv

    uuv[0] *= 2.0
    uuv[1] *= 2.0
    uuv[2] *= 2.0

    return [ v[0] + uv[0] + uuv[0],
        v[1] + uv[1] + uuv[1],
        v[2] + uv[2] + uuv[2] ]

def rotationInverse(q):
    scale = 1.0 / max(0.0001, np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]))

    return [ -q[0] * scale, -q[1] * scale, -q[2] * scale, q[3] * scale ]

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

        # Observations have additional information - IMU data, scaled into [-1, 1] through tanh
        additionalLow = np.array(6 * [ -1.0 ])
        additionalHigh = np.array(6 * [ 1.0 ])

        observationLowValues = np.concatenate((motorLowValues, additionalLow), axis=0)
        observationHighValues = np.concatenate((motorHighValues, additionalHigh), axis=0)

        self.observation_space = spaces.Box(observationLowValues, observationHighValues)
        self.action_space = spaces.Box(motorLowValues, motorHighValues)

        # Constants
        self.tipThresh = 0.7 # Threshold for detecting falling over
        self.maxTime = 10.0 # 5 Seconds per episode
        self.timeStep = 0.02 # 50 FPS
        self.motorForce = 100
        self.motorSmoothing = 0.3
        self.motorMomentum = 0.8
        self.lAccelSensitivity = 1.0 # Scaling factor for squashing lAccel into [-1, 1]
        self.eAccelSensitivity = 1.0 # Scaling factor for squashing eAccel into [-1, 1]

        self._seed()

        self.viewer = None
        self._configure()

        self.currentSimTime = 0.0

        self.smoothedMotorTargets = self.numJoints * [0]
        self.motorDeltas = self.numJoints * [0]

        # Track velocities
        self.lVel = 3 * [ 0.0 ]
        self.eVel = 4 * [ 0.0 ]

        self.done = False

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getUpVector(self):
        _, rot = p.getBasePositionAndOrientation(self.dog)

        return rotateVec(rot, [ 0.0, 0.0, 1.0 ])

    def _IMU(self):
        lVel, eVel = p.getBaseVelocity(self.dog)

        # Convert to euler, a standard IMU format
        eAccel = [ eVel[0] - self.eVel[0], eVel[1] - self.eVel[1], eVel[2] - self.eVel[2] ]

        # Get lAccel in frame of rotation
        _, rot = p.getBasePositionAndOrientation(self.dog)

        rotInv = rotationInverse(rot)

        lAccel = [ lVel[0] - self.lVel[0], lVel[1] - self.lVel[1], lVel[2] - self.lVel[2] ]

        relativeLAccel = rotateVec(rotInv, lAccel)

        return relativeLAccel + eAccel

    def _getObservation(self):
        # Obtain observation
        observation = self.numJoints * [ 0.0 ]

        for i in range(self.numJoints):
            observation[i] = min(self.observation_space.high[i], max(self.observation_space.low[i], p.getJointState(self.dog, i)[0]))

        # Additional information (IMU)
        imu = self._IMU()

        # Rescale IMU data into [-1, 1]
        scaledIMU = [ np.tanh(imu[0] * self.lAccelSensitivity), np.tanh(imu[1] * self.lAccelSensitivity), np.tanh(imu[2] * self.lAccelSensitivity),
            np.tanh(imu[3] * self.eAccelSensitivity), np.tanh(imu[4] * self.eAccelSensitivity), np.tanh(imu[5] * self.eAccelSensitivity) ]

        observation += scaledIMU

        return np.array(observation)

    def _step(self, action):
        p.stepSimulation()

        self.currentSimTime += self.timeStep

        lVel, eVel = p.getBaseVelocity(self.dog)

        reward = lVel[0] * 0.1 # Move along positive X axis

        up = self._getUpVector()

        # If fell over
        if up[2] < self.tipThresh:
            reward = -5.0 # Punish for tipping over

            self.done = True

        # Apply action
        for i in range(self.numJoints):
            delta = self.motorSmoothing * (action[i] - self.smoothedMotorTargets[i]) + self.motorMomentum * self.motorDeltas[i]

            self.smoothedMotorTargets[i] += delta

            self.motorDeltas[i] = delta

            p.setJointMotorControl2(self.dog, i, p.POSITION_CONTROL, targetPosition=self.smoothedMotorTargets[i], force=self.motorForce)

        if self.currentSimTime > self.maxTime:
            self.done = True

        # Must call before updating velocities
        observation = self._getObservation()

        self.lVel = lVel
        self.eVel = eVel

        return observation, reward, self.done, {}

    def _reset(self):
        self.done = False
        self.currentSimTime = 0.0

        self.lVel = 3 * [ 0.0 ]
        self.eVel = 4 * [ 0.0 ]

        p.resetSimulation()

        floor = p.loadURDF("myplane.urdf", [ 0.0, 0.0, 0.0 ])

        p.setPhysicsEngineParameter(numSolverIterations=100)

        self.dog = p.loadURDF("opendog.urdf", [ 0.0, 0.0, 0.36 ], globalScaling=0.1, flags=p.URDF_USE_SELF_COLLISION)

        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(self.timeStep)

        p.setRealTimeSimulation(0)

        return self._getObservation()

    def _render(self, mode='human', close=False):
        return

    # For compatibility with older gym versions
    if parse_version(gym.__version__)>=parse_version('0.9.6'):
        render = _render
        reset = _reset
        seed = _seed
        step = _step