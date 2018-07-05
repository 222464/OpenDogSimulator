import gym
import numpy as np
from EnvOpenDogStand import EnvOpenDogStand
import time
from es import CMAES
import tensorflow as tf
import gc

env = EnvOpenDogStand(renders=False)
env.seed(0)

episode_count = 100
reward = 0
done = False




class StandActor:
    def __init__(self):
        self.sess = tf.Session()
        batchSize = 1
        self.position = tf.placeholder(tf.float32,shape=(batchSize,12))
        self.velocity = tf.placeholder(tf.float32,shape=(batchSize,12))
        concat = tf.concat([self.position,self.velocity],axis=1)
        layerSize = 30
        h = tf.nn.relu(tf.layers.Dense(layerSize)(concat))
        h = tf.nn.relu(tf.layers.Dense(layerSize)(h))
        self.dp = 0.1*tf.layers.Dense(12)(h)
        self.sess.run(tf.global_variables_initializer())

    def getNumberOfParams(self):
        tv = tf.trainable_variables()
        num = 0
        for v in tv:
            prod = np.prod(np.array(v.shape))
            num += prod.value
        return num

    def getParameterValue(self):
        tv = tf.trainable_variables()
        vals = self.sess.run( [tf.reshape(v,(-1,)) for v in tv] )
        return np.concatenate(vals)

    def loadParams(self,p):
        tv = tf.trainable_variables(scope=None)
        num = 0
        for v in tv:
            prod = np.prod(np.array(v.shape))
            v.load( np.reshape( p[num:num+prod.value], v.shape) ,self.sess)
            num += prod.value

    def act(self, state):
        jp = np.expand_dims( np.array(state["JointPosition"]),axis=0)
        jv = np.expand_dims(np.array(state["JointVelocity"]), axis=0)

        feed_dict = {self.position:jp,
                     self.velocity:jv }
        dpval = self.sess.run(self.dp,feed_dict)
        return dpval[0]


agent = StandActor()

nbParams = agent.getNumberOfParams()
x0 = agent.getParameterValue()
solver = CMAES(x0)

#agent.loadParams( np.zeros( nbParams ) )


def evaluate( sol ):
    agent.loadParams( sol )
    ob = env.reset()
    while True:
        action = agent.act(ob)
        ob, reward, done, _ = env.step(action)
        #time.sleep(0.01)
        if done:
            print("episode done")
            print(reward)
            gc.collect()
            # Only return last reward
            return reward

MY_REQUIRED_REWARD = -0.1

while True:
  # ask the ES to give us a set of candidate solutions
  solutions = solver.ask()

  # create an array to hold the solutions.
  # solver.popsize = population size
  rewards = np.zeros(solver.popsize)

  # calculate the reward for each given solution
  # using your own evaluate() method
  for i in range(solver.popsize):
    print("candidate " + str(i))
    rewards[i] = evaluate(solutions[i])

  # give rewards back to ES
  solver.tell(rewards)

  # get best parameter, reward from ES
  reward_vector = solver.result()
  print("current Best : ")
  print(reward_vector[1] )

  if reward_vector[1] > MY_REQUIRED_REWARD :
    break


# Close the env and write monitor result info to disk
env.close()