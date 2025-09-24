from random import sample as random
import collections as memory
import numpy as np
import random

alpha=0.7
beta=0.5

class Memory:

     def __init__(self,MemorySize, act_dim, obs_dim):
         self.Memorysize = MemorySize
         self.container= memory.deque()
         self.containerSize = 0
         self.obs_dim=obs_dim
         self.act_dim = act_dim

     def size(self):
         return self.containerSize

     def select_batch(self, batchSize):
         return random.sample(self.container, batchSize)

     def add(self, experience):

         if self.containerSize < self.Memorysize:
            self.container.append(experience)
            self.containerSize += 1

         else:
             self.container.popleft()
             self.container.append(experience)


     def transform_sample(self,sample): 

         obs_dim=self.obs_dim
         
         #act_dim = self.act_dim

         current_state=  [x[0] for x in sample]
         actions =     np.asarray([x[1] for x in sample])
         rewards =     [x[2] for x in sample]
         next_state=   [x[3] for x in sample]
         done =        [x[4] for x in sample]
         batch_size = len(actions)

         current_state = np.resize(current_state,[batch_size,obs_dim])
         actions       = np.resize(actions, [batch_size, self.act_dim])
         rewards       = np.resize(rewards, [batch_size, 1])
         next_state    = np.resize(next_state, [batch_size, obs_dim])
         done          = np.resize(done, [batch_size, 1])


         return [current_state,actions,rewards,next_state,done]

     def select_sample(self,batch_size):
         #print "container size",self.containerSize
         sample = random.sample(self.container, int(batch_size))
         return self.transform_sample(sample)

     def clear_memory(self):
         self.container = memory.deque()
         self.containerSize=0
         self.num_experiences = 0











