import numpy as np
class MultiArm_Bandit(object):
    def __init__(self,n,env_seed):
        self.n = n
        np.random.seed(env_seed)
        self.mean_vector = np.random.randn(n)
    
    def step(self,action):
        assert action < self.n
        reward = self.mean_vector[action]+np.random.normal()
        #np.random.normal(loc=self.mean_vector[action],scale=self.stddev_vector[action])
        return reward
    
    @property
    def bandit_config(self):
        return self.mean_vector

