import numpy as np
class RandomAgent(object):
    def __init__(self,n):
        self.n = n
        self.estimation = np.zeros((self.n,),np.float32)
        self.visit_count = np.zeros((self.n,),np.float32)
    def action(self):
        return np.random.choice(self.n)
    def update_estimation(self,action,reward):
        self.visit_count[action] += 1
        self.estimation[action] = self.estimation[action] + (reward - self.estimation[action]) / self.visit_count[action]
    
class GreedyAgent(object):
    def __init__(self,n):
        self.n = n
        self.estimation = np.zeros((self.n,),np.float32)
        self.visit_count = np.zeros((self.n,),np.float32)
    def action(self):
        return np.argmax(self.estimation)
    def update_estimation(self,action,reward):
        self.visit_count[action] += 1
        self.estimation[action] = self.estimation[action] + (reward - self.estimation[action]) / self.visit_count[action]

class OptimisticGreedyAgent(object):
    def __init__(self,n):
        self.n = n
        self.estimation = np.zeros((self.n,),np.float32) + 10
        self.visit_count = np.zeros((self.n,),np.float32)
    def action(self):
        return np.argmax(self.estimation)
    def update_estimation(self,action,reward):
        self.visit_count[action] += 1
        self.estimation[action] = self.estimation[action] + (reward - self.estimation[action]) / self.visit_count[action]

class EGreedyAgent(object):
    def __init__(self,n,e=0.1):
        self.n = n
        self.e = e
        self.estimation = np.zeros((self.n,),np.float32)
        self.visit_count = np.zeros((self.n,),np.float32)
    
    def action(self):
        rnd = np.random.rand()
        if rnd < self.e:
            return np.random.choice(self.n)
        return np.argmax(self.estimation)
    
    def update_estimation(self,action,reward):
        self.visit_count[action] += 1
        self.estimation[action] = self.estimation[action] + (reward - self.estimation[action]) / self.visit_count[action]

    def dec_epsilon(self,dec):
        self.e -= dec
        self.e = max(self.e,0)

class UpperBoundAgent(object):
    def __init__(self,n,c):
        self.n = n
        self.c = c
        self.estimation = np.zeros((self.n,),np.float32)
        self.visit_count = np.zeros((self.n,),np.float32)
    def action(self):
        t = np.sum(self.visit_count)+1
        lower = (self.visit_count)
        return np.argmax(self.estimation + self.c*np.sqrt(np.log(t)/lower))
    def update_estimation(self,action,reward):
        self.visit_count[action] += 1
        self.estimation[action] = self.estimation[action] + (reward - self.estimation[action]) / self.visit_count[action]
