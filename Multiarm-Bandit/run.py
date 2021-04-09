import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from bandit_env import MultiArm_Bandit
from agents import RandomAgent,GreedyAgent,EGreedyAgent,UpperBoundAgent,OptimisticGreedyAgent

def smooth(data,alpha=0.9):
    last = data[0]
    result = []
    for y in data:
        result.append(alpha*last + (1-alpha)*y)
    return result

N = 10
Episodes = 1000
ExploreSteps = 0
GameSteps = 2000
#Average Score
random_agent_score = np.zeros((Episodes,GameSteps),np.float32)
greedy_agent_score = np.zeros((Episodes,GameSteps),np.float32)
egreedy_agent_score = np.zeros((Episodes,GameSteps),np.float32)
egreedy_agent2_score = np.zeros((Episodes,GameSteps),np.float32)
ogreedy_agent_score = np.zeros((Episodes,GameSteps),np.float32)
ucb_agent_score = np.zeros((Episodes,GameSteps),np.float32)
best_agent_score = np.zeros((Episodes,GameSteps),np.float32)
#Average Best Action
random_agent_best_prop = np.zeros((Episodes,GameSteps),np.float32)
greedy_agent_best_prop = np.zeros((Episodes,GameSteps),np.float32)
egreedy_agent_best_prop = np.zeros((Episodes,GameSteps),np.float32)
egreedy_agent2_best_prop = np.zeros((Episodes,GameSteps),np.float32)
ogreedy_agent_best_prop = np.zeros((Episodes,GameSteps),np.float32)
ucb_agent_best_prop = np.zeros((Episodes,GameSteps),np.float32)

for episode in range(Episodes):
    env = MultiArm_Bandit(N,episode)
    best_action = np.argmax(env.bandit_config)

    random_agent = RandomAgent(N)
    greedy_agent = GreedyAgent(N)
    egreedy_agent = EGreedyAgent(N,0.1)
    egreedy_agent2 = EGreedyAgent(N,0.01)
    ogreedy_agent = OptimisticGreedyAgent(N)
    ucb_agent = UpperBoundAgent(N,2)

    for i in range(ExploreSteps):
        action = np.random.choice(N)
        reward = env.step(action)
        random_agent.update_estimation(action,reward)
        greedy_agent.update_estimation(action,reward)
        egreedy_agent.update_estimation(action,reward)
        egreedy_agent2.update_estimation(action,reward)
        ogreedy_agent.update_estimation(action,reward)
        ucb_agent.update_estimation(action,reward)

    for i in range(GameSteps):
        # random agent interacion
        random_agent_action = random_agent.action()
        random_agent_reward = env.step(random_agent_action)
        random_agent.update_estimation(random_agent_action,random_agent_reward)
        random_agent_score[episode][i] = random_agent_reward
        if i == 0:
            random_agent_best_prop[episode][i] = (best_action == random_agent_action)
        else:
            random_agent_best_prop[episode][i] = random_agent_best_prop[episode][i-1] + ((best_action == random_agent_action) - random_agent_best_prop[episode][i-1])/(i+1)

        # greedy agent interaction
        greedy_agent_action = greedy_agent.action()
        greedy_agent_reward = env.step(greedy_agent_action)
        greedy_agent.update_estimation(greedy_agent_action,greedy_agent_reward)
        greedy_agent_score[episode][i] = greedy_agent_reward
        if i == 0:
            greedy_agent_best_prop[episode][i] = (best_action == greedy_agent_action)
        else:
            greedy_agent_best_prop[episode][i] = greedy_agent_best_prop[episode][i-1] + ((best_action == greedy_agent_action) - greedy_agent_best_prop[episode][i-1])/(i+1)
        

        # egreedy agent interaction
        egreedy_agent_action = egreedy_agent.action()
        egreedy_agent_reward = env.step(egreedy_agent_action)
        egreedy_agent.update_estimation(egreedy_agent_action,egreedy_agent_reward)
        egreedy_agent_score[episode][i] = egreedy_agent_reward
        if i == 0:
            egreedy_agent_best_prop[episode][i] = (best_action == egreedy_agent_action)
        else:
            egreedy_agent_best_prop[episode][i] = egreedy_agent_best_prop[episode][i-1] + ((best_action == egreedy_agent_action) - egreedy_agent_best_prop[episode][i-1])/(i+1)
        
        # egreedy agent interaction
        egreedy_agent2_action = egreedy_agent2.action()
        egreedy_agent2_reward = env.step(egreedy_agent2_action)
        egreedy_agent2.update_estimation(egreedy_agent2_action,egreedy_agent2_reward)
        egreedy_agent2_score[episode][i] = egreedy_agent2_reward
        if i == 0:
            egreedy_agent2_best_prop[episode][i] = (best_action == egreedy_agent2_action)
        else:
            egreedy_agent2_best_prop[episode][i] = egreedy_agent2_best_prop[episode][i-1] + ((best_action == egreedy_agent2_action) - egreedy_agent2_best_prop[episode][i-1])/(i+1)
        
        # ogreedy agent interaction
        ogreedy_agent_action = ogreedy_agent.action()
        ogreedy_agent_reward = env.step(ogreedy_agent_action)
        ogreedy_agent.update_estimation(ogreedy_agent_action,ogreedy_agent_reward)
        ogreedy_agent_score[episode][i] = ogreedy_agent_reward
        if i == 0:
            ogreedy_agent_best_prop[episode][i] = (best_action == ogreedy_agent_action)
        else:
            ogreedy_agent_best_prop[episode][i] = ogreedy_agent_best_prop[episode][i-1] + ((best_action == ogreedy_agent_action) - ogreedy_agent_best_prop[episode][i-1])/(i+1)

        # ogreedy agent interaction
        ucb_agent_action = ucb_agent.action()
        ucb_agent_reward = env.step(ucb_agent_action)
        ucb_agent.update_estimation(ucb_agent_action,ucb_agent_reward)
        ucb_agent_score[episode][i] = ucb_agent_reward
        if i == 0:
            ucb_agent_best_prop[episode][i] = (best_action == ucb_agent_action)
        else:
            ucb_agent_best_prop[episode][i] = ucb_agent_best_prop[episode][i-1] + ((best_action == ucb_agent_action) - ucb_agent_best_prop[episode][i-1])/(i+1)
        
        # best agent interaction
        best_reward = env.step(best_action)
        best_agent_score[episode][i] = best_reward

random_agent_prop = np.mean(random_agent_best_prop,axis=0)
greedy_agent_prop  = np.mean(greedy_agent_best_prop,axis=0)
egreedy_agent_prop  = np.mean(egreedy_agent_best_prop,axis=0)
egreedy_agent2_prop  = np.mean(egreedy_agent2_best_prop,axis=0)
ogreedy_agent_prop  = np.mean(ogreedy_agent_best_prop,axis=0)
ucb_agent_prop = np.mean(ucb_agent_best_prop,axis=0)

random_agent_score = np.mean(random_agent_score,axis=0)
greedy_agent_score = np.mean(greedy_agent_score,axis=0)
egreedy_agent_score = np.mean(egreedy_agent_score,axis=0)
egreedy_agent2_score = np.mean(egreedy_agent2_score,axis=0)
ogreedy_agent_score = np.mean(ogreedy_agent_score,axis=0)
ucb_agent_score = np.mean(ucb_agent_score,axis=0)
x = np.arange(GameSteps)

# Greedy Curve
sns.set()
agents = ['greedy','egreedy-0.1','egreedy-0.01']
sns.lineplot(x,smooth(greedy_agent_prop,0.0))
sns.lineplot(x,smooth(egreedy_agent_prop,0.0))
sns.lineplot(x,smooth(egreedy_agent2_prop,0.0))
plt.ylim(0,1)
plt.xlabel("Steps")
plt.ylabel("%Optimal Action")
plt.legend(agents)
plt.show()
sns.lineplot(x,smooth(greedy_agent_score,0.0))
sns.lineplot(x,smooth(egreedy_agent_score,0.0))
sns.lineplot(x,smooth(egreedy_agent2_score,0.0))
plt.ylim(0,2)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend(agents)
plt.show()


agents = ['egreedy-0.1','optimisic-greedy']
sns.lineplot(x,smooth(egreedy_agent_prop,0.0))
sns.lineplot(x,smooth(ogreedy_agent_prop,0.0))
plt.ylim(0,1)
plt.xlabel("Steps")
plt.ylabel("%Optimal Action")
plt.legend(agents)
plt.show()
sns.lineplot(x,smooth(egreedy_agent_score,0.0))
sns.lineplot(x,smooth(ogreedy_agent_score,0.0))
plt.ylim(0,2)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend(agents)
plt.show()

agents = ['random','greedy','egreedy-0.1','optimisic-greedy','ucb']
sns.lineplot(x,smooth(random_agent_prop,0.0))
sns.lineplot(x,smooth(greedy_agent_prop,0.0))
sns.lineplot(x,smooth(egreedy_agent_prop,0.0))
sns.lineplot(x,smooth(ogreedy_agent_prop,0.0))
sns.lineplot(x,smooth(ucb_agent_prop,0.0))
plt.ylim(0,1)
plt.legend(agents)
plt.show()
sns.lineplot(x,smooth(random_agent_score,0.0))
sns.lineplot(x,smooth(greedy_agent_score,0.0))
sns.lineplot(x,smooth(egreedy_agent_score,0.0))
sns.lineplot(x,smooth(ogreedy_agent_score,0.0))
sns.lineplot(x,smooth(ucb_agent_score,0.0))
plt.ylim(0,2)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend(agents)
plt.show()







