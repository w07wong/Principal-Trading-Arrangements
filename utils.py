import numpy as np
from scipy import linalg
from stable_baselines.common import make_vec_env

from env.agent_env import AgentEnv
from env.principal_env import PrincipalEnv

def opt(tau, agent, theta, lam, T):
    '''
        Parameters:
            tau - the contract weights (array of size T)
            agent - agent's actions (array of size T)
            theta - price theta param.
            lam - price lambda param.
            T - timesteps

        Returns:
            xopt - the agent's best strategy when given the contract tau
            agentreward - reward for time t
            bestreward - reward from following the best strategy
            -cost - the principal's reward from offering contract tau when the agent follows the best strategy
            copt - the principal's best contract
            -bestcost/2 - the principal's reward when offering the best contract and the agent responds optimally
    '''
    # tau is a Tx1 vector of weights
    e = np.zeros(T)
    e[-1] = 1
    # Define Ahat
    r1 = np.zeros(T)
    r1[0] = 4*theta+2*lam
    r1[1] = -(lam+2*theta)
    c1 = np.transpose(r1)
    Ahat = linalg.toeplitz(c1)
    Ahat[-1,-1] = 1
    Ahat[-1,-2] = 0
    # Define Ehat
    r1 = np.zeros(T)
    r1[0] = lam+theta
    r1[1] = -theta
    c1 = np.zeros(T)
    c1[0] = r1[0]
    c1[1:] = 0
    Ehat = linalg.toeplitz(c1,r1)
    Ehat[-1] = 1
    # Define F
    r1 = np.zeros(T)
    r1[0] = 1
    c1 = np.zeros(T)
    c1[0] = 1
    c1[1] = -1
    F = linalg.toeplitz(c1,r1)
    AEhat = np.matmul(np.linalg.inv(Ahat),Ehat)
    cumulativeX = np.dot(AEhat,tau)
    xopt = np.matmul(F,cumulativeX)
    learned_agent = np.array(agent)
    Mhat = lam*AEhat + theta*np.matmul(F,AEhat)
    Mhat = Mhat + Mhat.T
    agentreward = np.dot(tau-learned_agent,(lam*cumulativeX+theta*learned_agent))

    bestreward = np.dot(tau-xopt,(lam*cumulativeX+theta*xopt))
    Mones = np.dot(np.linalg.inv(Mhat),np.ones(T))
    bestcost = 1/np.dot(np.ones(T),Mones)
    copt = Mones*bestcost
    cost = np.dot(tau,np.dot(Mhat,tau))/2
    return xopt, agentreward, bestreward, -cost, copt, -bestcost/2 


def get_scores(principal_model, agent_model, n_avg=100, theta=1, lam=1, sigma=1, T=10):
    '''
        Parameters:
            principal_model - the trained principal model
            agent_model - the trained agent model
            n_avg - the number of times to run the simulation for evaluation
            theta, lam, sigma, T - same as previous

        Returns:
            agent_r - list of agent rewards (size n_avg)
            best_r - list of optimal rewards (size n_avg)
            diff_r - (best_r - agent_r) (size n_avg)
    '''
    env = make_vec_env(lambda: AgentEnv(principal_model, theta=theta, lam=lam, sigma=sigma, T=T), n_envs=1)
    obs = env.reset()

    agent_r, best_agent_r, diff_agent = [], [], []
    principal_r, best_principal_r, diff_principal = [], [], []

    for _ in range(n_avg):
        agent_actions = []
        agent_rewards = []
        agent_observations = []
        for i in range(T):
            agent_observations.append(obs)
            action, _ = agent_model.predict(obs)
            obs, r, done, info = env.step(action)
            
            agent_actions.append(action[0][0])
            agent_rewards.append(r[0])

        agent_actions[-1] = 1 - sum(agent_actions[:-1])
        contract = agent_observations[0][0][1:-(T+1)]

        xopt, agent_reward, best_agent_reward, principal_reward, copt, best_principal_reward = opt(contract, agent_actions, theta, lam, T)
        # print('agent optimal', xopt)
        # print('agent learned', agent_actions)
        # print('agent reward', agentreward)
        # print('best reward', bestreward)
        agent_r.append(agent_reward)
        best_agent_r.append(best_agent_reward)
        diff_agent.append(best_agent_reward - agent_reward)
        principal_r.append(principal_reward)
        best_principal_r.append(best_principal_reward)
        diff_principal.append(best_principal_reward - principal_reward)
        # print('contract optimal', copt)
    
    return agent_r, best_agent_r, diff_agent, principal_r, best_principal_r, diff_principal