import gym
from gym import spaces
import numpy as np

'''
    Principal environment
'''
class PrincipalEnv(gym.Env):
    def __init__(self,
                 agent,
                 agent_env,
                 theta,
                 lam,
                 sigma,
                 principal_min_action=0, # Should experiment with this value. 1/T is just the mean.
                 principal_max_action=1, # Should experiment with this value. 1/T is just the mean.
                 agent_min_action=-1,
                 agent_max_action=1,
                 min_price=0,
                 max_price=1000, # Should experiment with this value.
                 P0=0,
                 T=30,
                 multi_agent=False):
        '''
            Parameters:
                agent - the agent model
                agent_env - the agent enviroment
                theta - the temporary price impact parameter.
                lam - the permanent price impact parameter.
                sigma - the standard devation of the noise at each time period.
                principal_min_action - the minimum value to affect the principal's contract function.
                principal_max_action - the maximum value to affect the principal's contract function.
                agent_min_action - minimum number of shares the agent can buy at a timestep.
                agent_max_action - maximum number of shares the agent can buy at a timestep.
                min_price - minimum price, used to bound observation space and price model
                max_price - maximum price, used to bound observation space and price model.
                P0 - the initial price.
                T - the number of time periods to complete trading.
                multi_agent - if True, environment uses two different agent and observation spaces
        '''
        super(PrincipalEnv, self).__init__()
        self.agent = agent
        self.agent_env = agent_env
        self.C = np.zeros(shape=(T + 1)) # Initial principal state. No contract has been made yet.
        self.theta = theta
        self.lam = lam
        self.std = sigma
        self.principal_min_action = principal_min_action,
        self.principal_max_action = principal_max_action,
        self.agent_min_action = agent_min_action,
        self.agent_max_action = agent_max_action,
        self.min_price = min_price
        self.max_price = max_price
        self.T = T
        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        self.multi_agent = multi_agent

        # The agent can buy maximum 1 share or sell maximium 1 shares at each timestep.
        self.action_space = spaces.Box(low=principal_min_action, high=principal_max_action, shape=(T + 1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=min(agent_min_action, min_price), high=max(agent_max_action, max_price), shape=(3 * T + 2,), dtype=np.float32)
        
        # Initial contract
        self.C = np.zeros(self.T + 1)

        # Initial prices observed
        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        
        # Initial agent actions
        self.Xs = np.zeros(self.T) # Shares purchased at times 1,...,T
        
        self.reset()

    def step(self, contract):
        '''
            Parameters:
                contract - a contract
            
            Returns:
                obs - next observation
                reward - reward for time t
                done - whether to end the episode
        '''
        # Normalize contract
        contract[0] = 0
        contract = contract / np.sum(contract)
        self.C = contract

        self.agent_env.C = contract
        obs = self.agent_env.reset()

        actions = []
        for i in range(self.T):
            action, _ = self.agent.predict(obs)
            obs, _, _, _ = self.agent_env.step(action)
            if i == self.T - 2:
                prices = obs[0][len(contract):]
            actions.append(action[0][0])
        actions[-1] = 1 - sum(actions[:-1])
        
        self.Xs = actions
        self.Ps = prices

        reward = - contract.dot(prices)
        # print(reward)

        # xopt, agentreward, bestreward, _, copt, _ = opt(contract[1:], actions, self.theta, self.lam, self.T)
        # print(agentreward)
        
        self.done = True

        return self._get_obs(), reward, self.done, {}
    
    def reset(self):
        # Observation is previous contract, prices encountered and shared purchased by agent.
        self.observations = self._get_obs()
        
        self.C = np.zeros(self.T + 1)

        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        self.Xs = np.zeros(self.T) # Shares purchased at times 1,...,T
        
        self.done = False

        return self.observations

    def _get_obs(self):
        return np.concatenate([self.C, self.Ps, self.Xs])

    def render(self):
        pass