import gym
from gym import spaces
import numpy as np

'''
    Agent environment
'''
class AgentEnv(gym.Env):
    def __init__(self,
                 principal,
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
                principal - the principal model
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
        super(AgentEnv, self).__init__()
        self.principal = principal
        self.C = np.zeros(shape=(T + 1)) # Initial principal state. No contract has been made yet.
        self.theta = theta
        self.lam = lam
        self.std = sigma
        self.principal_min_action = principal_min_action
        self.principal_max_action = principal_max_action
        self.agent_min_action = agent_min_action
        self.agent_max_action = agent_max_action
        self.min_price = min_price
        self.max_price = max_price
        self.P0 = P0
        self.T = T
        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        self.Xs = np.zeros(self.T) # Shares purchased at times 1,...,T

        # The agent can buy maximum 1 share or sell maximium 1 shares at each timestep.
        self.action_space = spaces.Box(low=agent_min_action, high=agent_max_action, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=min(principal_min_action, min_price), high=max(principal_max_action, max_price), shape=(2 * T + 2,), dtype=np.float32)

        self.reset()

    def get_price_at_t(self, t):
        '''
            Calculates the price impact model at timestep t.
        '''
        price = self.Ps[0] + self.lam * self.Xs[t] + self.theta * np.sum(self.Xs) + np.sum(self.epsilons[:t])
        price = max(self.min_price, price)
        price = min(self.max_price, price)
        return price

    def step(self, action):
        '''
            Parameters:
                action - a scalar
            
            Returns:
                obs - next observation
                reward - reward for time t
                done - whether to end the episode
        '''
        # Amount of shares bought so far
        total_bought = np.sum(self.Xs)
        constraint_violated = False # track constraint violations for large negative reward

        if self.agent_t == self.T - 1: # At the last timestep
            self.Xs[self.agent_t] = 1 - total_bought # Must end with exactly 1 share. # TODO: So must sell or buy
            self.done = True
        else:
            if total_bought + action < 0: # You can't sell shares you don't own
                self.Xs[self.agent_t] = 0
                constraint_violated = True
            else:
                self.Xs[self.agent_t] = action

        self.Ps[self.agent_t + 1] = self.get_price_at_t(self.agent_t)
        self.cost += self.Xs[self.agent_t] * self.Ps[self.agent_t + 1]
        
        if not constraint_violated:
            reward = np.dot(np.transpose(self.Ps), self.C) - self.cost # Provide reward at every timestep to avoid sparse reward & difficult training.
          # reward = self.Ps[self.agent_t + 1] * self.C[self.agent_t + 1] - self.Xs[self.agent_t] * self.Ps[self.agent_t + 1]
        else:
            reward = -100

        self.agent_t += 1
        
        return self._get_obs(), reward, self.done, {}

    def reset(self):
        self.agent_t = 0 # Agent timestep
        
        if self.principal is not None:
            self.C = self.principal.predict(np.concatenate([self.C, self.Ps, self.Xs]))[0]
        else:
            self.C = np.random.uniform(size=(self.T + 1))
        # Set first contract weight to 0 for P0
        self.C[0] = 0
        # Normalize contract
        self.C = self.C / np.sum(self.C)

        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        self.Xs = np.zeros(self.T) # Shares purchased at times 1,...,T
        self.epsilons = np.random.normal(scale=self.std, size=(self.T)) # Random noise for price model
        
        self.done = False
        self.cost = 0 # Cost of purchasing all the shares
        self.Ps[self.agent_t] = self.P0

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.C, self.Ps])

    def render(self):
        pass