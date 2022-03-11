import gym
from gym import spaces
import numpy as np

'''
    Market environment
'''
class MarketEnv(gym.Env):
    def __init__(self,
                 theta,
                 lam,
                 sigma,
                 principal_min_action=0, # Should experiment with this value. 1/T is just the mean.
                 principal_max_action=1, # Should experiment with this value. 1/T is just the mean.
                 agent_min_action=0,
                 agent_max_action=1,
                 min_price=0,
                 max_price=1000, # Should experiment with this value.
                 P0=100,
                 T=30,
                 N=10000,
                 multi_agent=False):
        '''
            Parameters:
                theta - the temporary price impact parameter.
                lam - the permanent price impact parameter.
                sigma - the standard devation of the noise at each time period.
                principal_min_action - the minimum value to affect the principal's contract function.
                principal_max_action - the maximum value to affect the principal's contract function.
                agent_min_action - minimum number of shares the agent can buy at a timestep.
                agent_max_action - maximum number of shares the agent can buy at a timestep.
                min_price - minimum price, used to bound observation space and price model
                max_price - maximum price, used to bound observation space and price model.
                T - the number of time periods to complete trading.
                P0 - the initial price.
                multi_agent - if True, environment uses two different agent and observation spaces
        '''
        super(MarketEnv, self).__init__()
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
        self.P0 = P0
        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        self.multi_agent = multi_agent

        self.possible_agents = ['principal', 'agent']

        # We will treat the contract as some T+1-length vector, one value for each price timestep.
        self._principal_action_space = spaces.Box(
            low=principal_min_action, high=principal_max_action, shape=(T + 1,), dtype=np.float32
        )
        # The agent can buy maximum 1 share, minimum 0 shares at each timestep.
        self._agent_action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        if multi_agent:
            self.action_spaces = {
                'principal': self._principal_action_space,
                'agent': self._agent_action_space
            }
        else:
            # In this case, contract values are between 0 and 1. Can figure out how to get different bounds for first T+1 #s (the contract values)
            self.action_space = spaces.Box(low=0, high=1, shape=(T + 2,), dtype=np.float32)

        # The principal's state is the previous contract and the reward induced by C.
        self._principal_observation_space = spaces.Box(
            low=principal_min_action, high=principal_max_action, shape=(T + 1,), dtype=np.float32
        )
        self._agent_observation_space = spaces.Box(
            low=min(principal_min_action, min_price), high=max(principal_max_action, max_price), shape=(2 * T + 2,), dtype=np.float32)

        if multi_agent:
            self.observation_spaces = {
                'principal': self._principal_observation_space,
                'agent': self._agent_observation_space
            }
        else:
            # If not multiagent, we will just provide the contract and prices
            self.observation_space = self._agent_observation_space

        self.reset()

    def select_player(self):
        # Should modify. For now, it's just picking principal if global timestep = 0, agent otherwise.
        if self.t == 0:
            return self.possible_agents[0]
        return self.possible_agents[1]

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
        player = self.select_player()

        if player == 'principal':
            if self.multi_agent:
                self.C = action
            else:
                self.C = action[:-1]
            self.principal_t += 1 # Does nothing for now.
            reward = 0 # Not sure what reward to give for single-agent RL setting
        elif player == 'agent':
            if self.agent_t == self.T - 1: # At the last timestep
                self.Xs[self.agent_t] = 1 - np.sum(self.Xs) # Must end with exactly 1 share.
                self.done = True
            else:
                if self.multi_agent:
                    self.Xs[self.agent_t] = action
                else:
                    self.Xs[self.agent_t] = action[-1]

            if np.sum(self.Xs == 1):
                self.done = True
            
            self.Ps[self.agent_t + 1] = self.get_price_at_t(self.agent_t)
            self.cost += self.Xs[self.agent_t] * self.Ps[self.agent_t + 1]
            
            self.agent_t += 1
            reward = np.dot(np.transpose(self.Ps), self.C) - self.cost # Provide reward at every timestep to avoid sparse reward & difficult training.

        if self.done and self.multi_agent:
            # Concatenate principal reward with agent reward to be able to provide principal with a reward signal
            reward = (-np.dot(np.transpose(self.Ps), self.C), reward)

        self.t += 1
        next_player = self.select_player()
        
        return self._get_obs(next_player), reward, self.done, {}
    
    def reset(self):
        self.observations = {player: self._get_obs(player) for player in self.possible_agents}
        
        self.Ps = np.zeros(self.T + 1) # Will fill with prices encountered in the market at times 0,...,T
        self.Xs = np.zeros(self.T) # Shares purchased at times 1,...,T
        self.epsilons = np.random.normal(scale=self.std, size=(self.T)) 
        
        self.t = 0 # Global current timestep
        self.principal_t = 0 # Principal current timestep
        self.agent_t = 0 # Agent current timestep
        
        self.done = False
        self.cost = 0 # Cost of purchasing all the shares
        self.Ps[self.t] = self.P0

        return self._get_obs(self.select_player)

    def _get_obs(self, player=None):
        if self.multi_agent:
          if player is None:
              return np.concatenate([self.observations[player] for player in self.possible_agents])
          elif player == 'principal':
              return self.C
          elif player == 'agent':
              return np.concatenate([self.C, self.Ps])
        else:
          return np.concatenate([self.C, self.Ps])
