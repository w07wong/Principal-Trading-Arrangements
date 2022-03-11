import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = DummyVecEnv([lambda: MarketEnv(theta=1, lam=1, sigma=1)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200000)

for i in range(100):
    obs = env.reset()
    actions_arr, rewards_arr = [], []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        actions_arr.append(action[0][-1])
        rewards_arr.append(rewards[0])
        # env.render()

        if done:
            print(f"Actions: \n{actions_arr}\n Rewards: \n{rewards_arr}\n")
            break