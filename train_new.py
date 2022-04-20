import os
import argparse
import numpy as np
import gym

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO2, SAC
from stable_baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from stable_baselines.td3 import TD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import DDPG
from stable_baselines.common.policies import FeedForwardPolicy
from env.agent_env import AgentEnv
from env.principal_env import PrincipalEnv

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

from stable_baselines.ddpg import MlpPolicy

def run(args):
    principal_model, agent_model = args.principal_model, args.agent_model
    for _ in range(args.n_games):
        # Create the agent environment, passing in a pretrained principal model.
        agent_env = make_vec_env(lambda: AgentEnv(principal_model, theta=args.theta, lam=args.lam, sigma=args.sigma, T=args.T), n_envs=96)

        # Define agent model to train.
        agent_model = PPO2(CustomPolicy, agent_env, verbose=1, lam=0.8, gamma=0.99, learning_rate=agent_lr, n_steps=96)

        # Train agent model.
        agent_model.learn(total_timesteps=args.agent_timesteps)

        # Create gym environments, passing in the pretrained agent.
        principal_env =  make_vec_env(lambda: PrincipalEnv(agent_model, agent_env, theta=args.theta, lam=args.lam, sigma=args.sigma, T=args.T), n_envs=1)

        # Add some action noise for TD3 algorithm which trains principal.
        principal_n_actions = principal_env.action_space.shape[-1]
        principal_action_noise = NormalActionNoise(mean=np.zeros(principal_n_actions), sigma=0.5 * np.ones(principal_n_actions))

        # Define principal model to train.
        # principal_model = TD3(MlpPolicy, principal_env, verbose=1, learning_rate=1e-3, action_noise=principal_action_noise)
        principal_model = DDPG(MlpPolicy, principal_env, verbose=1, param_noise=None, action_noise=principal_action_noise)

        # Train principal model.
        principal_model.learn(total_timesteps=args.principal_timesteps)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    agent_model.save(f'{args.save_dir}/agent')
    principal_model.save(f'{args.save_dir}/principal')
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta', type=float, default=1.0,
                        help="theta value for price model")
    parser.add_argument('--lam', type=float, default=1.0,
                        help="lambda value for price model")
    parser.add_argument('--sigma', type=float, default=1.0,
                        help="sigma value for price model")
    parser.add_argument('--T', type=int, default=10, help='timesteps')

    parser.add_argument('--agent_timesteps', type=int, default=1200000,
                        help='number of timesteps for training the agent')
    parser.add_argument('--principal_timesteps', type=int, default=10000,
                        help='number of timesteps for training the principal')
    parser.add_argument('--n_games', type=int, default=3,
                        help='number of times to train the agent and principal each')

    parser.add_argument('--agent_model', type=str, default=None, 
                        help='path to trained agent model (defaults to none)')
    parser.add_argument('--principal_model', type=str, default=None, 
                        help='path to trained principal model (defaults to none)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='path to save models to')

    args = parser.parse_args()


    # Set custom learning rate schedule for agent.
    agent_lr = PiecewiseSchedule([(0, 7e-3), (100000, 1e-4), (1000000, 1e-6), (args.agent_timesteps, 1e-10)]).value
    args.agent_lr = agent_lr

    # Set custom learning rate schedule for principal.
    principal_lrs = np.linspace(1e-3, 1e-6, args.principal_timesteps)
    principal_lr = lambda step: principal_lrs[step]
    args.principal_lr = principal_lr

    run(args)