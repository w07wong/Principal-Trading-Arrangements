import numpy as np
import argparse
from stable_baselines import PPO2, DDPG
from utils import get_scores

def run(args):
    principal_model = DDPG.load(args.principal_model)
    agent_model = PPO2.load(args.agent_model)

    agent_r, best_agent_r, diff_agent, principal_r, best_principal_r, diff_principal = get_scores(principal_model,
        agent_model, n_avg=args.n_avg, theta=args.theta, lam=args.lam, sigma=args.sigma, T=args.T)
    
    print('Average learned agent reward: {}, Variance: {}'.format(np.mean(agent_r), np.std(agent_r) ** 2))
    print('Average optimal agent reward: {}, Variance: {}'.format(np.mean(best_agent_r), np.std(best_agent_r) ** 2))
    print('Average difference: {}'.format(np.mean(diff_agent)))
    print('Average learned principal reward: {}, Variance: {}'.format(np.mean(principal_r), np.std(principal_r) ** 2))
    print('Average optimal principal reward: {}, Variance: {}'.format(np.mean(best_principal_r), np.std(best_principal_r) ** 2))
    print('Average difference: {}'.format(np.mean(diff_principal)))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta', type=float, default=1.0,
                        help="theta value for price model")
    parser.add_argument('--lam', type=float, default=1.0,
                        help="lambda value for price model")
    parser.add_argument('--sigma', type=float, default=1.0,
                        help="sigma value for price model")
    parser.add_argument('--T', type=int, default=10, help='timesteps')

    parser.add_argument('--n_avg', type=int, default=100,
                        help='number of times to simulate for evaluation')
    parser.add_argument('--agent_model', type=str, default=None, required=True,
                        help='path to trained agent model (defaults to none)')
    parser.add_argument('--principal_model', type=str, default=None, required=True,
                        help='path to trained principal model (defaults to none)')

    args = parser.parse_args()
    run(args)