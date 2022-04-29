# Principal-Trading-Arrangements

## Train Models
To run experiments for the reward maximization alternating training framework, run ```train_new.py```.

To run experiments for the CVaR minimization alternating training framework, run ```train_cvar.py```.

Both alternate trading frameworks run TD3 for the principal and PPO2 for the agent. Pass in any experimental parameters using the flags provided in the files.

For the nonliear contracts, the training scripts can be found in ```notebooks/nonlinear-contracts.ipynb```.

## Gym Environments
The agent gym environment for reward maximization is contained in ```env/agent_env.py``` under the class AgentEnv. The agent gym environment for CVaR minimization is contained in ```env/agent_env_cvar.py``` under the class AgentEnv_CVaR. We approximate minimizing CVaR by providing the agent the CVaR of possible rewards at every time step.

The principal uses the same gym environment for both reward maximization and CVaR minimization as we only minimize the CVaR of the agent since the principal defines each game. This can be found under ```env/principal_env.py```.

We also experimnted with a multi-agent gym environment during initial experiments but found that splitting the problem into two separate environments was easier to work with. This multi-agent environment can be found under ```env/market_env.py``` and a demo of how to interact with it is in ```train.py```.


## Evaluation
Once you have trained agents, you can run ```eval.py``` to get the average performance of your learned models verus the analytic solution provided in [Baldauf et al. 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3778956).

Evaluation requires providing paths to the trained agent and principal models as arguments. Our trained models can be found here: [Principal](https://drive.google.com/file/d/18wmiDD5vzuyz0fEfpkpjgxE2k8fopyAI/view?usp=sharing), [Agent](https://drive.google.com/file/d/1AzBdEFF8C37UhXgw6NvnOtN3hmeW2zI6/view?usp=sharing).