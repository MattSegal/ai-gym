from environment import run_environment
from monte_carlo import MonteCarloAgent
from td_zero import TDZeroAgent


NUM_EPISODES = 10000
MAX_STEPS = 50
GAMMA = 0.9
ALPHA = 0.9

agent = TDZeroAgent(GAMMA, ALPHA)
run_environment(agent, NUM_EPISODES, MAX_STEPS)
