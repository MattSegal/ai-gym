"""
Run an agent in the Frozen Lake environment

    Average returns

    Random agent
        slippery            0.013
        solid               0.013

    TD Agent (gamma / alpha)

        slippery 0.5 / 0.5  0.35 (should be 1!)

    FrozenLake-v0 is considered "solved" when the agent obtains an
    average reward of at least 0.78 over 100 consecutive episodes.
    I can be solved in as few as 85 episodes.

"""
from environment import run_environment
from monte_carlo import MonteCarloAgent
from td_lambda import TDLambdaAgent
from td_zero import TDZeroAgent
from player import PlayerAgent
from random_agent import RandomAgent

NUM_EPISODES = 100 * 1000
MAX_STEPS = 100
GAMMA = 0.9 # How valuable is the expectation of future rewards?
ALPHA = 0.1  # How much do you trust this sample?
LAMBDA = 0.5  # How much do we prefer accuracy/high bias vs variance/low bias
              # At 1 we are doing Monte Carlo
              # At 0 we are doing TD(0)

# agent = MonteCarloAgent(ALPHA)
# agent = TDZeroAgent(GAMMA, ALPHA)
agent = TDLambdaAgent(GAMMA, ALPHA, LAMBDA)
# agent = PlayerAgent()
# agent = RandomAgent()

run_environment(agent, NUM_EPISODES, MAX_STEPS, render=False)



