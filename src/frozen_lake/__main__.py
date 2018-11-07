"""
Frozen Lake v0 Problem

https://gym.openai.com/envs/FrozenLake-v0/

    The agent controls the movement of a character in a grid world.
    Some tiles of the grid are walkable, and others lead to the agent
    falling into the water. Additionally, the movement direction of
    the agent is uncertain and only partially depends on the chosen direction.
    The agent is rewarded for finding a walkable path to a goal tile.

    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)

    In a 4x4 grid the observation is an integer {0...15}, which
    represents the position of the agent and is the state.

    FrozenLake-v0 is considered "solved" when the agent obtains an
    average reward of at least 0.78 over 100 consecutive episodes.
    I can be solved in as few as 85 episodes.

"""
import time
from collections import deque

import gym
from gym.envs.registration import register

from .. import agents

NUM_EPISODES = 100 * 1000
MAX_STEPS = 100
GAMMA = 0.9 # How valuable is the expectation of future rewards?
ALPHA = 0.2  # How much do you trust this sample?
LAMBDA = 0.5  # How much do we prefer accuracy/high bias vs variance/low bias
              # At 1 we are doing Monte Carlo
              # At 0 we are doing TD(0)

# GYM_ENV = 'FrozenLake-v0'
GYM_ENV = 'FrozenLakeNotSlippery-v0'
REPORT_PERIOD = 1000
SOLVED = 0.98

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)


def main():
    # agent = MonteCarloAgent(ALPHA)
    agent = agents.discrete.TDZeroAgent(GAMMA, ALPHA)
    # agent = TDLambdaAgent(GAMMA, ALPHA, LAMBDA)
    # agent = PlayerAgent()
    # agent = RandomAgent()
    run_environment(agent, NUM_EPISODES, MAX_STEPS, render=False)


def run_environment(agent, num_episodes, max_steps, render=False):
    env = gym.make(GYM_ENV)
    agent.start_environment(env)
    returns = deque(maxlen=1000)

    for k in range(num_episodes):
        agent.start_episode()
        observation = env.reset()
        for t in range(max_steps):
            if render:
                display(env, agent, t, k)

            agent.observe(observation)
            action = agent.get_next_action()
            observation, reward, done, info = env.step(action)

            if render:
                # Clear screen
                print(chr(27) + "[2J")

            agent.receive_reward(reward)
            if done:
                if render:
                    display(env, agent, t, k)

                episode_return = agent.finish_episode(observation)
                returns.append(episode_return)
                break

        if (k + 1) % REPORT_PERIOD == 0:
            average_return = sum(returns) / float(len(returns))
            # agent.print_values()
            print('Average return of ', average_return, 'in episode', k + 1)
            if average_return >= SOLVED:
                print('Solved!')
                break


def display(env, agent, t, k):
    # print('EPISODE:\t', k)
    # print('TIME:\t\t', t)
    # print('EPSILON:\t', max(0.1, 1 / agent.episodes**0.5))
    # agent.print_values()
    env.render()
    # time.sleep(1)
    input()


main()
