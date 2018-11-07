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

"""
import time
from collections import deque

import gym

# GYM_ENV = 'FrozenLake-v0'
GYM_ENV = 'FrozenLakeNotSlippery-v0'

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

REPORT_PERIOD = 1000
SOLVED = 1

def display(env, agent, t, k):
    print('EPISODE:\t', k)
    print('TIME:\t\t', t)
    print('EPSILON:\t', max(0.1, 1 / agent.episodes**0.5))
    agent.print_values()
    env.render()
    # time.sleep(1)
    input()


def run_environment(agent, num_episodes, max_steps, render=False):
    env = gym.make(GYM_ENV)
    agent.start_environment(env)
    returns = deque(maxlen=100)

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
            print('Average return of ', average_return, 'in episode', k + 1)
            if average_return >= SOLVED:
                print('Solved!')
                break

