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
import gym

GYM_ENV = 'FrozenLake-v0'

def run_environment(agent, num_epsodes, max_steps):
    env = gym.make(GYM_ENV)
    agent.start_environment(env)
    for k in range(num_epsodes):
        agent.start_episode()
        observation = env.reset()
        for t in range(max_steps):
            # env.render()
            agent.observe(observation)
            action = agent.get_next_action()
            observation, reward, done, info = env.step(action)
            agent.receive_reward(reward)
            if done:
                # env.render()
                msg = 'Episode {} finished after {} timesteps'
                agent.finish_episode()
                print(msg.format(k, t + 1))
                break
