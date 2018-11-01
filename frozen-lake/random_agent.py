"""
Totally random agent
"""
import random

class RandomAgent:

    def start_environment(self, env):
        """
        Setup observation space
        """
        self.actions = [a for a in range(env.action_space.n)]

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        self.episode_return = 0

    def get_next_action(self):
        """
        Select next action randomly
        """
        return random.choice(self.actions)

    def observe(self, observation):
        pass

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        self.episode_return += reward

    def finish_episode(self):
        """
        Returns episode return
        """
        return self.episode_return

