"""
Base agent for running in a discrete environment
"""
import random


class BaseAgent:

    def __init__(self, gamma=0, alpha=0, lambd=0):
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.episodes = 0

    def start_environment(self, env):
        """
        Setup observation space
        """
        self.states = [s for s in range(env.observation_space.n)]
        self.actions = [a for a in range(env.action_space.n)]

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        self.episodes += 1
        self.episode_return = 0

    def observe(self, observation):
        """
        Observe data from envrionment
        """
        pass

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        return self.get_action_randomly()

    def get_action_greedily(self, state):
        best_action = None
        best_value = float('-inf')
        for possible_action in self.actions:
            action_value = self.values[state][possible_action]
            if action_value > best_value:
                best_value = self.values[state][possible_action]
                best_action = possible_action

        return best_action

    def get_action_randomly(self):
        return random.choice(self.actions)

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        self.episode_return += reward

    def print_values(self):
        """
        Print tabular value function as a table for debugging
        """
        values = getattr(self, 'values', {})
        action_names = getattr(self, 'ACTION_NAMES')
        if action_names:
            print('\t\t' + '\t\t'.join(action_names))
        print('\t\t'.join(['STATE', *[str(a) for a in self.actions]]))
        for state, actions in values.items():
            sorted_actions = [actions[a] for a in self.actions]
            print('\t\t'.join(['{}'] + ['{:.2f}' for _ in self.actions]).format(state, *sorted_actions))

    def finish_episode(self, final_observation):
        """
        Returns episode return
        """
        return self.episode_return

