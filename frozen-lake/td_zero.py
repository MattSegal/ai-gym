"""
Tabular Monte Carlo solution to frozen lake,
using epsilon-greedy exploration
"""
import random
import pprint

class TDZeroAgent:

    def __init__(self, gamma, alpha):
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = 0

    def start_environment(self, env):
        """
        Setup observation space
        """
        self.states = [s for s in range(env.observation_space.n)]
        self.actions = [a for a in range(env.action_space.n)]
        self.values = {
            state: {action: 0.0001 * random.random() for action in self.actions}
            for state in self.states
        }

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        self.episodes += 1
        self.episode_return = 0
        self.observation = None
        self.chosen_action = None

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        epsilon = 1.0 / float(self.episodes**0.3)
        if random.random() > epsilon:
            # Follow greedy policy
            state = self.observation
            best_action = None
            best_value = float('-inf')
            for possible_action in self.actions:
                if self.values[state][possible_action] > best_value:
                    best_value = self.values[state][possible_action]
                    best_action = possible_action

            chosen_action = best_action
        else:
            # Follow random policy
            chosen_action = random.choice(self.actions)

        self.prev_action = self.chosen_action
        self.chosen_action = chosen_action

        return chosen_action

    def observe(self, observation):
        """
        Observe data from envrionment
        """
        self.prev_observation = self.observation
        self.observation = observation

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        self.episode_return += reward

        state = self.observation
        prev_state = self.prev_observation

        action = self.chosen_action
        prev_action = self.prev_action

        if prev_action and prev_state:
            td_target = reward + self.gamma * self.values[state][action]
            td_error = td_target - self.values[prev_state][prev_action]
            self.values[prev_state][prev_action] += self.alpha * td_error

    def finish_episode(self):
        print('Final return of ', self.episode_return, '\n')
        # pprint.pprint(self.values)

