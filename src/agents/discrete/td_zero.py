"""
Tabular Temporal Difference solution
using epsilon-greedy exploration
"""
import random
from .base_agent import BaseAgent


class TDZeroAgent(BaseAgent):

    ACTION_NAMES = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    def start_environment(self, env):
        """
        Setup observation space
        """
        super().start_environment(env)
        # Initialize state-action values somewhat optimistically
        self.values = {
            state: {action: 0.5 for action in self.actions}
            for state in self.states
        }

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        super().start_episode()
        self.obs, self.prev_obs = None, None
        self.action, self.prev_action = None, None
        self.reward = None

    def observe(self, new_obs):
        """
        Observe data from envrionment
        """
        self.obs, self.prev_obs = new_obs, self.obs

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        epsilon = max(0.05, 1 / self.episodes**0.4)
        if random.random() >= epsilon:
            # Follow greedy policy
            state = self.obs
            action = self.get_action_greedily(state)
        else:
            # Follow random policy
            action = self.get_action_randomly()

        self.action, self.prev_action = action, self.action
        return action

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        super().receive_reward(reward)
        self.reward, prev_reward = reward, self.reward
        state, prev_state = self.obs, self.prev_obs
        action, prev_action = self.action, self.prev_action
        should_update  = not any([x is None for x in (prev_action, prev_state, prev_reward)])

        prev_val = self.values[prev_state][prev_action] if should_update else None
        td_target = None
        td_error = None

        if should_update:
            td_target = prev_reward + self.gamma * self.values[state][action]
            td_error = td_target - self.values[prev_state][prev_action]
            self.values[prev_state][prev_action] += self.alpha * td_error

        # print('\nUpdating:\t', prev_state, '/', prev_action)
        # print('Target:\t\t', state, '/', action)
        # print('Reward:\t\t', prev_reward)
        # print('TD target:\t', td_target)
        # print('TD error:\t', td_error)
        # print('TD update:\t', self.alpha * td_error if td_error else None)
        # print('Prev value:\t', prev_val)
        # print('New value\t', self.values[prev_state][prev_action] if should_update else None)

    def finish_episode(self, final_obs):
        """
        Perform final update for end of episode
        Returns episode return
        """
        # Set value of all actions for terminal state to zero.
        # And then take a dummy action
        self.observe(final_obs)
        self.values[final_obs] = {action: 0 for action in self.actions}
        self.get_next_action()
        # Perform final TD update.
        self.receive_reward(0)
        return super().finish_episode(final_obs)

