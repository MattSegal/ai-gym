"""
Tabular Temporal Difference solution to frozen lake,
using epsilon-greedy exploration
"""
import random
from base_agent import BaseAgent


class TDZeroAgent(BaseAgent):

    def start_environment(self, env):
        """
        Setup observation space
        """
        super().start_environment(env)
        self.values = {
            state: {action: 0.01 * random.random() for action in self.actions}
            for state in self.states
        }

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        super().start_episode()
        self.observation = None
        self.chosen_action = None
        self.reward = None

    def observe(self, observation):
        """
        Observe data from envrionment
        """
        # print('now in state', observation)
        self.prev_observation = self.observation
        self.observation = observation

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        if self.episodes < 1000:
            epsilon = 1
        else:
            epsilon = 1 / self.episodes**0.2

        # print(epsilon)
        if random.random() >= epsilon:
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

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        super().receive_reward(reward)
        # print('received reward', reward)
        reward *= 100

        prev_reward = self.reward
        self.reward = reward

        state = self.observation
        prev_state = self.prev_observation

        action = self.chosen_action
        prev_action = self.prev_action

        should_update_previous_action  = (
            prev_action is not None and
            prev_state is not None and
            prev_reward is not None
        )

        # print('prev state', prev_state, 'and prev action', prev_action)
        if should_update_previous_action:
            # print('updating state / action', prev_state, '/', prev_action, 'with reward', prev_reward)
            # print('estimate is for state / action', state, '/', action,':', self.values[state][action])
            # print('before: ', self.values[prev_state][prev_action])
            td_target = prev_reward + self.gamma * self.values[state][action]
            td_error = td_target - self.values[prev_state][prev_action]
            # print('TD target: ', td_target)
            # print('TD error:  ', td_error)
            self.values[prev_state][prev_action] += self.alpha * td_error
            # print('after:  ', self.values[prev_state][prev_action])

        # self.print_values()

    def print_values(self):
        print('STATE\t\tUP(3)\t\tDOWN(1)\t\tLEFT(0)\t\tRIGHT(2)')
        for state, actions in self.values.items():
            print('{}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}'.format(
                state,
                actions[3],
                actions[1],
                actions[0],
                actions[2],
            ))

    def finish_episode(self, final_observation):
        """
        Perform final update for end of episode
        Returns episode return
        """
        # Set value of all actions for terminal state to zero.
        self.observe(final_observation)
        self.values[final_observation] = {action: 0 for action in self.actions}
        # Perform final TD update.
        self.receive_reward(0)
        # self.print_values()
        return super().finish_episode(final_observation)

