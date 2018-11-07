"""
Tabular TD-Lambda solution to frozen lake,
using epsilon-greedy exploration
"""
import random
from .base_agent import BaseAgent


class TDLambdaAgent(BaseAgent):

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
        self.eligibility = {
            state: {action: 0 for action in self.actions}
            for state in self.states
        }

    def observe(self, observation):
        """
        Observe data from envrionment
        """
        self.prev_observation = self.observation
        self.observation = observation

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        if self.episodes < 10000:
            epsilon = 1
        else:
            epsilon = 1 / self.episodes**0.2

        if random.random() >= epsilon:
            # Follow greedy policy
            state = self.observation
            best_action = None
            best_value = float('-inf')
            for possible_action in self.actions:
                action_value = self.values[state][possible_action]
                if action_value > best_value:
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
        # https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2007-49.pdf
        super().receive_reward(reward)
        reward *= 100
        self.reward, prev_reward = reward, self.reward
        state, prev_state = self.observation, self.prev_observation
        action, prev_action = self.chosen_action, self.prev_action
        should_update_previous_action  = (
            prev_action is not None and
            prev_state is not None and
            prev_reward is not None
        )
        if should_update_previous_action:
            # Calculate TD error between previous state and current state
            td_target = prev_reward + self.gamma * self.values[state][action]
            td_error = td_target - self.values[prev_state][prev_action]
            for s in self.states:
                for a in self.actions:
                    # Update eligibility trace
                    self.eligibility[s][a] *= self.gamma * self.lambd
                    if s == prev_state and a == prev_action:
                        self.eligibility[s][a] += 1

                    # Update action-value function accordind to eligibility
                    update = self.alpha * td_error * self.eligibility[s][a]
                    self.values[s][a] += update

    def print_values(self):
        print('STATE\t\tUP(3)\t\tDOWN(1)\t\tLEFT(0)\t\tRIGHT(2)')
        for state, actions in self.eligibility.items():
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

