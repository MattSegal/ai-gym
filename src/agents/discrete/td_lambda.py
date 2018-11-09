"""
Tabular TD-Lambda solution to frozen lake,
using epsilon-greedy exploration
"""
import random
from .base_agent import BaseAgent


class TDLambdaAgent(BaseAgent):

    ACTION_NAMES = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    def start_environment(self, env):
        """
        Setup observation space
        """
        super().start_environment(env)
        # Initialize state-action values somewhat optimistically
        self.values = {
            state: {action: 1 for action in self.actions}
            for state in self.states
        }

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        super().start_episode()
        super().start_episode()
        self.obs, self.prev_obs = None, None
        self.action, self.prev_action = None, None
        self.reward = None
        self.eligibility = {
            state: {action: 0 for action in self.actions}
            for state in self.states
        }

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
            best_action = None
            best_value = float('-inf')
            for possible_action in self.actions:
                action_value = self.values[state][possible_action]
                if action_value > best_value:
                    best_value = self.values[state][possible_action]
                    best_action = possible_action

            action = best_action
        else:
            # Follow random policy
            action = random.choice(self.actions)

        self.action, self.prev_action = action, self.action
        return action

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        # https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2007-49.pdf
        super().receive_reward(reward)
        self.reward, prev_reward = reward, self.reward
        state, prev_state = self.obs, self.prev_obs
        action, prev_action = self.action, self.prev_action
        should_update  = not any([x is None for x in (prev_action, prev_state, prev_reward)])

        if should_update:
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

