"""
Tabular Monte Carlo solution to frozen lake,
using epsilon-greedy exploration

I've implemented this wrong somewhere because it doesn't work.
"""
import random
from .base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):

    ACTION_NAMES = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    def start_environment(self, env):
        """
        Setup observation space
        """
        super().start_environment(env)
        # How many times we've seen a state-action pair
        self.visits = {
            state: {action: 0 for action in self.actions}
            for state in self.states
        }
        # The value of a given state action pair - initialize optimistically
        self.values = {
            state: {action: 0.5 for action in self.actions}
            for state in self.states
        }

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        super().start_episode()
        self.obs = None
        self.rewards = []
        self.visited = []

    def observe(self, obs):
        """
        Observe data from envrionment
        """
        self.obs = obs

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        state = self.obs
        epsilon = max(0.05, 1 / self.episodes**0.4)
        if random.random() >= epsilon:
            # Follow greedy policy
            action = self.get_action_greedily(state)
        else:
            # Follow random policy
            action = self.get_action_randomly()

        self.visited.append((state, action))
        self.visits[state][action] += 1
        return action


    def receive_reward(self, reward):
        """
        Keep track off all rewards for this episode
        """
        self.rewards.append(reward)

    def finish_episode(self, final_obs):
        # Calculate episode returns from time T to 1
        num_timesteps = len(self.rewards)
        episode_returns = []
        for t in reversed(range(num_timesteps)):
            prev_reward = self.rewards[t - 1] if t < num_timesteps - 1 else 0
            reward = self.rewards[t]
            return_t = reward + self.alpha * prev_reward
            episode_returns.append(return_t)

        # Run through states from time 0 to T
        for t in range(num_timesteps):
            state_t, action_t = self.visited[t]
            return_t = episode_returns[t]
            # Incrmentally update action-value function
            error_t =  return_t - self.values[state_t][action_t]
            self.values[state_t][action_t] += (1 / self.visits[state_t][action_t]) * error_t

        return episode_returns[-1]
