"""
Tabular Monte Carlo solution to frozen lake,
using epsilon-greedy exploration

I think I implemented this wrong somewhere
"""
import random

class MonteCarloAgent:

    def __init__(self, gamma):
        self.gamma = gamma
        self.episodes = 0

    def start_environment(self, env):
        """
        Setup observation space
        """
        self.states = [s for s in range(env.observation_space.n)]
        self.actions = [a for a in range(env.action_space.n)]
        self.visits = {
            state: {action: 0 for action in self.actions}
            for state in self.states
        }
        self.values = {
            state: {action: 0.001 * random.random() for action in self.actions}
            for state in self.states
        }

    def start_episode(self):
        """
        Reset rewards so that we can calculate return for this episode
        """
        self.episodes += 1
        self.rewards = []
        self.visited = []

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        epsilon = 1.0 / float(self.episodes)
        print(epsilon, self.episodes)
        if random.random() > epsilon:
            print('greedy')
            # Follow greedy policy
            state = self.last_observation
            best_action = None
            best_value = float('-inf')
            for possible_action in self.actions:
                if self.values[state][possible_action] > best_value:
                    best_value = self.values[state][possible_action]
                    best_action = possible_action

            chosen_action = best_action
        else:
            # Follow random policy
            print('random')
            chosen_action = random.choice(self.actions)

        self.visited.append((self.last_observation, chosen_action))
        self.visits[self.last_observation][chosen_action] += 1
        return chosen_action

    def observe(self, observation):
        """
        Observe data from envrionment
        """
        self.last_observation = observation

    def receive_reward(self, reward):
        """
        Keep track off all rewards
        """
        self.rewards.append(reward)

    def finish_episode(self):
        # Calculate episode returns from time T to 1
        num_timesteps = len(self.rewards)
        episode_returns = []
        for t in reversed(range(num_timesteps)):
            prev_reward = self.rewards[t - 1] if t < num_timesteps - 1 else 0
            reward = self.rewards[t]
            return_t = reward + self.gamma * prev_reward
            episode_returns.append(return_t)

        print('Final return of ', episode_returns[-1], '\n')

        # Run through states from time 0 to T
        for t in range(num_timesteps):
            state_t, action_t = self.visited[t]
            return_t = episode_returns[t]
            # Incrmentally update action-value function
            error_t =  return_t - self.values[state_t][action_t]
            self.values[state_t][action_t] += (1 / self.visits[state_t][action_t]) * error_t

        print(self.values)
