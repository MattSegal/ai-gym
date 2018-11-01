"""
Player controls with arrow key
"""
KEY_MAP = {
   'a': 0,
   's': 1,
   'd': 2,
   'w': 3,
}

class PlayerAgent:

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
        self.observation = None
        self.chosen_action = None

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        chosen_action = None
        while chosen_action is None:
            choice = input('Action: ')
            chosen_action = KEY_MAP.get(choice)

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


    def finish_episode(self):
        return self.episode_return

