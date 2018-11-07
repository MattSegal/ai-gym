"""
Player controls with arrow key
"""
from .base_agent import BaseAgent

KEY_MAP = {
   'a': 0,
   's': 1,
   'd': 2,
   'w': 3,
}

class PlayerAgent(BaseAgent):

    def get_next_action(self):
        """
        Select next action from action space using learned policy
        """
        chosen_action = None
        while chosen_action is None:
            choice = input('Action: ')
            chosen_action = KEY_MAP.get(choice)

        return chosen_action
