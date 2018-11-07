"""
Check that agents learn  basic MDPs
"""
from gym.envs.toy_text import discrete

from .. import agents

START = 0
END = 1
LEFT = 0
RIGHT = 1


def assert_equalish(expected, actual, error, message):
    assert expected - error <= actual <= expected + error, message


class TwoDoorsTestEnv(discrete.DiscreteEnv):
    """
    There are two doors (LEFT, RIGHT) which both lead to a terminal state.
    LEFT produces reward 1 with some probability [0, 1].
    RIGHT produces a reward 1 with some probablility [0, 1].
    """
    def __init__(self, p_left, p_right):
        assert 0 <= p_left <= 1
        assert 0 <= p_right <= 1
        number_states = 2
        number_actions = 2
        initial_state_distribution = [0]
        transitions = {
            # States
            START: {
                # Actions
                LEFT: [
                    # probability, nextstate, reward, done
                    (p_left, END, 1, False),
                    (1- p_left, END, 0, False),
                ],
                RIGHT: [
                    (p_right, 1, 1, False),
                    (1- p_right, 1, 0, False),
                ],
            },
            END: {
                LEFT: [(1, END, 0, True)],
                RIGHT: [(1, END, 0, True)],
            },
        }
        super().__init__(number_states, number_actions, transitions, initial_state_distribution)


def run_two_doors(agent, env, num_episodes):
    agent.start_environment(env)
    for k in range(num_episodes):
        agent.start_episode()
        observation = env.reset()
        for t in range(2):
            agent.observe(observation)
            action = agent.get_next_action()
            observation, reward, done, info = env.step(action)
            agent.receive_reward(reward)
            if done:
                agent.finish_episode(observation)
                break


def test_two_doors_deterministic__td_zero():
    """
    Basic test to see if agent learns to open the correct doors.
    """
    NUM_EPISODES = 1000
    GAMMA, ALPHA = 1, 0.4
    ERROR = 0.05
    test_cases = [
        {
            'name': 'Agent should always go left',
            'p_left': 1,
            'p_right': 0,
            'expected':  {START: {LEFT: 1, RIGHT: 0}, END: {LEFT: 0, RIGHT: 0}}
        },
        {
            'name': 'Agent should always go right',
            'p_left': 0,
            'p_right': 1,
            'expected':  {START: {LEFT: 0, RIGHT: 1}, END: {LEFT: 0, RIGHT: 0}}
        },
        {
            'name': 'Agent choose randomly (both 1)',
            'p_left': 1,
            'p_right': 1,
            'expected':  {START: {LEFT: 1, RIGHT: 1}, END: {LEFT: 0, RIGHT: 0}}
        },
        {
            'name': 'Agent should choose randomly (both 0)',
            'p_left': 0,
            'p_right': 0,
            'expected':  {START: {LEFT: 0, RIGHT: 0}, END: {LEFT: 0, RIGHT: 0}}
        },
                {
            'name': 'Agent should choose randomly (both 0.5)',
            'p_left': 0.5,
            'p_right': 0.5,
            'expected':  {START: {LEFT: 0.5, RIGHT: 0.5}, END: {LEFT: 0, RIGHT: 0}}
        },
        {
            'name': 'Agent should prefer left',
            'p_left': 0.7,
            'p_right': 0.3,
            'expected':  {START: {LEFT: 0.7, RIGHT: 0.3}, END: {LEFT: 0, RIGHT: 0}}
        },
    ]

    for test_case in test_cases:
        agent = agents.discrete.TDZeroAgent(GAMMA, ALPHA)
        env = TwoDoorsTestEnv(p_left=test_case['p_left'], p_right=test_case['p_right'])
        agent.start_environment(env)
        run_two_doors(agent, env, NUM_EPISODES)
        for state, action in ((START, LEFT), (START, RIGHT), (END, LEFT), (END, RIGHT)):
            try:
                assert_equalish(
                    expected=test_case['expected'][state][action],
                    actual=agent.values[state][action],
                    error=ERROR,
                    message='{}: {}-{}'.format(test_case['name'], state, action)
                )
            except AssertionError:
                agent.print_values()
                raise
