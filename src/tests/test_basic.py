"""
Veify that agents learn basic MDPs
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


def run_environment(agent, env, num_episodes):
    """
    Run the agent through the environment for the given number of episodes.
    """
    agent.start_environment(env)
    for k in range(num_episodes):
        agent.start_episode()
        observation = env.reset()
        # Run the episode for at most 100 timesteps
        for t in range(100):
            agent.observe(observation)
            action = agent.get_next_action()
            observation, reward, done, info = env.step(action)
            agent.receive_reward(reward)
            if done:
                agent.finish_episode(observation)
                break


def assert_test_case(agent, test_case, num_episodes, error):
    """
    Assert that the agent passes the test case within the given margin of error.
    """
    env = TwoDoorsTestEnv(p_left=test_case['p_left'], p_right=test_case['p_right'])
    agent.start_environment(env)
    run_environment(agent, env, num_episodes)
    for state, action in ((START, LEFT), (START, RIGHT), (END, LEFT), (END, RIGHT)):
        try:
            assert_equalish(
                expected=test_case['expected'][state][action],
                actual=agent.values[state][action],
                error=error,
                message='{}: {}-{}'.format(test_case['name'], state, action)
            )
        except AssertionError:
            agent.print_values()
            raise


def assert_two_doors_deterministic(get_agent, num_episodes, error):
    """
    Basic test to see if agent learns to open the correct doors
    in a fully deterministic environment.
    """
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
    ]

    for test_case in test_cases:
        agent = get_agent()
        assert_test_case(agent, test_case, num_episodes, error)


def assert_two_doors_stochastic(get_agent, num_episodes, error):
    """
    Basic test to see if agent learns to open the correct doors
    in a stochastic environment.
    """
    test_cases = [
        {
            'name': 'Agent should prefer left',
            'p_left': 0.7,
            'p_right': 0.3,
            'expected':  {START: {LEFT: 0.7, RIGHT: 0.3}, END: {LEFT: 0, RIGHT: 0}}
        },
        {
            'name': 'Agent should prefer right',
            'p_left': 0.3,
            'p_right': 0.7,
            'expected':  {START: {LEFT: 0.3, RIGHT: 0.7}, END: {LEFT: 0, RIGHT: 0}}
        },
    ]
    for test_case in test_cases:
        agent = get_agent()
        assert_test_case(agent, test_case, num_episodes, error)


def test_two_doors_deterministic__monte_carlo():
    """
    Ensure Monte Carlo works in a deterministic 2 door env.
    """
    NUM_EPISODES = 100
    ALPHA = 1
    ERROR = 0.01
    def get_agent():
        return agents.discrete.MonteCarloAgent(ALPHA)

    assert_two_doors_deterministic(get_agent, NUM_EPISODES, ERROR)


# def test_two_doors_stochastic__td_lambda():
#     """
#     Ensure TD Lambda works in a stochastic 2 door env.
#     """
#     NUM_EPISODES = 10000
#     GAMMA, ALPHA, LAMBDA = 1, 0.05, 0.5
#     ERROR = 0.2  # We're being pretty generous here
#     def get_agent():
#         return agents.discrete.TDLambdaAgent(GAMMA, ALPHA, LAMBDA)

#     assert_two_doors_stochastic(get_agent, NUM_EPISODES, ERROR)


def test_two_doors_deterministic__td_zero():
    """
    Ensure TD Zero works in a deterministic 2 door env.
    """
    NUM_EPISODES = 100
    GAMMA, ALPHA = 1, 1
    ERROR = 0.01
    def get_agent():
        return agents.discrete.TDZeroAgent(GAMMA, ALPHA)

    assert_two_doors_deterministic(get_agent, NUM_EPISODES, ERROR)


def test_two_doors_stochastic__td_zero():
    """
    Ensure TD Zero works in a stochastic 2 door env.
    """
    NUM_EPISODES = 10000
    GAMMA, ALPHA = 1, 0.05
    ERROR = 0.2  # We're being pretty generous here
    def get_agent():
        return agents.discrete.TDZeroAgent(GAMMA, ALPHA)

    assert_two_doors_stochastic(get_agent, NUM_EPISODES, ERROR)


def test_two_doors_deterministic__td_lambda():
    """
    Ensure TD Lambda works in a deterministic 2 door env.
    """
    NUM_EPISODES = 100
    GAMMA, ALPHA, LAMBDA = 1, 1, 0.5
    ERROR = 0.01
    def get_agent():
        return agents.discrete.TDLambdaAgent(GAMMA, ALPHA, LAMBDA)

    assert_two_doors_deterministic(get_agent, NUM_EPISODES, ERROR)


def test_two_doors_stochastic__td_lambda():
    """
    Ensure TD Lambda works in a stochastic 2 door env.
    """
    NUM_EPISODES = 10000
    GAMMA, ALPHA, LAMBDA = 1, 0.05, 0.5
    ERROR = 0.2  # We're being pretty generous here
    def get_agent():
        return agents.discrete.TDLambdaAgent(GAMMA, ALPHA, LAMBDA)

    assert_two_doors_stochastic(get_agent, NUM_EPISODES, ERROR)
