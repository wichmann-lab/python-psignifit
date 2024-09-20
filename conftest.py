import numpy as np
import pytest


# add a commandline option to pytest
def pytest_addoption(parser):
    """Add random seed option to py.test.
    """
    parser.addoption('--seed', dest='seed', type=int, action='store',
                     help='set random seed')


# configure pytest to automatically set the rnd seed if not passed on CLI
def pytest_configure(config):
    seed = config.getvalue("seed")
    # if seed was not set by the user, we set one now
    if seed is None or seed == ('NO', 'DEFAULT'):
        config.option.seed = int(np.random.randint(2 ** 31 - 1))


def pytest_report_header(config):
    return f'Using random seed: {config.option.seed}'


@pytest.fixture
def random_state(request):
    random_state = np.random.RandomState(request.config.option.seed)
    return random_state
