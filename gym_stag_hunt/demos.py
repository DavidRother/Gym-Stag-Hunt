from time import sleep

from gym_stag_hunt.envs.escalation import EscalationEnv
from gym_stag_hunt.envs.harvest import HarvestEnv
from gym_stag_hunt.envs.hunt import HuntEnv
from gym_stag_hunt.envs.simple import SimpleEnv
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND

ENVS = {
    'CLASSIC': SimpleEnv,
    'HUNT': HuntEnv,
    'HARVEST': HarvestEnv,
    'ESCALATION': EscalationEnv
}


def print_ep(obs, reward, done, info):
    print({
        'observation': obs,
        'reward': reward,
        'simulation over': done,
        'info': info
    })


def dir_parse(key):
    d = {
        LEFT: "LEFT",
        UP: "UP",
        DOWN: "DOWN",
        RIGHT: "RIGHT",
        STAND: "STAND"
    }
    return d[key]


def manual_input():
    i = input()
    if i in ['w', 'W']:
        i = UP
    elif i in ['a', 'A']:
        i = LEFT
    elif i in ['s', 'S']:
        i = DOWN
    elif i in ['d', 'D']:
        i = RIGHT
    elif i in ['x', 'X']:
        i = STAND

    return i


ENV = 'CLASSIC'
acc_rewards = [0, 0]

if __name__ == "__main__":
    if ENV == 'CLASSIC':
        env = ENVS[ENV]()
    else:
        env = ENVS[ENV](obs_type='image', enable_multiagent=True)
    obs = env.reset()
    for i in range(1000):
        actions = [env.action_space.sample(), env.action_space.sample()]

        obs, rewards, done, info = env.step(actions=actions)
        acc_rewards[0] += rewards[0]
        acc_rewards[1] += rewards[1]
        print(rewards)
        # print_ep(obs, rewards, done, info)
        # sleep(.4)
        # if ENV == 'CLASSIC':
        #     env.render(rewards=rewards)
        # else:
        #     env.render(mode='human')
    env.close()
    quit()
