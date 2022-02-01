from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import numpy as np

from gym_stag_hunt.envs.simple import SimpleEnv


def env(cooperation_reward=5, defect_alone_reward=2, defect_together_reward=-1,
        failed_cooperation_punishment=-5, eps_per_game=1, max_time_steps=100):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = ZooSimpleEnvironment(cooperation_reward, defect_alone_reward, defect_together_reward,
                                    failed_cooperation_punishment, eps_per_game, max_time_steps)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class ZooSimpleEnvironment(AECEnv):

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    def __init__(self, cooperation_reward=5, defect_alone_reward=1, defect_together_reward=1,
                 failed_cooperation_punishment=-5, eps_per_game=1, max_time_steps=100):
        """
        :param cooperation_reward: How much reinforcement the agents get for catching the stag
        :param defect_alone_reward: How much reinforcement an agent gets for defecting if the other one doesn't
        :param defect_together_reward: How much reinforcement an agent gets for defecting if the other one does also
        :param failed_cooperation_punishment: How much reinforcement the agents get for trying to catch a stag alone
        :param eps_per_game: How many games happen before the internal done flag is set to True. Only included for
                             the sake of convenience.
        """

        super().__init__()
        self.simple_env = SimpleEnv(cooperation_reward, defect_alone_reward, defect_together_reward,
                                    failed_cooperation_punishment, eps_per_game)
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {agent: self.simple_env.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.simple_env.action_space for agent in self.possible_agents}
        self.has_reset = True

        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_observation = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        self.t = 0
        self.last_rewards = [0, 0]
        self.max_time_steps = max_time_steps

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self):
        obs = self.simple_env.reset()
        obs = np.asarray((1, 1))
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.current_observation = {agent: obs for idx, agent in enumerate(self.agents)}

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.t = 0

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        obs, rewards, done, info = self.simple_env.step(actions)
        self.last_rewards = rewards

        if self.t >= self.max_time_steps:
            done = True

        info = {"t": self.t}

        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.current_observation[agent] = np.asarray(obs)
            self.rewards[agent] = rewards[idx]
            self.infos[agent] = info

    def observe(self, agent):
        returned_observation = self.current_observation[agent]
        return returned_observation

    def render(self, mode='human'):
        self.simple_env.render(mode, self.last_rewards)

    def state(self):
        pass

    def close(self):
        self.simple_env.close()
