"""Generate a fully supervised trace for addition."""
import time
import logging
import functools
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProgramNode(object):
    def __init__(self, prev, prog_name, entry_obs):
        self.prog_name = prog_name
        self.entry_obs = entry_obs
        self.exit_obs = None
        self.prev_node = prev
        self.subprograms = []
        self.sub_index = 0
        self.time_step = 0
        self.ret = False

    def step(self, prog, ret, obs):
        """Each step the agent has done one of three things:
            1. Exited the current program
            2. Entered a new subprogram
            3. Executed a primitive operation

        This checks each possibility in turn, verifying that the agent complied
        with the trace.
        """
        self.time_step += 1
        if self.time_step > self.time_limit:
            return (None, -1.0, True)

        # Exited?
        if self.ret:
            if obs != self.exit_obs:
                return self.failed()
            elif not self.completed_all_subprograms():
                return self.failed()
            else:
                return (self.prev_node, 1.0, False)
        self.ret = ret

        # Executed primitive? Just keep on going...
        if prog == 'act':
            return (self, 0.0, False)

        # Must have called subprogram
        if not self.subprograms:
            return self.failed()
        else:
            if prog != self.subprograms[self.sub_index].prog_name:
                return self.failed()
            elif obs != self.subprograms[self.sub_index].entry_obs:
                return self.failed()
            else:
                # The most important state. The agent successfully called a
                # subprogram.
                self.sub_index += 1
                return (self.subprograms[self.sub_index], 1.0, False)

    def failed(self):
        """There are a million ways to fail. So this makes it easy to handle.
        """
        return (None, -1.0, True)

    def completed_all_subprograms(self):
        return self.sub_index == len(self.subprograms)


def program_wrapper(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        name = f.__name__.lstrip('_')
        node = ProgramNode(name, self._get_obs())
        f(self, node, *args, **kwargs)
        node.exit_obs = self._get_obs()
        return program

    return wrapper


class NPIAddEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    PROGRAMS = ['act', 'lshift', 'carry', 'add1', 'add']
    PRIMITIVES = ['write', 'ptr']
    POINTERS = ['in1', 'in2', 'carry', 'out']
    ARGS = ['left', 'right']
    BASE = 10

    def __init__(self):
        # Begin diagnostics
        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = 503
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1
        # End diagnostics

        # Four sub-actions:
        #      1. High-level function, ie write to a ptr, or move a pointer
        #      2 - 4. Arguments for the high-level function, ie, low-level
        #             operation, which pointer, what value.
        self.action_space = Tuple(
            [
                Discrete(len(self.PROGRAMS)),
                Discrete(len(self.PRIMITIVES)),
                Discrete(len(self.POINTERS)),
                Discrete(self.BASE)
            ]
        )
        self.prog_action_space = Discrete(len(self.PROGRAMS))
        self.arg1_action_space = Discrete(len(self.PRIMITIVES))
        self.arg2_action_space = Discrete(len(self.POINTERS))
        self.arg3_action_space = Discrete(self.base)

        self.observation_space = Discrete(4)

        self._init_scratch = np.array([[0, 0, 0, 9, 6],
                                       [0, 0, 1, 2, 5],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]])
        self._scratch = np.copy(self._init_scratch)
        self._ptrs = [4, 4, 4, 4]

        self.step_sum = 0

        self._trace = []
        self._cur_target_node = None
        self.time = 0

    def _reset(self):
        self.step_sum = 0
        self.time = 0

        self._generate_trace()
        self._cur_node = self._trace

        return self._after_reset(self._get_obs())

    def _generate_trace(self):
        # Clear agent's scratch pad
        self._reset_scratch()

        self._trace = self._add()

        # Clear truth scratch pad for agent
        self._reset_scratch()

    def _step(self, action):
        assert self.action_space.contains(action)
        prog, arg1, arg2, arg3 = action
        done = False
        reward = 0.0
2        #assert self._ptrs[Pointer.OUT] >= 0

        # Alter environment
        if prog == self.PROGRAMS.index('act'):
            if arg1 == self.PRIMITIVES.index('write'):
                self._write(arg2, arg3)
            elif arg1 == self.PRIMITIVES.index('ptr'):
                self._ptr(arg2, arg3)

        obs = self._get_obs()

        self.time += 1
        # Terminate if agent took too long
        if self.time >= self.time_limit:
            # print('time limit')
            return (obs, -1.0, True, {})

        # Terminate if a pointer is out of bounds
        if None in obs:
            # print('out of bounds')
            reward = -1.0
            done = True

        # Terminate if agent wrote incorrect value to output
        if self._wrote_bad_output:
            # print('bad write')
            reward = -1.0
            done = True

        if self._wrote_good_output:
            print('        good write')
            reward = 1.0

        # Success?!
        if np.array_equal(self._scratch[Pointer.OUT], self._target_output):
            print('success')
            reward = 10.0
            done = True

        print(self._scratch[Pointer.OUT])
        return self._after_step(obs, reward, done, {})
        #return (obs, reward, done, {})

    def _get_ptr(self, ptr):
        try:
            return self._scratch[
                ptr, self._ptrs[ptr]
            ]
        except IndexError:
            return None

    @property
    def _ptr_in1(self):
        return self._get_ptr(self.POINTERS.index('in1'))

    @property
    def _ptr_in2(self):
        return self._get_ptr(self.POINTERS.index('in2'))

    @property
    def _ptr_carry(self):
        return self._get_ptr(self.POINTERS.index('carry'))

    @property
    def _ptr_out(self):
        return self._get_ptr(self.POINTERS.index('out'))

    def _get_obs(self):
        return [
            self._ptr_in1,
            self._ptr_in2,
            self._ptr_carry,
            self._ptr_out
        ]

    def _reset_scratch(self):
        self._scratch = np.copy(self._init_scratch)
        self._ptrs = [4, 4, 4, 4]

    # Primitive Operations
    def _write(self, ptr_id, value):
        write_col = self._ptrs[ptr_id]
        self._scratch[ptr_id, write_col] = value

    def _ptr(self, ptr_id, direction):
        if direction == self.ARGS.index('left'):
            self._ptrs[ptr_id] -= 1
        elif direction == self.ARGS.index('right'):
            self._ptrs[ptr_id] += 1

    # Compound Operations
    @program_wrapper
    def _carry(self, program):
        self._ptr(self.POINTERS.index('carry'), self.ARGS.index('left'))
        self._write(self.POINTERS.index('carry'), 1)
        self._ptr(self.POINTERS.index('carry'), self.ARGS.index('right'))

    @program_wrapper
    def _lshift(self, program):
        node.subprograms.append(
            self._ptr(self.POINTERS.index('in1'), self.ARGS.index('left'))
        )
        node.subprograms.append(
            self._ptr(self.POINTERS.index('in2'), self.ARGS.index('left'))
        )
        node.subprograms.append(
            self._ptr(self.POINTERS.index('carry'), self.ARGS.index('left'))
        )
        node.subprograms.append(
            self._ptr(self.POINTERS.index('out'), self.ARGS.index('left'))
        )

    @program_wrapper
    def _add1(self, program):
        if self.step_sum > 9:
            out_value = self.step_sum % 10
            self._write(self.POINTERS.index('out'), out_value)
            node.subprograms.append(self._carry())
        else:
            self._write(self.POINTERS.index('out'), self.step_sum)

    @program_wrapper
    def _add(self, program):
        while self._keep_adding():
            add1 = self._add1()
            node.subprograms.append(add1)
            lshift = self._lshift()
            node.subprograms.append(lshift)

    def _keep_adding(self):
        self.step_sum = (self._ptr_in1 + self._ptr_in2 + self._ptr_carry)

        return self.step_sum != 0

    def _render(self, mode='human', close=False):
        pass

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info[
                    "stats.gauges.diagnostics.lag.observation"
                ][0]
                to_log["diagnostics/observation_lag_ub"] = info[
                    "stats.gauges.diagnostics.lag.observation"
                ][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            # logger.info(
            #     'Episode terminating: episode_reward=%s episode_length=%s',
            #     self._episode_reward, self._episode_length
            # )
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation
