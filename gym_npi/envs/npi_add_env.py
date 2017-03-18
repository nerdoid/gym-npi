"""Generate a fully supervised trace for addition."""
import time
import logging
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pointer(object):
    IN1 = 0
    IN2 = 1
    CARRY = 2
    OUT = 3


class Arg(object):
    NO_ARG = 0
    LEFT = 1
    RIGHT = 2


class NPIAddEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    OPERATIONS = ['write', 'ptr']
    WRITE = 0
    PTR = 1

    POINTERS = ['in1', 'in2', 'carry', 'out']

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

        self.base = 10
        # Three sub-actions:
        #      1. Operation, ie write to a ptr, or move a pointer
        #      2 & 3. Arguments for the operation, ie which pointer, what value
        self.action_space = Tuple(
            [
                Discrete(len(self.OPERATIONS)),
                Discrete(len(self.POINTERS)),
                Discrete(self.base)
            ]
        )
        self.prog_action_space = Discrete(len(self.OPERATIONS))
        self.arg1_action_space = Discrete(len(self.POINTERS))
        self.arg2_action_space = Discrete(self.base)

        self.observation_space = Discrete(4)

        self._wrote_bad_output = False
        self._wrote_good_output = False
        self._truth_check_col = 4

        self._init_scratch = np.array([[0, 0, 0, 9, 6],
                                       [0, 0, 1, 2, 5],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]])
        self._scratch = np.copy(self._init_scratch)
        self._ptrs = [4, 4, 4, 4]

        self.step_sum = 0

        self._truth_trace = []
        self._agent_trace = []
        self.time = 0

        self._target_output = None
        self._generate_truth()

    def _get_ptr(self, ptr):
        try:
            return self._scratch[
                ptr, self._ptrs[ptr]
            ]
        except IndexError:
            return None

    @property
    def _ptr_in1(self):
        return self._get_ptr(Pointer.IN1)

    @property
    def _ptr_in2(self):
        return self._get_ptr(Pointer.IN2)

    @property
    def _ptr_carry(self):
        return self._get_ptr(Pointer.CARRY)

    @property
    def _ptr_out(self):
        return self._get_ptr(Pointer.OUT)

    def _read_scratch(self):
        return [
            self._ptr_in1,
            self._ptr_in2,
            self._ptr_carry,
            self._ptr_out
        ]

    def _reset_scratch(self):
        self._scratch = np.copy(self._init_scratch)
        self._ptrs = [4, 4, 4, 4]

    def _amend_trace(self, trace):
        obs = self._read_scratch()
        if trace == 'truth':
            self._truth_trace.append(obs)
        else:
            self._agent_trace.append(obs)

    # Primitive Operations
    def _write(self, ptr_id, value, trace='truth'):
        write_col = self._ptrs[ptr_id]
        self._scratch[ptr_id, write_col] = value
        self._amend_trace(trace)

        if trace != 'truth' and ptr_id == Pointer.OUT:
            if value != self._target_output[write_col]:
                self._wrote_bad_output = True

            if write_col == self._truth_check_col:
                truth_out = self._target_output[self._truth_check_col]
                if value == truth_out:
                    self._truth_check_col -= 1
                    self._wrote_good_output = True


    def _ptr(self, ptr_id, direction, trace='truth'):
        if direction == Arg.LEFT:
            self._ptrs[ptr_id] -= 1
        elif direction == Arg.RIGHT:
            self._ptrs[ptr_id] += 1

        self._amend_trace(trace)

    # Compound Operations
    def _carry(self):
        self._ptr(Pointer.CARRY, Arg.LEFT)
        self._write(Pointer.CARRY, 1)
        self._ptr(Pointer.CARRY, Arg.RIGHT)

    def _add1(self):
        if self.step_sum > 9:
            out_value = self.step_sum % 10
            self._write(Pointer.OUT, out_value)
            self._carry()
        else:
            self._write(Pointer.OUT, self.step_sum)

    def _add(self):
        while self._keep_adding():
            self._add1()
            self._lshift()

    def _lshift(self):
        self._ptr(Pointer.IN1, Arg.LEFT)
        self._ptr(Pointer.IN2, Arg.LEFT)
        self._ptr(Pointer.CARRY, Arg.LEFT)
        self._ptr(Pointer.OUT, Arg.LEFT)

    def _keep_adding(self):
        self.step_sum = (self._ptr_in1 + self._ptr_in2 + self._ptr_carry)

        return self.step_sum != 0

    def _generate_truth(self):
        self._reset_scratch()
        self._add()
        self._target_output = self._scratch[Pointer.OUT]
        self._reset_scratch()

    @property
    def time_limit(self):
        # Why not?
        return len(self._truth_trace) * 2

    # def _reward(self):
    #     out_line = self._scratch[3]
    #     agent_out = out_line[self._truth_check_pos]
    #     truth_out = self._target_output[self._truth_check_pos]
    #     if agent_out == truth_out:
    #         self._truth_check_pos -= 1
    #         return 1.0

    #     return 0.0

    def _step(self, action):
        assert self.action_space.contains(action)
        self._wrote_bad_output = False
        op, arg1, arg2 = action
        done = False
        reward = 0.0
        #assert self._ptrs[Pointer.OUT] >= 0

        # Do stuff
        if op == self.WRITE:
            self._write(arg1, arg2, trace='agent')
        elif op == self.PTR:
            self._ptr(arg1, arg2, trace='agent')

        obs = self._read_scratch()

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

    def _reset(self):
        #self._reset_scratch()

        self._wrote_bad_output = False
        self._wrote_good_output = False
        self._truth_check_col = 4
        self.step_sum = 0
        self._truth_trace = []
        self._agent_trace = []
        self.time = 0
        self._target_output = None

        self._generate_truth()

        return self._after_reset(self._read_scratch())
        #return self._read_scratch()

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _render(self, mode='human', close=False):
        pass

    def _after_step(self, observation, reward, done, info):
        self._wrote_bad_output = False
        self._wrote_good_output = False

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

