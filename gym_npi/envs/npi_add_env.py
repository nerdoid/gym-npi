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


PROGRAMS = ['act', 'lshift', 'carry', 'add1', 'add']
PRIMITIVES = ['write', 'ptr']
POINTERS = ['in1', 'in2', 'carry', 'out']
ARGS = ['left', 'right']
BASE = 10


class ProgramNode(object):
    def __init__(self, prog_name, entry_obs):
        self.prog_name = prog_name
        self.prog_id = PROGRAMS.index(prog_name)
        self.entry_obs = entry_obs
        self.is_leaf = False
        self.entry_scratch = None
        self.entry_ptrs = None
        self.exit_scratch = None
        self.exit_ptrs = None
        self.prev = None
        self.subprograms = []
        self.sub_index = 0
        self.time_step = 0
        self.time_limit = 20
        self.ret = False

    def step(self, prog, ret, env):
        """Each step the agent has done one of three things:
            1. Exited the current program
            2. Entered a new subprogram
            3. Executed a primitive operation

        This checks each possibility in turn, verifying that the agent complied
        with the trace.
        """
        self.time_step += 1
        if self.time_step > self.time_limit:
            return (self, -1.0, True, {'env/failure': 'out_of_time'})

        # Executed primitive?
        if prog == PROGRAMS.index('act'):
            if ret == 1:
                self.ret = 1
                return self.traverse_return(env)

            return (self, 0.0, False, {})

        # Called subprogram

        # We can't assess exit observation yet since agent is calling into
        # a subprogram. So save the return flag now and wait for
        # traverse_return to handle it.
        self.ret = ret

        if self.is_leaf:
            return (self, 0.0, False, {})
        else:
            # Since it's not a leaf node, the agent MUST comply with the
            # specified instructions.
            if not self.subprograms:
                return self.failed('no_subprograms')
            elif self.completed_all_subprograms():
                return self.failed('executed_too_many_subprograms')
            elif prog != self.subprograms[self.sub_index].prog_id:
                return self.failed('wrong_next_subprogram')
            elif not self.subprograms[self.sub_index].correct_entry(env):
                return self.failed('wrong_entry_obs')
            else:
                # The agent successfully called a subprogram.
                next_node = self.subprograms[self.sub_index]
                self.sub_index += 1
                return (next_node, 1.0, False, {})

    def traverse_return(self, env, reward=0):
        """Traverses the call graph upward until it reaches a node that has
        not received a return flag. Each successful step in the traversal
        adds 1 to the accumulated reward that is returned at the end.
        """
        if self.ret == 0:
            return (self, reward, False, {})

        if self.correct_exit(env) and self.completed_all_subprograms():
            if self.prev == None:
                return (self, 1.0 + reward, True, {'env/success': True})
            else:
                reward += 1
                return self.prev.traverse_return(env._scratch, reward=reward)
        else:
            return self.failed('bad_exit_scratch')

    def correct_entry(self, env):
        scratches_equal = np.equal(env._scratch, self.entry_scratch).all()
        ptrs_equal = np.equal(env._ptrs, self.entry_ptrs).all()

        return scratches_equal and ptrs_equal

    def correct_exit(self, env):
        scratches_equal = np.equal(env._scratch, self.exit_scratch).all()
        ptrs_equal = np.equal(env._ptrs, self.exit_ptrs).all()

        return scratches_equal and ptrs_equal

    def failed(self, reason):
        """There are a million ways to fail. So this makes it easy to handle.
        """
        print(reason)
        return (self, -1.0, True, {'env/failure': reason})

    def completed_all_subprograms(self):
        return self.is_leaf or self.sub_index == len(self.subprograms)


def program_wrapper(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        name = f.__name__.lstrip('_')
        node = ProgramNode(name, self._get_obs())
        node.entry_scratch = np.copy(self._scratch)
        node.entry_ptrs = list(self._ptrs)
        f(self, node, *args, **kwargs)
        node.exit_scratch = np.copy(self._scratch)
        node.exit_ptrs = list(self._ptrs)
        return node

    return wrapper


class NPIAddEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Begin diagnostics
        #self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = 503
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1
        # End diagnostics

        # Five sub-actions:
        #      1. High-level function, ie write to a ptr, or move a pointer
        #      2 - 4. Arguments for the high-level function, ie, low-level
        #             operation, which pointer, what value.
        #      5. Return flag
        self.action_space = Tuple(
            [
                Discrete(len(PROGRAMS)),
                Discrete(len(PRIMITIVES)),
                Discrete(len(POINTERS)),
                Discrete(BASE),
                Discrete(2)
            ]
        )
        self.prog_action_space = Discrete(len(PROGRAMS))
        self.arg1_action_space = Discrete(len(PRIMITIVES))
        self.arg2_action_space = Discrete(len(POINTERS))
        self.arg3_action_space = Discrete(BASE)
        self.ret_action_space = 2

        self.observation_space = Discrete(4)
        self.arg_space = Discrete(3)

        self._init_scratch = np.array([[0, 0, 0, 9, 6],
                                       [0, 0, 1, 2, 5],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]])
        self._scratch = np.copy(self._init_scratch)
        self._ptrs = [4, 4, 4, 4]

        self.step_sum = 0

        self.input_trace = []
        self.output_trace = []
        self._trace = []
        self._cur_node = None
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

        # TODO Use curriculum to select correct function to start generation
        # self._trace = self._add()
        self._trace = self._lshift(do_return=True)

        # Clear truth scratch pad for agent
        self._reset_scratch()

    def get_supervised_trace(self):
        return zip(self.input_trace, self.output_trace)

    def _step(self, action):
        assert self.action_space.contains(action)
        prog, arg1, arg2, arg3, ret = action
        done = False
        reward = 0.0
        #assert self._ptrs[Pointer.OUT] >= 0

        # Alter environment
        if prog == PROGRAMS.index('act'):
            if arg1 == PRIMITIVES.index('write'):
                self._write(arg2, arg3)
            elif arg1 == PRIMITIVES.index('ptr'):
                self._ptr(arg2, arg3)

        obs = self._get_obs()

        self._cur_node, reward, done, info = self._cur_node.step(
            prog, ret, self
        )

        # Add a distance-based reward at end of episode
        if done:
            ptr_diff = np.subtract(self._ptrs, self._cur_node.exit_ptrs)
            for diff in ptr_diff:
                if diff == 0:
                    reward += 1

        return self._after_step(obs, reward, done, {})

    def _get_ptr(self, ptr):
        try:
            return self._scratch[
                ptr, self._ptrs[ptr]
            ]
        except IndexError:
            return -1

    @property
    def _ptr_in1(self):
        return self._get_ptr(POINTERS.index('in1'))

    @property
    def _ptr_in2(self):
        return self._get_ptr(POINTERS.index('in2'))

    @property
    def _ptr_carry(self):
        return self._get_ptr(POINTERS.index('carry'))

    @property
    def _ptr_out(self):
        return self._get_ptr(POINTERS.index('out'))

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

    def amend_trace(self, line, do_return=False):
        # print(
        #     '{0}{1} {2} {3} {4}'.format(
        #         ' ' * self.indent,
        #         PROGRAMS[line[0]],
        #         PRIMITIVES[line[1]],
        #         POINTERS[line[2]],
        #         ARGS[line[3]]
        #     )
        # )

        self.input_trace.append(self._get_obs() + line)
        if len(self.input_trace) > 1:
            self.output_trace.append(
                [1 if do_return else 0] + line
            )

    # Primitive Operations
    def _write(self, ptr_id, value, do_return=False, trace=False):
        if trace:
            self.amend_trace(
                [
                    PROGRAMS.index('act'),
                    PRIMITIVES.index('write'),
                    POINTERS.index('out'),
                    value
                ],
                do_return
            )

        write_col = self._ptrs[ptr_id]
        try:
            self._scratch[ptr_id, write_col] = value
        except IndexError:
            pass

    def _ptr(self, ptr_id, direction, do_return=False, trace=False):
        if trace:
            self.amend_trace(
                [
                    PROGRAMS.index('act'),
                    PRIMITIVES.index('ptr'),
                    ptr_id,
                    direction
                ],
                do_return
            )

        if direction == ARGS.index('left'):
            self._ptrs[ptr_id] -= 1
        elif direction == ARGS.index('right'):
            self._ptrs[ptr_id] += 1

    # Compound Operations
    @program_wrapper
    def _carry(self, node, prev=None, do_return=False):
        self.amend_trace([PROGRAMS.index('carry'), 0, 0, 0], do_return)
        node.is_leaf = True
        node.prev = prev

        self._ptr(POINTERS.index('carry'), ARGS.index('left'), trace=True)
        self._write(POINTERS.index('carry'), 1, trace=True)
        self._ptr(POINTERS.index('carry'), ARGS.index('right'), trace=True)

    @program_wrapper
    def _lshift(self, node, prev=None, do_return=False):
        self.amend_trace([PROGRAMS.index('lshift'), 0, 0, 0], do_return)
        node.is_leaf = True
        node.prev = prev

        node.subprograms.append(
            self._ptr(POINTERS.index('in1'), ARGS.index('left'), trace=True)
        )
        node.subprograms.append(
            self._ptr(POINTERS.index('in2'), ARGS.index('left'), trace=True)
        )
        node.subprograms.append(
            self._ptr(POINTERS.index('carry'), ARGS.index('left'), trace=True)
        )
        node.subprograms.append(
            self._ptr(
                POINTERS.index('out'),
                ARGS.index('left'),
                do_return=True,
                trace=True
            )
        )

    @program_wrapper
    def _add1(self, node, prev=None, do_return=False):
        self.amend_trace([PROGRAMS.index('add1'), 0, 0, 0], do_return)
        node.prev = prev
        if self.step_sum > 9:
            out_value = self.step_sum % 10
            self._write(POINTERS.index('out'), out_value, trace=True)
            node.subprograms.append(self._carry(node, do_return=True))
        else:
            node.is_leaf = True
            self._write(
                POINTERS.index('out'),
                self.step_sum,
                do_return=True,
                trace=True
            )

    @program_wrapper
    def _add(self, node, prev=None, do_return=False):
        self.amend_trace([PROGRAMS.index('add'), 0, 0, 0], do_return)
        while self._keep_adding():
            add1 = self._add1(node)
            node.subprograms.append(add1)
            lshift = self._lshift(node, do_return=True)
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
            # to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log

    def _after_reset(self, observation):
        # logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation
