"""Generate a fully supervised trace for addition."""
import enum
import numpy as np


class Code(enum.Enum):
    NO_OP = 0
    ACT = 1
    WRITE = 2
    CARRY = 3
    OUT = 4
    PTR = 5
    LEFT = 6
    RIGHT = 7
    ADD1 = 8
    LSHIFT = 9
    ADD = 10
    DONE = 11


IN1 = 0
IN2 = 1
CARRY = 2
OUT = 3

CONTINUE = 0
RETURN = 1

class Env(object):
    def __init__(self):
        self.scratch = np.array([[0, 0, 0, 9, 6],
                                 [0, 0, 1, 2, 5],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])
        self.ptrs = [4, 4, 4, 4]

        self.input_trace = []
        self.output_trace = []
        self.step_sum = 0

        # Used for printing printing code
        self.indent = 0

    def display_indent(self):
        self.indent += 1

    def display_dedent(self):
        self.indent -= 1

    def read_scratch(self):
        return [
            self.scratch[IN1, self.ptrs[IN1]],
            self.scratch[IN2, self.ptrs[IN2]],
            self.scratch[CARRY, self.ptrs[CARRY]],
            self.scratch[OUT, self.ptrs[OUT]]
        ]

    def amend_trace(self, line, do_return=False):
        if line[0] != Code.NO_OP:
            display_line = [
                code.name if isinstance(code, Code) else code for code in line
            ]

            print(
                '{0}{1} {2} {3} {4}'.format(
                    ' ' * self.indent,
                    display_line[IN1],
                    display_line[IN2],
                    display_line[CARRY],
                    display_line[OUT]

                )
            )

        self.input_trace.append(self.read_scratch() + line)
        if len(self.input_trace) > 1:
            self.output_trace.append(
                [RETURN if do_return else CONTINUE] + line
            )

    def return_op(self):
        self.display_dedent()
        self.amend_trace(
            [Code.NO_OP, Code.NO_OP, Code.NO_OP, Code.NO_OP],
            do_return=True
        )

    def write(self, ptr_id, value):
        self.amend_trace([Code.ACT, Code.WRITE, Code.OUT, value])
        self.scratch[ptr_id, self.ptrs[ptr_id]] = value

    def carry(self):
        self.amend_trace([Code.CARRY, 0, 0, 0])
        self.display_indent()
        self.ptr(CARRY, Code.LEFT)
        self.write(CARRY, 1)
        self.ptr(CARRY, Code.RIGHT)
        self.return_op()

    def ptr(self, ptr_id, direction):
        self.amend_trace([Code.ACT, Code.PTR, ptr_id, direction])
        if direction == Code.LEFT:
            self.ptrs[ptr_id] -= 1
        else:
            self.ptrs[ptr_id] += 1

    def add1(self):
        self.amend_trace([Code.ADD1, 0, 0, 0])
        self.display_indent()

        if self.step_sum > 9:
            out_value = self.step_sum % 10
            self.write(OUT, out_value)
            self.carry()
        else:
            self.write(OUT, self.step_sum)

        self.return_op()

    def add(self):
        self.amend_trace([Code.ADD, 0, 0, 0])
        self.display_indent()

        while self.keep_adding():
            self.add1()
            self.lshift()

    def lshift(self):
        self.amend_trace([Code.LSHIFT, 0, 0, 0])
        self.display_indent()
        self.ptr(IN1, Code.LEFT)
        self.ptr(IN2, Code.LEFT)
        self.ptr(CARRY, Code.LEFT)
        self.ptr(OUT, Code.LEFT)
        self.return_op()

    def keep_adding(self):
        self.step_sum = (self.scratch[IN1, self.ptrs[IN1]] +
                         self.scratch[IN2, self.ptrs[IN2]] +
                         self.scratch[CARRY, self.ptrs[CARRY]])

        return self.step_sum != 0

    def generate(self):
        self.add()
        return zip(self.input_trace, self.output_trace)

env = Env()
trace = env.generate()

for pair in trace:
    print(pair[0])
    print(pair[1])
    print()

env.input_trace[1]