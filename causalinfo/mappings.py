"""A set of re-usable functions for mapping inputs to outputs

For each function we need to map the current state of a set of input variables
to a set of output *distributions*.

NOTES
-----

1. For variables with only two states which we consider binary, we adopt the
   convention of state 0 = False and state 1 = True
2. I use the convention of prefixing them with `f_` to avoid conflicts with
   keywords (like `and`).
"""


def f_same(i, o):
    o[i] = 1.0


def f_xnor(i1, i2, o):
    if i1 == i2:
        o[1] = 1.0
    else:
        o[0] = 1.0


def f_xor(i1, i2, o):
    if (i1 or i2) and not (i1 and i2):
        o[1] = 1.0
    else:
        o[0] = 1.0


def f_and(i1, i2, o):
    if i1 and i2:
        o[1] = 1.0
    else:
        o[0] = 1.0


def f_or(i1, i2, o):
    if i1 or i2:
        o[1] = 1.0
    else:
        o[0] = 1.0


def f_branch_same(i, o1, o2):
    o1[i] = 1.0
    o2[i] = 1.0
