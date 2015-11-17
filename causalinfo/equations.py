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


def same_(i, o):
    o[i] = 1.0


def rotate_right_(i, o):
    ii = (i + 1) % len(o)
    o[ii] = 1.0


def xnor_(i1, i2, o):
    if i1 == i2:
        o[1] = 1.0
    else:
        o[0] = 1.0


def xor_(i1, i2, o):
    if (i1 or i2) and not (i1 and i2):
        o[1] = 1.0
    else:
        o[0] = 1.0


def and_(i1, i2, o):
    if i1 and i2:
        o[1] = 1.0
    else:
        o[0] = 1.0

def anotb_(i1, i2, o):
    if i1 and not i2:
        o[1] = 1.0
    else:
        o[0] = 1.0

def or_(i1, i2, o):
    if i1 or i2:
        o[1] = 1.0
    else:
        o[0] = 1.0


def branch_same_(i, o1, o2):
    o1[i] = 1.0
    o2[i] = 1.0
