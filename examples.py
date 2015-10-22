"""Testing

Usage:
  test.py blarg
  test.py (-h | --help)
  test.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""
# from docopt import docopt
import numpy as np

from causalinfo import (
    make_variables, JointDist, Equation, CausalGraph, PayoffMatrix,
    MeasureSuccess, mappings, vs
)

def merge(i1, i2, o1):
    if i2:
        # Reduce the specificity, mapping everything down
        if i1 == 2:
            o1[0] = 1.0
        elif i1 == 3:
            o1[1] = 1.0
        else:
            o1[i1] = 1.0
    else:
        # Perfect spec
        o1[i1] = 1.0

def simple_payoff(C1, C2, A):
    # Full spec
    if C1 == A:
        return 1.0
    return 0

def make_graph():
    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2 = make_variables('C2 S2', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [a], merge)
    gr = CausalGraph([eq1, eq2, eq3])
    return gr

def make_random():
    def payoffs_simple(C, A, K):
        if C == A:
            return 1
        return 0

    def payoffs_weighted(C, A, K):
        if C == A:
            if C == 0:
                return 3
            return 2
        return 0

    def randomise(i1, i2, o1):
        if i2 == 1:
            o1[:] = .5
        else:
            o1[i1] = 1.0

    c, s, k, a = make_variables('C S K A', 2)
    eq1 = Equation('Send', [c, k], [s], randomise)
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
    gr = CausalGraph([eq1, eq2])
    po = PayoffMatrix([c, k], [a], payoffs_simple)

    for p in np.linspace(0, 1, 10):
        root_dist = JointDist({c: [.5, .5], k: [1-p, p]})
        m = MeasureSuccess(gr, root_dist, po)
        print 'mutual on c', m.mutual_info(s, c)
        print 'causal', m.average_sad(s, a)
        print 'payoffs', m.payoff_for_signal(s, c)

def generate():
    gr = make_graph()
    po = PayoffMatrix([vs.C1, vs.C2], [vs.A], simple_payoff)

    probs = np.linspace(0, 1, 10)
    for p in probs:
        root_dist = JointDist({vs.C1: [.25] * 4, vs.C2: [1-p, p]})
        m = MeasureSuccess(gr, root_dist, po)
        print 'mut', m.mutual_info(vs.C1, vs.S1)
        print 'causal', m.average_sad(vs.S1, vs.A)
        print m.payoff_for_signal(vs.S1, [vs.C1, vs.C2])

def partial_misrep():

    def merge(i1, i2, o1):
        if i2 == 0:
            # Perfect spec
            o1[i1] = 1.0
        else:
            # Still perfect spec, but mapping changed
            # swap the last two
            if i1 == 2:
                o1[3] = 1.0
            elif i1 == 3:
                o1[2] = 1.0
            else:
                o1[i1] = 1.0

    def ignore(i1, i2, o1):
        rot = (i1 + 1) % len(o1)
        o1[rot] = 1.0

    def simple_payoff(C1, C2, A):
        # Full spec, ignore C2
        if C1 == A:
            return 1
        return 0

    def complex_payoff(C1, C2, A):
        if C1 == A:
            if C1 == 2:
                return 5
            return 3
        else:
            if C1 == 0:
                return -10
        return 0
        

    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2 = make_variables('C2 S2', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [a], merge)
    # eq3 = Equation('Rec1', [s1, s2], [a], ignore)
    network = CausalGraph([eq1, eq2, eq3])
    po = PayoffMatrix([c1, c2], [a], simple_payoff)

    for p in np.linspace(0, 1, 10):
        root_dist = JointDist({c1: [.25] * 4, c2: [1 - p, p]})
        m = MeasureSuccess(network, root_dist, po)
        print 'mut', m.average_sad(c1, s1)
        print 'causal', m.average_sad(s1, a)
        print 'payoffs', m.payoff_for_signal(s1, [c1, c2])

if __name__ == '__main__':
    # make_random()
    partial_misrep()
    # arguments = docopt(__doc__, version='Naval Fate 2.0')





