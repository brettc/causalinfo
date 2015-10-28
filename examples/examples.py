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
import sys

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

_here = Path(__file__).absolute().parent
_package_folder = _here.parent
sys.path.append(str(_package_folder))

from causalinfo import (
    make_variables, JointDist, Equation, CausalGraph, PayoffMatrix,
    MeasureSuccess, equations, Variable
)


def reducedspec():
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
        return 0.0

    c1, s1, a = make_variables('C1 S1 A', 4)
    c2, s2 = make_variables('C2 S2', 2)
    eq1 = Equation('Send1', [c1], [s1], equations.f_same)
    # eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, c2], [a], merge)
    gr = CausalGraph([eq1, eq3])
    nx.to_agraph(gr.full_network).draw('reducedspec.png', prog='dot')
    po = PayoffMatrix([c1, c2], [a], simple_payoff)

    tuples = []
    for p in np.linspace(0, 1, 20):
        root_dist = JointDist({c1: [.25] * 4, c2: [1 - p, p]})
        m = MeasureSuccess(gr, root_dist, po)

        # Create the measures
        mi = m.mutual_info(c1, s1)
        miact = m.mutual_info(a, s1)
        spec = m.average_sad(s1, a)
        actual, fixed, best = m.payoff_for_signal(s1, [c1, c2])
        bestval = best - fixed
        actval = actual - fixed
        tuples.append((p, mi, miact, spec, bestval, actval))
        print tuples[-1]

    df = pd.DataFrame.from_records(
        tuples, index=["Prob"], 
        columns="Prob Mutual MutualA Spec Best Actual".split()
    )

    df.to_pickle(str(_here / 'reducedspec.pickle'))


def noisy():
    def payoffs_simple(C, A, K):
        if C == A:
            return 1
        return 0

    def randomise(i1, i2, o1):
        if i2 == 1:
            o1[:] = .25
        else:
            o1[i1] = 1.0

    c, s, a = make_variables('C S A', 4)
    k = Variable('K', 2)
    eq1 = Equation('Send', [c, k], [s], randomise)
    eq2 = Equation('Recv', [s], [a], equations.f_same)
    gr = CausalGraph([eq1, eq2])
    nx.to_agraph(gr.full_network).draw('noisy.png', prog='dot')
    po = PayoffMatrix([c, k], [a], payoffs_simple)

    tuples = []
    for p in np.linspace(0, 1, 20):
        root_dist = JointDist({c: [.25] * 4, k: [1 - p, p]})
        m = MeasureSuccess(gr, root_dist, po)

        # Create the measures
        mi = m.mutual_info(c, s)
        miact = m.mutual_info(a, s)
        spec = m.average_sad(s, a)
        actual, fixed, best = m.payoff_for_signal(s, c)
        bestval = best - fixed
        actval = actual - fixed
        tuples.append((p, mi, miact, spec, bestval, actval))
        print tuples[-1]

    df = pd.DataFrame.from_records(
        tuples, index=["Prob"], 
        columns="Prob Mutual MutualA Spec Best Actual".split()
    )

    df.to_pickle(str(_here / 'noisy.pickle'))


def mismapping():

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
    eq1 = Equation('Send1', [c1], [s1], equations.f_same)
    # eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, c2], [a], merge)
    network = CausalGraph([eq1, eq3])
    nx.to_agraph(network.full_network).draw('mismapping.png', prog='dot')

    # nx.draw_graphviz(network.full_network, prog='dot')
    po = PayoffMatrix([c1, c2], [a], simple_payoff)

    tuples = []
    for p in np.linspace(0, 1, 20):
        root_dist = JointDist({c1: [.25] * 4, c2: [1 - p, p]})
        m = MeasureSuccess(network, root_dist, po)

        # Create the measures
        mi = m.mutual_info(c1, s1)
        miact = m.mutual_info(a, s1)
        spec = m.average_sad(s1, a)
        # actual, fixed, best = m.payoff_for_signal(s1, [c1])
        actual, fixed, best = m.payoff_for_signal(s1, [c1, c2])
        bestval = best - fixed
        actval = actual - fixed
        tuples.append((p, mi, miact, spec, bestval, actval))
        print tuples[-1]

    df = pd.DataFrame.from_records(
        tuples,
        index=["Prob"],
        columns="Prob Mutual MutualA Spec Best Actual".split()
    )
    df.to_pickle(str(_here / 'mismapping.pickle'))


def plot(pth):
    assert isinstance(pth, Path)

    df = pd.read_pickle(str(pth))
    df.drop('MutualA', axis=1, inplace=True)
    axes = df.plot(subplots=True)
    mi, spec, best, act = axes
    for a in mi, spec:
        a.set_ylim(0, 2.1)
        a.set_ylabel("Bits")

    for a in best, act:
        a.set_ylim(0, 1.0)
        a.set_yticks(np.linspace(0, 1, 5))
        a.set_ylabel("Fitness")

    fig = axes[0].get_figure()
    fig.savefig(str(pth.with_suffix('.pdf')))


if __name__ == '__main__':
    # controlled_diamond()
    # mismapping()
    # noisy()
    # reducedspec()
    plot(_here / 'mismapping.pickle')
    plot(_here / 'noisy.pickle')
    plot(_here / 'reducedspec.pickle')





