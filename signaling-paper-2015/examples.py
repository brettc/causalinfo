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
import pandas as pd
import seaborn as sb
import networkx as nx
from pathlib import Path
import sys

_here = Path(__file__).absolute().parent
_package_folder = _here.parent
sys.path.append(str(_package_folder))

from causalinfo import (
    make_variables, JointDist, Equation, CausalGraph, PayoffMatrix,
    MeasureSuccess, mappings, vs, MeasureCause, Variable
)


def controlled_diamond():
    def simple_payoff(C1, C2, A):
        # Full spec, ignore C2
        if C1 == A:
            return 1
        return 0

    c1, c2, s1, s2, s3, s4, a = make_variables('C1 C2 S1 S2 S3 S4 A', 2)
    eq2 = Equation('SAMEB', [c1], [s1, s2], mappings.f_branch_same)
    eq1 = Equation('SAME', [c2], [s3], mappings.f_same)
    eq3 = Equation('AND', [s2, s3], [s4], mappings.f_and)
    eq4 = Equation('OR', [s1, s4], [a], mappings.f_or)
    gr = CausalGraph([eq1, eq2, eq3, eq4])
    # po = PayoffMatrix([c1, c2], [a], simple_payoff)

    # Let's just use Uniform
    # Mutual info is pretty useless, as it is the same across these...
    # assert m.mutual_info(s2, a1) == m.mutual_info(s3, a1)
    #
    # # Look how much better average sad is!
    # assert m.average_sad(s2, a1) < m.average_sad(s3, a1)
    tuples = []
    for p in np.linspace(0, 1, 10):
        root_dist = JointDist({c1: [.5] * 2, c2: [1 - p, p]})
        m = MeasureCause(gr, root_dist)
        mi = m.mutual_info(c1, s1)
        miact = m.mutual_info(a, s1)
        mi2 = m.mutual_info(c1, s2)
        miact2 = m.mutual_info(a, s2)
        spec = m.average_sad(s1, a)
        spec2 = m.average_sad(s2, a)
        # actual, fixed, best = m.payoff_for_signal(s1, [c1, c2])
        # bestval = best - fixed
        # actval = actual - fixed
        tuples.append((p, mi, miact, spec, mi2, miact2, spec2))
        print tuples[-1]

    df = pd.DataFrame.from_records(
        tuples, index=["Prob"],
        columns="Prob M_S1C M_S1A S_S1A M_S2C, M_S2A, C_S2A".split()
    )

    df.to_pickle(str(_here / 'diamond.pickle'))


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
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
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
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
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
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
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
    plot(str(_here / 'mismapping.pickle'))
    plot(str(_here / 'noisy.pickle'))
    plot(str(_here / 'reducedspec.pickle'))





