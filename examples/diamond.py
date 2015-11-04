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
import seaborn as sb
import networkx as nx
from pathlib import Path

_here = Path(__file__).absolute().parent
_package_folder = _here.parent
sys.path.append(str(_package_folder))

from causalinfo import (
    make_variables, JointDist, Equation, CausalGraph, PayoffMatrix,
    MeasureSuccess, equations, Variable
)


def diamond():
    def simple_payoff(C1, C2, A):
        # Full spec, ignore C2
        if C1 == A:
            return 1
        return 0

    def complex_payoff(C1, C2, A):
        # Full spec, ignore C2
        if C1 == A:
            if C2 == 1:
                return 10
            return 1
        return 0

    c1, c2, s1, s2, s3, a = make_variables('C1 C2 S1 S2 S3 A', 2)
    eq1 = Equation('branch', [c1], [s1, s2], equations.f_branch_same)
    eq2 = Equation('and', [s2, c2], [s3], equations.f_and)
    eq3 = Equation('or', [s1, s3], [a], equations.f_or)
    gr = CausalGraph([eq1, eq2, eq3])
    dot = nx.to_agraph(gr.full_network)
    dot.write('diamond.dot')
    dot.draw('diamond.png', prog='dot')
    po = PayoffMatrix([c1, c2], [a], complex_payoff)

    # Let's just use Uniform
    # Mutual info is pretty useless, as it is the same across these...
    # assert m.mutual_info(s2, a1) == m.mutual_info(s3, a1)
    p = .5
    root_dist = JointDist({c1: [.5, .5], c2: [1 - p, p]})
    m = MeasureSuccess(gr, root_dist, po)
    print m.generate_signal_payoff(s1, [c1, c2])
    print m.generate_signal_payoff(s2, [c1, c2])

    print '----'
    print m.average_sad(s1, a)
    print m.causal_flow(s1, a, s2)
    print m.causal_flow(s1, a, s3)
    print m.causal_flow(s1, a, [s2, c2])
    print '----'
    print m.average_sad(s2, a)
    print m.causal_flow(s2, a, s1)
    print m.causal_flow(s2, a, [s1, c2])
    # print m.causal_flow(s2, a, s3)
    return
    #
    # # Look how much better average sad is!
    # assert m.average_sad(s2, a1) < m.average_sad(s3, a1)
    tuples = []
    for p in np.linspace(0, 1, 20):
        root_dist = JointDist({c1: [.5, .5], c2: [1 - p, p]})
        m = MeasureSuccess(gr, root_dist, po)
        mi_s1 = m.mutual_info(s1, a)
        mi_s2 = m.mutual_info(s2, a)
        spec_s1 = m.average_sad(s1, a)
        spec_s2 = m.average_sad(s2, a)
        actual, fixed, best = m.payoff_for_signal(s2, c1)
        actval_s2 = best - fixed
        actual, fixed, best = m.payoff_for_signal(s1, c1)
        actval_s1 = best - fixed
        print actval_s1, actval_s2
        tuples.append((p, mi_s1, mi_s2, spec_s1, spec_s2))
        # print tuples[-1]

    df = pd.DataFrame.from_records(
        tuples, index=["Prob"],
        columns="Prob MI_S1 MI_S2 SP_S1 SP_S2".split()
    )

    df.to_pickle(str(_here / 'diamond.pickle'))

def diamond2():
    def simple_payoff(C1, C2, A):
        # Full spec, ignore C2
        if C1 == A:
            return 1
        return 0

    def complex_payoff(C1, C2, A):
        # Full spec, ignore C2
        if C1 == A:
            if C2 == 1:
                return 10
            return 1
        return 0

    c1, c2, s1, s2, s3, s4, a = make_variables('C1 C2 S1 S2 S3 S4 A', 2)
    eq1 = Equation('branch', [c1], [s1, s2], equations.f_branch_same)
    eq2 = Equation('and', [s2, c2], [s3], equations.f_and)

    eq3 = Equation('anotb', [s1, c2], [s4], equations.f_anotb)

    eq4 = Equation('or', [s3, s4], [a], equations.f_or)
    gr = CausalGraph([eq1, eq2, eq3, eq4])
    dot = nx.to_agraph(gr.full_network)
    dot.write('diamond2.dot')
    dot.draw('diamond2.png', prog='dot')
    po = PayoffMatrix([c1, c2], [a], simple_payoff)

    # Let's just use Uniform
    # Mutual info is pretty useless, as it is the same across these...
    # assert m.mutual_info(s2, a1) == m.mutual_info(s3, a1)
    p = .1
    root_dist = JointDist({c1: [.7, .3], c2: [1 - p, p]})
    m = MeasureSuccess(gr, root_dist, po)
    # print m.generate_signal_payoff(s1, [c1, c2])
    # print m.generate_signal_payoff(s2, [c1, c2])

    print '----'
    print m.average_sad(s1, a)
    print m.mutual_info(s1, a)
    # print m.causal_flow(s1, a, s2)
    # print m.causal_flow(s1, a, s3)
    # print m.causal_flow(s1, a, [s2, c2])
    print '----'
    print m.average_sad(s2, a)
    print m.mutual_info(s1, a)
    # print m.causal_flow(s2, a, s1)
    # print m.causal_flow(s2, a, [s1, c2])
    # print m.causal_flow(s2, a, s3)
    return
    #
    # # Look how much better average sad is!
    # assert m.average_sad(s2, a1) < m.average_sad(s3, a1)
    tuples = []
    for p in np.linspace(0, 1, 20):
        root_dist = JointDist({c1: [.5, .5], c2: [1 - p, p]})
        m = MeasureSuccess(gr, root_dist, po)
        mi_s1 = m.mutual_info(s1, a)
        mi_s2 = m.mutual_info(s2, a)
        spec_s1 = m.average_sad(s1, a)
        spec_s2 = m.average_sad(s2, a)
        actual, fixed, best = m.payoff_for_signal(s2, c1)
        actval_s2 = best - fixed
        actual, fixed, best = m.payoff_for_signal(s1, c1)
        actval_s1 = best - fixed
        print actval_s1, actval_s2
        tuples.append((p, mi_s1, mi_s2, spec_s1, spec_s2))
        # print tuples[-1]

    df = pd.DataFrame.from_records(
        tuples, index=["Prob"],
        columns="Prob MI_S1 MI_S2 SP_S1 SP_S2".split()
    )

    df.to_pickle(str(_here / 'diamond2.pickle'))

def plot(pth):
    assert isinstance(pth, Path)

    df = pd.read_pickle(str(pth))
    print df
    axes = df.plot(subplots=True)
    for a in axes:
        a.set_ylim(-.1, 1.1)
        a.set_yticks(np.linspace(0, 1, 5))

    # mi, mspec, best, act = axes
    # for a in mi, spec:
    #     a.set_ylim(0, 2.1)
    #     a.set_ylabel("Bits")
    #
    # for a in best, act:
    #     a.set_ylim(0, 1.0)
    #     a.set_yticks(np.linspace(0, 1, 5))
    #     a.set_ylabel("Fitness")
    #
    fig = axes[0].get_figure()
    fig.savefig(str(pth.with_suffix('.pdf')))

if __name__ == '__main__':
    # diamond()
    diamond2()
    # plot(_here / 'diamond.pickle')


