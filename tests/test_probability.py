from pandas.util.testing import assert_frame_equal

from causalinfo import equations
from causalinfo.graph import Equation, CausalGraph
from causalinfo.probability import (Variable, make_variables, UniformDist,
                                    expand_variables, JointDistByState)


def test_variable_creation():
    a = Variable('A', 5)
    assert a.states == range(5)


def test_expand_variables():
    a, b, c, d = make_variables("A B C D", 2)
    assert expand_variables([a, [b, c]]) == [a, b, c]
    assert expand_variables(a) == [a]


def test_joint():
    a, b, c, d = make_variables("A B C D", 2)
    j = UniformDist(a, b, c, d)
    assert_frame_equal(j.joint([a, b]).probabilities, j.joint(a, b).probabilities)


def test_entropy():
    a, b = make_variables("A B", 2)
    c, d = make_variables("C D", 4)
    j = UniformDist(a, b, c, d)
    assert j.entropy(a) == 1
    assert j.entropy(c) == 2
    assert j.entropy([a, b]) == j.entropy(a, b)


def test_mutual_info():
    a, b = make_variables("A B", 2)
    c, d = make_variables("C D", 4)
    j = UniformDist(a, b, c, d)
    assert j.mutual_info([a, b], c) == 0
    assert j.mutual_info(a, b, c) == 0


def test_uniform():
    a, b = make_variables("A B", 4)
    c = Variable("C", 2)
    j = UniformDist(a, b, c)
    assert j.probabilities.size == 4 * 4 * 2
    assert j.entropy(a) == 2.0
    assert j.entropy(b) == 2.0
    assert j.entropy(c) == 1.0


def test_distribution():
    a, b, c = make_variables("A B C", 2)
    eq1 = Equation('xor', [a, b], [c], equations.xor_)
    net = CausalGraph([eq1])
    ab = UniformDist(a, b)
    j_obs = net.generate_joint(ab)
    j_do_a = net.generate_joint(ab, do_dist=JointDistByState({a: 0}))

    # for ass, p in j_obs.iter_conditional(c, a):
    #     print ass, p
    #     # assert p == 0.5
    #
    # for ass, p in j_do_a.iter_conditional(c, a):
    #     # assert p == 0.5
    #     print ass, p

    assert j_obs.mutual_info(b, c) == 0
    # Very different under "Doing"
    assert j_do_a.mutual_info(b, c) == 1.0
