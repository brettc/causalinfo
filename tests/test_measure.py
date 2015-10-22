from numpy.testing import assert_allclose
from numpy import log2

from causalinfo.probability import (make_variables, UniformDist, JointDist)
from causalinfo.network import Equation, CausalGraph
from causalinfo.measure import MeasureCause
from causalinfo import mappings


def test_controlled_diamond():
    """This examples can move us from a correlation case to a diamond
    """
    c1, c2, s1, s2, s3, s4, a1 = make_variables('c1 c2 s1 s2 s3 s4 a1', 2)

    eq1 = Equation('SAME', [c1], [s1], mappings.f_same)
    eq2 = Equation('SAMEB', [c2], [s2, s3], mappings.f_branch_same)
    eq3 = Equation('AND', [s1, s2], [s4], mappings.f_and)
    eq4 = Equation('OR', [s3, s4], [a1], mappings.f_or)
    net = CausalGraph([eq1, eq2, eq3, eq4])

    # Let's just use Uniform
    m = MeasureCause(net, UniformDist(c1, c2))

    # Mutual info is pretty useless, as it is the same across these...
    assert m.mutual_info(s2, a1) == m.mutual_info(s3, a1)

    # Look how much better average sad is!
    assert m.average_sad(s2, a1) < m.average_sad(s3, a1)


def test_ay_polani():
    w, x, y, z = make_variables("W X Y Z", 2)
    wdist = UniformDist(w)

    # Ay & Polani, Example 3
    eq1 = Equation('BR', [w], [x, y], mappings.f_branch_same)
    eq2 = Equation('XOR', [x, y], [z], mappings.f_xor)

    # Build the graph
    eg3 = CausalGraph([eq1, eq2])
    m_eg3 = MeasureCause(eg3, wdist)

    # See the table on p29
    assert m_eg3.mutual_info(x, y) == 1
    assert m_eg3.mutual_info(x, y, w) == 0
    assert m_eg3.mutual_info(w, z, y) == 0

    assert m_eg3.causal_flow(x, y) == 0
    assert m_eg3.causal_flow(x, y, w) == 0
    assert m_eg3.causal_flow(w, z, y) == 1

    # Ay & Polani, Example 5.1
    def f_copy_first(i1, i2, o1):
        o1[i1] = 1.0

    eq2 = Equation('COPYX', [x, y], [z], f_copy_first)
    eg51 = CausalGraph([eq1, eq2])
    m_eg51 = MeasureCause(eg51, wdist)

    # See paragraph at top of page 30
    assert m_eg51.mutual_info(x, z, y) == 0
    assert m_eg51.causal_flow(x, z, y) == 1
    assert m_eg51.causal_flow(x, z) == 1

    # Ay & Polani, Example 5.2
    def f_random_sometimes(i1, i2, o1):
        if i1 != i2:
            o1[:] = .5
        else:
            mappings.f_xor(i1, i2, o1)

    eq2 = Equation('RAND', [x, y], [z], f_random_sometimes)
    eg52 = CausalGraph([eq1, eq2])
    m_eg52 = MeasureCause(eg52, wdist)

    # See pg 30
    expected = 3.0 / 4.0 * log2(4.0 / 3.0)
    assert_allclose(m_eg52.causal_flow(x, z, y), expected)


def test_signal_of_for():
    c, s, a, k = make_variables('C S A K', 2)
    eq1 = Equation('Send', [c], [s, k], mappings.f_branch_same)
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
    network = CausalGraph([eq1, eq2])
    root_dist = JointDist({c: [.7, .3]})
    m = MeasureCause(network, root_dist)

    assert_allclose(m.mutual_info(s, a), m.mutual_info(k, a))
    assert_allclose(m.causal_flow(s, a), m.average_sad(s, a))

    assert m.causal_flow(k, a) == 0
    assert m.average_sad(k, a) == 0


def xxtest_half_signal():

    def merge(i1, i2, o1):
        if i2:
            o1[i1] = 1.0
        else:
            o1[0] = 1.0

    print 'signaling 2'
    c1, c2, s1, s2, s3, a = make_variables('C1 C2 S1 S2 S3 A', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [a], merge)
    network = CausalGraph([eq1, eq2, eq3])
    root_dist = UniformDist(c1, c2)
    root_dist = JointDist({c1: [.5, .5], c2: [0, 1]})
    print root_dist.probabilities
    m = MeasureCause(network)
    # print m.causal_flow2(s2, a, root_dist)


def test_signal3():

    def merge(i1, i2, o1):
        if i2:
            # Perfect spec
            o1[i1] = 1.0
        else:
            o1[i1/2] = 1.0

    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2 = make_variables('C2 S2', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [a], merge)
    network = CausalGraph([eq1, eq2, eq3])
    root_dist = JointDist({c1: [.25] * 4, c2: [.5, .5]})
    m = MeasureCause(network, root_dist)
    j_observe = network.generate_joint(root_dist)
    print j_observe.joint(s1, a).probabilities
    print j_observe.joint(s2, a).probabilities

    # We check on the equivalence of these.
    assert_allclose(m.causal_flow(s1, a, s2), m.average_sad(s1, a))
    assert_allclose(m.causal_flow(s2, a, s1), m.average_sad(s2, a))

    print m.average_sad(s1, a), m.mutual_info(s1, a)
    print m.average_sad(s2, a), m.mutual_info(s2, a)


def test_random_spec():

    def add_noise(i1, i2, o1):
        if i2:
            # Random Coin toss
            o1[:] = .5
        else:
            # Otherwise specific
            o1[i1] = 1.0

    c1, c2, s1, a = make_variables('C1 C2 S1 A', 2)
    eq1 = Equation('Send1', [c1, c2], [s1], add_noise)
    eq2 = Equation('Send2', [s1], [a], mappings.f_same)
    network = CausalGraph([eq1, eq2])
    root_dist = JointDist({c1: [.2, .8], c2: [.5, .5]})
    m = MeasureCause(network, root_dist)

    # Spec is the same
    assert m.mutual_info(s1, a) == m.average_sad(s1, a)


def xxxtest_signal4():

    def merge1(i1, i2, o1):
        if i2:
            # Perfect spec
            o1[i1] = 1.0
        else:
            o1[i1/2] = 1.0

    def merge2(i1, i2, o1):
        if i1:
            # Perfect spec
            o1[i2] = 1.0
        else:
            # Swap them
            if i2 == 0:
                o1[1] = 1.0
            elif i2 == 1:
                o1[0] = 1.0
            else:
                o1[i2] = 1.0

    print 'signaling 4'
    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2, c3, s4 = make_variables('C2 S2 C3 S4', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s2, s1], [s4], merge1)
    eq4 = Equation('Rec2', [c3], [s3], mappings.f_same)
    eq5 = Equation('Rec3', [s3, s4], [a], merge2)
    network = CausalGraph([eq1, eq2, eq3, eq4, eq5])
    root_dist = JointDist({c1: [.25] * 4, c2: [.2, .8], c3: [.5, .5]})
    # for ass, p in root_dist.iter_assignments():
    #     print ass, p
    m = MeasureCause(network, root_dist)
    # print m.causal_flow2(s1, a, root_dist)
    # print m.causal_flow2(s2, a, root_dist)
    print m.causal_flow(s1, a, [s2, s3])
    print m.average_sad(s1, a)
    print m.causal_flow(s3, a, [s1, s2])
    print m.average_sad(s3, a)

    # print m.causal_flow(s2, a, s1, root_dist)
    # print m.average_sad(s2, a, root_dist)

    # print 'observe signal s1'
    # j_observe_tree = network.generate_tree(root_dist)
    # j_observe = TreeDistribution(j_observe_tree)
    # print j_observe.joint(s1).probabilities
    # print j_observe.joint(c1, c2, a).probabilities


def xxtest_diamond():
    def merge(i1, i2, o1):
        if i2:
            # Perfect spec
            o1[i1] = 1.0
        else:
            o1[i1/2] = 1.0

    c, s1, s2, s3, s4, a = make_variables('C S1 S2 S3 S4 A', 2)
    eq1 = Equation('Send', [c], [s1, s2], mappings.f_branch_same)
    eq2 = Equation('Relay1', [s1], [s3], mappings.f_same)
    eq3 = Equation('Relay2', [s2], [s4], mappings.f_same)
    eq4 = Equation('XOR', [s3, s4], [a], mappings.f_xor)
    n = CausalGraph([eq1, eq2, eq3, eq4])
    root_dist = JointDist({c: [.5, .5]})
    m = MeasureCause(n, root_dist)
    print m.mutual_info(s3, a)
    print m.causal_flow(s3, a, s4)
    print m.average_sad(s3, a)


def test_signal_success():
    def payoffs(C, A):
        if C == A:
            return 1
        return 0

    c, s, a = make_variables('C S A', 2)
    eq1 = Equation('Send', [c], [s], mappings.f_same)
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
    network = CausalGraph([eq1, eq2])
    root_dist = JointDist({c: [.7, .3]})

    # m = MeasureSuccess(network, root_dist, PayoffMatrix([c], [a], payoffs))
