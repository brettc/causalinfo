from causalinfo import (
    make_variables, JointDist, Equation, CausalGraph, PayoffMatrix,
    MeasureSuccess, mappings, Variable, UniformDist
)

from tabulate import tabulate

def layering_case():

    def merge(i1, i2, o1):
        if i2:
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
            return 1
        return 0

    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2 = make_variables('C2 S2', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [a], merge)
    gr = CausalGraph([eq1, eq2, eq3])
    print eq3.to_frame()
    root_dist = JointDist({c1: [.25] * 4, c2: [.9, .1]})
    po = PayoffMatrix([c1, c2], [a], simple_payoff)
    # print po.to_frame()
    m = MeasureSuccess(gr, root_dist, po)
    j_observe = gr.generate_joint(root_dist)
    print j_observe.joint(c1, a).probabilities

    print 'mutual of env c1', m.mutual_info(s1, c1)
    print 'mutual of env c2', m.mutual_info(s1, c2)
    print 'causal', m.average_sad(s1, a)
    print 'payoffs', m.payoff_for_signal(s1, [c1, c2])
    print m.generate_signal_payoff(s1, [c1, c2])
    # print m.payoff_for_signal(s1, [c1, c2])

    # j_observe = gr.generate_joint(root_dist)
    # print j_observe.joint(s1, a).probabilities
    # print j_observe.joint(s2, a).probabilities
    #
    # print m.average_sad(s1, a)
    # print m.mutual_info(s1, a)
    # print m.mutual_info(s1, a, c2)

    # print m.average_sad(s2, a), m.mutual_info(s2, a)
    
def partial_misrep():

    def merge(i1, i2, o1):
        if i2:
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
    root_dist = JointDist({c1: [.25] * 4, c2: [.2, .8]})
    po = PayoffMatrix([c1, c2], [a], simple_payoff)
    # po = PayoffMatrix([c1, c2], [a], complex_payoff)
    print po.to_frame() ## .reset_index(), headers='keys', tablefmt='pipe')
    m = MeasureSuccess(network, root_dist, po)
    print 'causal', m.average_sad(s1, a)
    print 'payoffs', m.payoff_for_signal(s1, [c1, c2])
    print m.generate_signal_payoff(s1, [c1, c2]) #, headers='keys')

def random_case():
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
    root_dist = JointDist({c: [.5, .5], k: [.5, .5]})
    po = PayoffMatrix([c, k], [a], payoffs_simple)
    m = MeasureSuccess(gr, root_dist, po)

    print 'mutual on c', m.mutual_info(s, c)
    print 'causal', m.average_sad(s, a)
    print 'payoffs', m.payoff_for_signal(s, c)
    print m.generate_signal_payoff(s, c)

def noisy_robust():

    def payoffs_simple(C, A):
        if C == A:
            if C == 0:
                return 2
            else:
                return 10
        return 0

    def noise(i1, i2, o1):
        if i2 == 1:
            o1[:] = .5
        else:
            o1[i1] = 1.0

    c, s1, s2, s3, s4, a = make_variables('C S1 S2 S3 S4 A', 2)
    k1, k2 = make_variables('K1 K2', 2)
    eq1 = Equation('Send', [c], [s1, s2], mappings.f_branch_same)
    eq2 = Equation('Recv1', [s1, k1], [s3], noise)
    eq3 = Equation('Recv2', [s2, k2], [s4], noise)
    eq4 = Equation('Recv3', [s3, s4], [a], mappings.f_or)
    gr = CausalGraph([eq1, eq2, eq3, eq4])
    root_dist = JointDist({c: [.5] * 2, k1: [.9, .1], k2: [.9, .1]})
    po = PayoffMatrix([c], [a], payoffs_simple)
    # print po.to_frame()
    m = MeasureSuccess(gr, root_dist, po)

    print 'mutual on c', m.mutual_info(s1, c)
    print 'causal', m.average_sad(s1, a)
    print 'causal', m.average_sad(s3, a)
    print 'payoffs', m.payoff_for_signal(s1, c)
    print m.generate_signal_payoff(s1, c)
    # print 'payoffs', m.payoff_for_signal(s3, c)
    # print 'payoffs', m.payoff_for_signal(s, c)
    #
    
def basic_signal():
    def payoffs(C, A):
        if C == 0:
            if A == 0:
                return 2
            return -2
        elif C == 1:
            if A == 0:
                return -4
            return 5
        return 0

    c, s, a = make_variables('C S A', 2)
    eq1 = Equation('Send', [c], [s], mappings.f_same)
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
    network = CausalGraph([eq1, eq2])
    # root_dist = JointDist({c: [.3, .7]})
    root_dist = UniformDist(c)
    po = PayoffMatrix([c], [a], payoffs)
    print tabulate(po.to_frame())
    # print tabulate(po.to_frame().reset_index(), headers='A B C'.split(), tablefmt='pipe')
    print tabulate(po.to_frame().reset_index(), headers='A B C'.split(), tablefmt='pipe')
    m = MeasureSuccess(network, root_dist, po)
    print m.average_sad(s, a)
    stab = m.generate_signal_payoff(s, c)
    print stab
    print tabulate(stab, tablefmt='pipe', headers='keys')
    print m.payoff_for_signal(s, c)

    eq2 = Equation('Recv', [s], [a], mappings.f_rotate_right)
    network = CausalGraph([eq1, eq2])
    # root_dist = JointDist({c: [.3, .7]})
    m = MeasureSuccess(network, root_dist, po)
    print m.average_sad(s, a)
    print tabulate(m.generate_signal_payoff(s, c), headers='keys', tablefmt='pipe')
    print m.payoff_for_signal(s, c)


def signal_of_for():
    def payoffs(C, A):
        if C == A:
            return 1
        return 0

    c, s, a, k = make_variables('C S A K', 2)
    eq1 = Equation('Send', [c], [s, k], mappings.f_branch_same)
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
    network = CausalGraph([eq1, eq2])
    root_dist = JointDist({c: [.7, .3]})
    po = PayoffMatrix([c], [a], payoffs)
    m = MeasureSuccess(network, root_dist, po)

    print 'payoffs s', m.payoff_for_signal(s, c)
    print m.generate_signal_payoff(s, c)
    print 'payoffs k', m.payoff_for_signal(k, c)
    print m.generate_signal_payoff(k, c)


if __name__ == '__main__':
    # random_case()
    # # layering_case()
    # partial_misrep()
    # signal_of_for()
    # noisy_robust()
    basic_signal()
