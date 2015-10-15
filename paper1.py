from causalinfo import (
    make_variables, JointDist, JointDistByState, Equation, CausalGraph,
    PayoffMatrix, MeasureSuccess, mappings
)

def layering_case():

    def merge(i1, i2, o1):
        if i2:
            # Perfect spec
            o1[i1] = 1.0
        else:
            o1[i1/2] = 1.0

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
    network = CausalGraph([eq1, eq2, eq3])
    root_dist = JointDist({c1: [.25] * 4, c2: [.2, .8]})
    po = PayoffMatrix([c1, c2], [a], simple_payoff)
    # print po.to_frame()
    m = MeasureSuccess(network, root_dist, po)
    # print m.payoff_for_signal(s1, [c1, c2])

    # j_observe = network.generate_joint(root_dist)
    # print j_observe.joint(s1, a).probabilities
    # print j_observe.joint(s2, a).probabilities
    #
    print m.average_sad(s1, a)
    print m.mutual_info(s1, a)
    print m.mutual_info(s1, a, c2)

    # print m.average_sad(s2, a), m.mutual_info(s2, a)
    
def layering_case2():

    def merge(i1, i2, o1):
        if i2:
            # Perfect spec
            o1[i1] = 1.0
        else:
            o1[i1/2] = 1.0

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
    network = CausalGraph([eq1, eq2, eq3])
    root_dist = JointDist({c1: [.25] * 4, c2: [.2, .8]})
    po = PayoffMatrix([c1, c2], [a], simple_payoff)
    print po.to_frame()
    m = MeasureSuccess(network, root_dist, po)
    print m.payoff_for_signal(s1, [c1, c2])

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
    root_dist = JointDist({c: [.5, .5], k: [.9, .1]})
    pm = PayoffMatrix([c, k], [a], payoffs_weighted)
    # pm = PayoffMatrix([c, k], [a], payoffs_simple)
    print 'payoffs'
    print pm.to_frame()
    # fm = MeasureSuccess(gr, root_dist, pm)
    # print fm.average_sad(s, a)
    # x = pm.to_frame()
    # x['m'] = x.apply(max, axis=1)
    # print x
    # print sum(x['m'])
    # This is what happens to this network (without interventions)
    observed = gr.generate_joint(root_dist)
    print 'fitness is', pm.fitness_of(observed)

    # GENERALISE THIS. It calculates the BEST you can do with S.
    #
    # Get the best signal to send in each environment
    # Does this assume the receiver is perfect?
    tot = 0.0
    table = {}
    for ass, p in root_dist.joint(c).iter_assignments():
        cur_mx = 0.0
        for sval in s.states:
            curr_ass = {s: sval}
            # Add in the assignments from the enviroments
            curr_ass.update(ass)
            t = JointDistByState(curr_ass)
            d = gr.generate_joint(root_dist, do_dist=t)
            f = pm.fitness_of(d)
            table.setdefault(sval, []).append(f * p)
            if f > cur_mx:
                cur_mx = f

        tot += p * cur_mx
        # maxes[cval] = cur_mx
    print 'best is', tot
    best_fixed = max(sum(x) for x in table.values())
    print 'best fixed is', best_fixed

    # NOW ADD. A way to see how well you could do by fixing S. 
    # So we get the value of having control over S!


if __name__ == '__main__':
    # random_case()
    layering_case()
