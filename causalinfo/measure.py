from variables import (TreeDistribution, UniformDist, JointDist,
                       make_variables, JointDistByState)
from network import CausalNetwork, Equation
import mappings

class Measure(object):
    def __init__(self, network):
        self.network = network

    def causal_flow(self, a_var, b_var, s_var, root_dist):
        """Measure the flow of information from a -> b | s

        We rely on the fact that the calculation of flow can be broken down
        into 2 steps. First we generate a joint distribution *under
        interventions*, then we calculate simple conditional mutual
        information over this distribution.

        This is clearly described in Ay and Polani's original paper (see
        equation 9), where they say: "It should be noted that the information
        flow measure can be reformulated in terms of conditional mutual
        information with respect to a modified distribution..."

        In other words, if we set the joint distribution to:

        p_flow(x_S, x_A, x_B) :=
            p(x_S) p(x_A | do(x_S)) p(x_B | do(x_S), do(x_A))

        ... then we can simple measure conditional mutual information:

        I(X_A -> X_B | do(X_S)) = I_{pflow}(X_B : X_A | X_S)

        This also makes it clear that information flow can be thought of as
        calculated using the information gathered from a consecutive series of
        intervention experiments.
        """
        # 1. First, we simply observe the 'imposed' variable S as it occurs.
        j_observe = self.network.generate_joint(root_dist)
        do_s_var = j_observe.joint(s_var)

        # 2. We then use the observed distribution of s to intervene,
        #    and record the resulting joint distribution of s and a
        j_do_s = self.network.generate_joint(root_dist, do_dist=do_s_var)
        sa_dist = j_do_s.joint(s_var, a_var)

        # 3. We need to assign using the probabilities of DOING (s and a),
        #    thus need to supply a joint probability over the two of them! 
        j_do_a_s = self.network.generate_joint(root_dist, do_dist=sa_dist)

        # 4. Now we simply calculate the conditional mutual information over
        #    this final distribution.
        return j_do_a_s.mutual_info(a_var, b_var, s_var)

    def causal_flow2(self, a_var, b_var, root_dist):
        j_observe_tree = self.network.generate_tree(root_dist)
        j_observe = TreeDistribution(j_observe_tree)
        do_a_var = j_observe.joint(a_var)
        j_do_a_tree = self.network.generate_tree(root_dist, do_dist=do_a_var)
        j_do_a = TreeDistribution(j_do_a_tree)
        # print j_do_a.probabilities
        return j_do_a.mutual_info(a_var, b_var)

    def average_sad(self, a_var, b_var, root_dist):
        tot = 0.0
        a_dist = TreeDistribution(self.network.generate_tree(root_dist)).joint(a_var)
        for ass, p in root_dist.iter_assignments():
            j = JointDistByState(ass)
            d = TreeDistribution(self.network.generate_tree(j, do_dist=a_dist))
            tot += p * d.mutual_info(a_var, b_var)
        return tot
    

def signal_complex():
    c1, c2, s1, s2, s3, s4, a1 = make_variables('c1 c2 s1 s2 s3 s4 a1', 2)
    in_dist = UniformDist(c1, c2)

    eq1 = Equation('SAME', [c1], [s1], mappings.f_same)
    eq2 = Equation('SAMEB', [c2], [s2, s3], mappings.f_branch_same)
    eq3 = Equation('AND', [s1, s2], [s4], mappings.f_and)
    eq4 = Equation('OR', [s3, s4], [a1], mappings.f_or)
    net = CausalNetwork([eq1, eq2, eq3, eq4])
    m = Measure(net)
    print m.causal_flow(s3, a1, c1, in_dist)
    print m.causal_flow(s4, a1, c1, in_dist)

def testing():
    w, x, y, z = make_variables("W X Y Z", 2)

    wdist = UniformDist(w)

    # Ay & Polani, Example 3
    eq1 = Equation('BR', [w], [x, y], mappings.f_branch_same)
    eq2 = Equation('XOR', [x, y], [z], mappings.f_xor)
    network = CausalNetwork([eq1, eq2])
    m = Measure(network)
    print m.causal_flow(x, y, w, wdist)
    print m.causal_flow(w, z, y, wdist)

    # Ay & Polani, Example 5.1
    def f_copy_first(i1, i2, o1):
        o1[i1] = 1.0

    eq2 = Equation('COPYX', [x, y], [z], f_copy_first)
    network = CausalNetwork([eq1, eq2])
    m = Measure(network)
    # print network.generate_joint().mutual_info(x, z, y)
    print m.causal_flow(x, z, y, wdist)

    # Ay & Polani, Example 5.2
    def f_random_sometimes(i1, i2, o1):
        if i1 != i2:
            o1[:] = .5
        else:
            mappings.f_xor(i1, i2, o1)

    eq2 = Equation('RAND', [x, y], [z], f_random_sometimes)
    network = CausalNetwork([eq1, eq2])
    m = Measure(network)
    # print network.generate_joint().mutual_info(x, z, y)
    print m.causal_flow(x, z, y, root_dist=wdist)

def test_signal():
    print 'signaling'
    c, s, a = make_variables('C S A', 2)
    eq1 = Equation('Send', [c], [s], mappings.f_same)
    eq2 = Equation('Recv', [s], [a], mappings.f_same)
    network = CausalNetwork([eq1, eq2])
    m = Measure(network)
    # root_dist = UniformDist(c)
    root_dist = JointDist({c: [.7, .3]})
    # print m.causal_flow(s, a, c, root_dist)
    print m.causal_flow2(s, a, root_dist)
    print m.average_sad(s, a, root_dist)
    # print m.causal_flow(c, a, s, root_dist)
    # j = JointDist({c: [0, 1]})
    # t = network.generate_tree(j)
    # print TreeDistribution(t).joint(a).probabilities
    # j = JointDist({c: [1, 0]})
    # t = network.generate_tree(j)
    # print TreeDistribution(t).joint(a).probabilities
    #




def test_half_signal():

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
    network = CausalNetwork([eq1, eq2, eq3])
    root_dist = UniformDist(c1, c2)
    root_dist = JointDist({c1: [.5, .5], c2: [0, 1]})
    print root_dist.probabilities
    m = Measure(network)
    print m.causal_flow2(s1, a, root_dist)
    # print m.causal_flow2(s2, a, root_dist)
    #
def test_signal3():

    def merge(i1, i2, o1):
        if i2:
            # Perfect spec
            o1[i1] = 1.0
        else:
            o1[i1/2] = 1.0

    print 'signaling 2'
    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2 = make_variables('C2 S2', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [a], merge)
    network = CausalNetwork([eq1, eq2, eq3])
    root_dist = JointDist({c1: [.25] * 4, c2: [.2, .8]})
    m = Measure(network)
    # print m.causal_flow2(s1, a, root_dist)
    # print m.causal_flow2(s2, a, root_dist)
    print m.causal_flow(s1, a, s2, root_dist)
    print m.average_sad(s1, a, root_dist)

    print m.causal_flow(s2, a, s1, root_dist)
    print m.average_sad(s2, a, root_dist)

def test_signal4():

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
            if i2 == 0:
                o1[1] = 1.0
            elif i2 == 1:
                o1[0] = 1.0
            else:
                o1[i2] = 1.0

    print 'signaling 3'
    c1, s1, s3, a = make_variables('C1 S1 S3 A', 4)
    c2, s2, c3, s4, s5 = make_variables('C2 S2 C3 S4 S5', 2)
    eq1 = Equation('Send1', [c1], [s1], mappings.f_same)
    eq2 = Equation('Send2', [c2], [s2], mappings.f_same)
    eq3 = Equation('Rec1', [s1, s2], [s3], merge1)
    eq4 = Equation('Rec2', [c3], [s4], mappings.f_same)
    eq5 = Equation('Rec3', [s4, s3], [a], merge2)
    network = CausalNetwork([eq1, eq2, eq3, eq4, eq5])
    root_dist = JointDist({c1: [.25] * 4, c2: [.2, .8], c3: [.5, .5]})
    m = Measure(network)
    # print m.causal_flow2(s1, a, root_dist)
    # print m.causal_flow2(s2, a, root_dist)
    print m.causal_flow(s1, a, s2, root_dist)
    print m.average_sad(s1, a, root_dist)
    print m.causal_flow(s4, a, s2, root_dist)
    print m.average_sad(s4, a, root_dist)

    # print m.causal_flow(s2, a, s1, root_dist)
    # print m.average_sad(s2, a, root_dist)

    # print 'observe signal s1'
    # j_observe_tree = network.generate_tree(root_dist)
    # j_observe = TreeDistribution(j_observe_tree)
    # print j_observe.joint(s1).probabilities
    # print j_observe.joint(c1, c2, a).probabilities

def test_diamond():
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
    network = CausalNetwork([eq1, eq2, eq3, eq4])
    root_dist = JointDist({c: [.5, .5]})
    m = Measure(network)
    print m.causal_flow(s3, a, s4, root_dist)
    print m.average_sad(s3, a, root_dist)
    print m.causal_flow2(s3, a, root_dist)


def test_random():
    c, s1, s2, s3, s4, a = make_variables('C S1 S2 S3 S4 A', 2)



if __name__ == "__main__":
    # test_diamond()
    test_signal3()
    # test_signal4()
    # test_half_signal()
    # test_signal()
    # testing()
    # signal_complex()
