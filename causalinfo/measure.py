from network import CausalNetwork, Equation, make_variables
import mappings


class Measure(object):
    def __init__(self, network):
        self.network = network

    def causal_flow(self, a_var, b_var, s_var):
        """Measure the flow of information from a -> b | do(s)
        
        We rely on the fact that the calculation of flow can be broken down
        into 2 steps. First we generate a joint distribution *under
        interventions*, then we calculate simple conditional mutual
        information over this distribution. 
        
        The ability to do this is clearly described in Ay and Polani's
        original paper (see equation 9), where they say: "It should be noted
        that the information flow measure can be reformulated in terms of
        conditional mutual information with respect to a modified
        distribution..."

        In other words, if we set the modified joint distribution to:

        p_flow(x_S, x_A, x_B) := 
            p(x_S) p(x_A | do(x_S)) p(x_B | do(x_S), do(x_A))

        then:

        I(X_A -> X_B | do(X_S)) = I_{pflow}(X_B : X_A | X_S)

        This also makes it clear that information flow can be thought of as
        calculated using the data gathered from a consecutive series of
        intervention experiments.
        """
        # 1. First, we simply observe the 'imposed' variable S as it occurs.
        j_observe = self.network.generate_joint()

        # print 'S'
        # print j_observe.joint(s_var)

        # 2. We then use the observed distribution of s to intervene...
        s_var.assign_from_joint(j_observe)
        j_do_s = self.network.generate_joint(do=[s_var])

        # print 'A'
        # print j_observe.joint(s_var, a_var)
        # 3. Then we use this distribution of A|do(S), along with do(s) to
        #    intervene again.
        a_var.assign_from_joint(j_do_s)
        j_do_a_s = self.network.generate_joint(do=[s_var, a_var])

        # Not sure how to combine them yet. Is this right...?
        # print 'B'
        # print j_do_a_s.joint(s_var, a_var, b_var)

        # 4. Now we simply calculate the conditional mutual information over
        #    this final distribution.
        return j_do_a_s.mutual_info(a_var, b_var, s_var)


def signal_complex():
    c1, c2, s1, s2, s3, s4, a1 = make_variables('c1 c2 s1 s2 s3 s4 a1', 2)
    # c1.assign_uniform()
    c1.assign([1, 0])
    c2.assign_uniform()
    eq1 = Equation('SAME', [c1], [s1], mappings.f_same)
    eq2 = Equation('SAMEB', [c2], [s2, s3], mappings.f_branch_same)
    eq3 = Equation('AND', [s1, s2], [s4], mappings.f_and)
    eq4 = Equation('OR', [s3, s4], [a1], mappings.f_or)
    net = CausalNetwork([eq1, eq2, eq3, eq4])
    m = Measure(net)
    print m.causal_flow(s3, a1, c2)
    print m.causal_flow(s4, a1, c1)

def testing():
    w, x, y, z = make_variables("W X Y Z", 2)
    w.assign_uniform()

    # Ay & Polani, Example 3
    eq1 = Equation('BR', [w], [x, y], mappings.f_branch_same)
    eq2 = Equation('XOR', [x, y], [z], mappings.f_xor)
    network = CausalNetwork([eq1, eq2])
    m = Measure(network)
    # print m.causal_flow(x, y, w)
    # print m.causal_flow(w, z, y)

    # Ay & Polani, Example 5.1
    # def f_copy_first(i1, i2, o1):
    #     o1[i1] = 1.0
    #
    # eq2 = Equation('COPYX', [x, y], [z], f_copy_first)
    # network = CausalNetwork([eq1, eq2])
    # m = Measure(network)
    # print network.generate_joint().mutual_info(x, z, y)
    # print m.causal_flow(x, z, y)

    # Ay & Polani, Example 5.2
    def f_random_sometimes(i1, i2, o1):
        if i1 != i2:
            o1[:] = .5
        else:
            mappings.f_xor(i1, i2, o1)

    eq2 = Equation('RAND', [x, y], [z], f_random_sometimes)
    network = CausalNetwork([eq1, eq2])
    m = Measure(network)
    print network.generate_joint().mutual_info(x, z, y)
    print m.causal_flow(x, z, y)

    # print 'signaling'
    # c, s, a = make_variables('C S A', 2)
    # c.assign_uniform()
    # eq1 = Equation('Send', [c], [s], mappings.f_same)
    # eq2 = Equation('Recv', [s], [a], mappings.f_same)
    # network = CausalNetwork([eq1, eq2])
    # m = Measure(network)
    # print m.causal_flow(s, a, c)
    # print m.causal_flow(c, a, s)

if __name__ == "__main__":
    signal_complex()
