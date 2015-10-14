import pandas as pd
import numpy as np
from network import CausalNetwork
from variables import JointDistByState, Distribution


class MeasureCause(object):
    def __init__(self, network, root_dist):
        assert isinstance(network, CausalNetwork)
        assert isinstance(root_dist, Distribution)

        self.network = network
        self.root_dist = root_dist

    def mutual_info(self, a_var, b_var):
        j_observe = self.network.generate_joint(self.root_dist)
        return j_observe.mutual_info(a_var, b_var)

    def causal_flow(self, a_var, b_var, s_var=None):
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
        if s_var is None:
            return self._causal_flow_null(a_var, b_var)

        # 1. First, we simply observe the 'imposed' variable S as it occurs.
        j_observe = self.network.generate_joint(self.root_dist)
        do_s_var = j_observe.joint(s_var)

        # 2. We then use the observed distribution of s to intervene,
        #    and record the resulting joint distribution of s and a
        j_do_s = self.network.generate_joint(self.root_dist, do_dist=do_s_var)
        sa_dist = j_do_s.joint(s_var, a_var)

        # 3. We need to assign using the probabilities of DOING (s and a),
        #    thus need to supply a joint probability over the two of them!
        j_do_a_s = self.network.generate_joint(self.root_dist, do_dist=sa_dist)

        # 4. Now we simply calculate the conditional mutual information over
        #    this final distribution.
        return j_do_a_s.mutual_info(a_var, b_var, s_var)

    def _causal_flow_null(self, a_var, b_var):
        """Simple un-conditional version of above"""

        # 1. Observe a
        j_observe = self.network.generate_joint(self.root_dist)
        see_a = j_observe.joint(a_var)

        # 2. Use observed distribution of a, but now 'do' it.
        j_do_a = self.network.generate_joint(self.root_dist, do_dist=see_a)
        return j_do_a.mutual_info(a_var, b_var)

    def average_sad(self, a_var, b_var):
        """The specificity of a on b, averaged across all root states

        (s)pecific (a)ctual (d)ifference making, averaged across all
        environments.  This assumes that the intervention distribution is the
        global distribution (as in causal flow). Note that this calculation is
        equivalent to:

        I(X_A -> X_B | NOT(X_A))

        where NOT(X_A) is all pathways between the root variables and X_B not
        going through X_A.
        """
        tot = 0.0
        # First, get the unconditional distribution of the variable a_var
        a_dist = self.network.generate_joint(self.root_dist).joint(a_var)

        # Now average this across all of the root assignments 'doing a' with
        # the above distribution.
        for ass, p in self.root_dist.iter_assignments():
            j = JointDistByState(ass)
            d = self.network.generate_joint(j, do_dist=a_dist)
            tot += p * d.mutual_info(a_var, b_var)
        return tot


class MeasureSuccess(MeasureCause):
    def __init__(self, network, root_dist, payoff_mapping):
        super(MeasureSuccess, self).__init__(network, root_dist)

        # This is what happens to this network (without interventions)
        observed = network.generate_joint(root_dist)

        # Let's look at just the inputs and outputs
        inputs_and_outputs = list(self.network.inputs) + list(self.network.outputs)
        results = observed.joint(inputs_and_outputs).probabilities.reset_index()
        results['Fit'] = results.apply(payoff_mapping, axis=1)

        # We now have fitness assigned for each positive probability
        # environment. Let's construct something that allows us look up the
        # fitness of any environment
        summed = pd.pivot_table(
            results,
            values=['Fit'],
            index=[v.name for v in self.network.inputs],
            aggfunc=np.sum,
        )
        print summed
       
def payoffs(row):
    if row['C'] == 0:
        if row['A'] == 0:
            return 5.0
    elif row['C'] == 1:
        if row['A'] == 1:
            return 2.0
    return 0.0


def test1():
    from variables import make_variables, JointDist
    from network import Equation, CausalNetwork
    import mappings
    c, s, a = make_variables('C S A', 2)
    eq1 = Equation('Send', [c], [s], mappings.f_rotate_right)
    eq2 = Equation('Recv', [s], [a], mappings.f_rotate_right)
    network = CausalNetwork([eq1, eq2])
    root_dist = JointDist({c: [.7, .3]})
    # for i, c in root_dist.probabilities.iterrows():
    #     print i, c
    #
    ms = MeasureSuccess(network, root_dist, payoffs)

    # tot = 0.0

    # for ass, p in observed.iter_assignments():
    #     dd = dict([(v.name, val) for v, val in ass.items()])
    #     blarg(**dd)
    #
    # Now average this across all of the root assignments 'doing a' with
    # the above distribution.
    # for ass, p in root_dist.iter_assignments():
    #     j = JointDistByState(ass)
    #     d = network.generate_joint(j, do_dist=s_dist)
    #     print d.probabilities
    #     tot += p * d.mutual_info(s, a)
    # print tot


if __name__ == '__main__':
    test1()
