import pandas as pd

from causalinfo.payoff import PayoffMatrix
from graph import CausalGraph
from probability import JointDistByState, Distribution, expand_variables, \
    Variable


class MeasureCause(object):
    def __init__(self, graph, root_dist):
        assert isinstance(graph, CausalGraph)
        assert isinstance(root_dist, Distribution)

        self.graph = graph
        self.root_dist = root_dist

    def mutual_info(self, a_var, b_var, c_var=None):
        j_observe = self.graph.generate_joint(self.root_dist)
        return j_observe.mutual_info(a_var, b_var, c_var)

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

        ... then we can simply measure conditional mutual information:

        I(X_A -> X_B | do(X_S)) = I_{pflow}(X_B : X_A | X_S)

        This also makes it clear that information flow measure can be thought
        of as the information gathered from a consecutive series of
        intervention experiments.
        """
        assert isinstance(a_var, Variable)
        assert isinstance(b_var, Variable)

        if s_var is None:
            # No conditioning? Just use the simpler version...
            return self._causal_flow_null(a_var, b_var)

        # 1. First, we simply observe the 'imposed' variable S as it occurs.
        j_observe = self.graph.generate_joint(self.root_dist)
        do_s_var = j_observe.joint(s_var)

        # 2. We then use the observed distribution of s to intervene,
        #    and record the resulting joint distribution of s and a
        j_do_s = self.graph.generate_joint(self.root_dist, do_dist=do_s_var)
        sa_dist = j_do_s.joint(s_var, a_var)

        # 3. We now calculate the distribution yet again, doing (s and a) this
        #    time. The resulting distribution will be the 'modified
        #    distribution' outlined above.
        j_do_a_s = self.graph.generate_joint(self.root_dist, do_dist=sa_dist)

        # 4. Now we simply calculate the conditional mutual information over
        #    this final distribution.
        return j_do_a_s.mutual_info(a_var, b_var, s_var)

    def _causal_flow_null(self, a_var, b_var):
        """Simple un-conditional version of above

        It takes one less step.
        """

        # 1. Observe a
        j_observe = self.graph.generate_joint(self.root_dist)
        see_a = j_observe.joint(a_var)

        # 2. Use observed distribution of a, but now 'do' it.
        j_do_a = self.graph.generate_joint(self.root_dist, do_dist=see_a)

        # 3. Now we simply calculate the conditional mutual information over
        #    this final distribution.
        return j_do_a.mutual_info(a_var, b_var)

    def average_sad(self, a_var, b_var):
        """The specificity of a on b, averaged across all root states

        (s)pecific (a)ctual (d)ifference making, averaged across all
        environments.  This assumes that the intervention distribution is the
        unconditional (global) distribution (as in causal flow). Note that
        this calculation is equivalent to:

        I(X_A -> X_B | NOT(X_A))

        where NOT(X_A) is all pathways between the root variables and X_B not
        going through X_A.
        """
        tot = 0.0
        # First, get the unconditional distribution of the variable a_var
        a_dist = self.graph.generate_joint(self.root_dist).joint(a_var)

        # Now average this across all of the root assignments 'doing a' with
        # the above distribution.
        for ass, p in self.root_dist.iter_assignments():
            j = JointDistByState(ass)
            d = self.graph.generate_joint(j, do_dist=a_dist)
            tot += p * d.mutual_info(a_var, b_var)
        return tot


class MeasureSuccess(MeasureCause):
    def __init__(self, graph, root_dist, payoffs):
        super(MeasureSuccess, self).__init__(graph, root_dist)

        assert isinstance(payoffs, PayoffMatrix)
        self.payoffs = payoffs

        # # This is what happens to this network (without interventions)
        # observed = network.generate_joint(root_dist)
        #
        # # Let's look at just the inputs and outputs
        # inputs_and_outputs = list(self.network.inputs) + list(
        # self.network.outputs)
        # results = observed.joint(inputs_and_outputs)
        #
        # for ass, p in results.iter_assignments():
        #     f = payoffs.fitness_from_assignments(ass)
        #     print p, ass, f

    def payoff_for_signal(self, signal_var, world_var):
        """Generate the various payoffs

        :param signal_var: The signal under manipulation.
        :param world_var: All or less of the root variables
        :return:
        """

        # 1. Work out the actual fitness
        observed = self.graph.generate_joint(self.root_dist)
        actual_fitness = self.payoffs.fitness_of(observed)

        # 2. Get the best signal to send in each environment. This requires
        #    constructing a payoff matrix for the signal against world.
        tot = 0.0
        table = {}
        for ass, p in self.root_dist.joint(world_var).iter_assignments():
            # Record the current maximum
            cur_mx = 0.0

            # Go through each of the possible signal states.
            for sval in signal_var.states:
                # Assign the state to the variable...
                curr_ass = {signal_var: sval}
                # Add in the assignments from the environment and generate a
                # joint distribution based on this.
                curr_ass.update(ass)
                j = JointDistByState(curr_ass)

                # Now "do" this distribution on the causal graph, and record
                # the payoffs.
                d = self.graph.generate_joint(self.root_dist, do_dist=j)
                f = self.payoffs.fitness_of(d)

                # Update the table tracking the fitness across particular
                # signal state.
                table.setdefault(sval, []).append(f * p)

                # What is the best choice so far?
                if f > cur_mx:
                    cur_mx = f

            # Ok, we now know the best signal to send in this world state.
            tot += p * cur_mx

        best_possible_fitness = tot
        best_fixed_fitness = max(sum(x) for x in table.values())

        return actual_fitness, best_fixed_fitness, best_possible_fitness

    def generate_signal_payoff(self, signal_var, world_var):

        # There could be one or more...
        world_var = expand_variables(world_var)
        world_var_names = [v.name for v in world_var]

        data = {"F": [], signal_var.name: []}
        data.update(dict([(name, []) for name in world_var_names]))

        for ass, p in self.root_dist.joint(world_var).iter_assignments():
            # Go through each of the possible signal states.
            for sval in signal_var.states:
                for v, val in ass.items():
                    data[v.name].append(val)
                data[signal_var.name].append(sval)

                # Assign the state to the variable...
                curr_ass = {signal_var: sval}
                # Add in the assignments from the environment and generate a
                # joint distbution based on this.
                curr_ass.update(ass)
                j = JointDistByState(curr_ass)

                # Now "do" this distribution on the causal graph, and record
                # the payoffs.
                d = self.graph.generate_joint(self.root_dist, do_dist=j)
                f = self.payoffs.fitness_of(d)

                data["F"].append(f * p)

        all_data = pd.DataFrame(data=data)
        return pd.pivot_table(data=all_data, values=["F"],
                              index=world_var_names,
                              columns=[signal_var.name])

    def weighted_average_sad(self, a_var, b_var):
        # TODO: The plan is to weight the individuals contributions in each
        # environment by the success they have.
        tot = 0.0
        # First, get the unconditional distribution of the variable a_var
        a_dist = self.graph.generate_joint(self.root_dist).joint(a_var)

        # Now average this across all of the root assignments 'doing a' with
        # the above distribution.
        for ass, p in self.root_dist.iter_assignments():
            j = JointDistByState(ass)
            d = self.graph.generate_joint(j, do_dist=a_dist)
            tot += p * d.mutual_info(a_var, b_var)
        return tot
