import pandas as pd

from probability import TreeDistribution
from util import cartesian
from probability import JointDistByState


class PayoffMatrix(object):
    """Holds a payoff matrix relating two sets of variables
    """

    LABEL = 'Payoff'

    def __init__(self, inputs, outputs, mapping_func):
        # Take copies. Note that the ordering is important here, and it must
        # be consistent over all of the calculations.
        self.inputs = inputs[:]
        self.outputs = outputs[:]
        self.input_names = [vi.name for vi in self.inputs]
        self.output_names = [vo.name for vo in self.outputs]

        all_envs = cartesian(vi.states for vi in inputs)
        all_outs = cartesian(vo.states for vo in outputs)

        self.fitness_lookup = {}

        for env in all_envs:
            env_dict = dict(zip(self.input_names, env))
            for out in all_outs:
                out_dict = dict(zip(self.output_names, out))
                out_dict.update(env_dict)
                # This magically transforms the names into variables. I think
                # this is an okay approach, but it might be a bit confusing to
                # the newcomer.
                fitness = mapping_func(**out_dict)
                self.fitness_lookup[(tuple(env), tuple(out))] = float(fitness)

    def fitness_from_assignments(self, assignments):
        """Get the fitness from the current assignments"""
        env = tuple([assignments[vi] for vi in self.inputs])
        out = tuple([assignments[vo] for vo in self.outputs])

        # Go and find out what the payoff is.
        return self.fitness_lookup[(env, out)]

    def to_frame(self):
        """Convert to DataFrame for nice display"""

        # TODO: Thinks... can we just store it in this format, and use pandas
        # indexing for the lookup?
        data = {}
        data.update(dict([(i_var.name, []) for i_var in self.inputs]))
        data.update(dict([(o_var.name, []) for o_var in self.outputs]))
        data.update({PayoffMatrix.LABEL: []})

        for (env, out), f in self.fitness_lookup.items():
            for env_name, e in zip(self.input_names, env):
                data[env_name].append(e)
            for out_name, o in zip(self.output_names, out):
                data[out_name].append(o)
            data[PayoffMatrix.LABEL].append(f)

        all_data = pd.DataFrame(data=data)

        # Return a table that looks like a ... Payoff Matrix! Tada!
        return pd.pivot_table(data=all_data, values=[PayoffMatrix.LABEL],
                              index=self.input_names,
                              columns=self.output_names)

    def fitness_of(self, dist):
        """Calculate the fitness of a TreeDistribution"""
        assert isinstance(dist, TreeDistribution)
        fit = 0.0
        for ass, p in dist.iter_assignments():
            f = self.fitness_from_assignments(ass)
            fit += f * p
        return fit

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.to_frame()._repr_html_()


def test1():
    from probability import make_variables

    def m1(A, C1, C2):
        if (C1 and C2) == A:
            if C1:
                return 5
            return 2

        return 0

    c, s, a = make_variables('C S A', 2)
    c1, c2 = make_variables('C1 C2', 2)

    p = PayoffMatrix([c1, c2], [a], m1)
    f = p.to_frame()
    print f
    print f.ix[(1, 1)].ix[1]
    # net = CausalGraph([eq1, eq2])
    # print p.fitness_lookup
    # print p.fitness_from_assignments({c1: 1, c2: 1, a: 1})
    # print p.to_frame()

    t = JointDistByState({c1: 0, s: 0})

    print t.probabilities


if __name__ == '__main__':
    test1()
