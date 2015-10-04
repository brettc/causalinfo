import pandas as pd
import numpy as np
from util import cartesian
from itertools import product


class Variable(object):
    """A discrete variable, with a distribution over N states.

    The variable can be assigned a distribution if is being used as input, or
    as an intervention. Otherwise the distribution is calculated.
    """

    USED_NAMES = set()

    def __init__(self, name, n_states):
        """Name the variable, and say how many states it has. Variables start
        off as unassigned.
        """
        assert str(name) == name
        assert name.isalnum()
        # Don't recreate variables
        assert name not in Variable.USED_NAMES

        self.name = name
        Variable.USED_NAMES.add(name)

        # Generate the actual states; this makes it easy to work with.
        self.n_states = n_states
        self.states = range(n_states)

    def uniform(self):
        return np.ones(self.n_states) / float(self.n_states)

    def with_state(self, state):
        assert state in self.states
        dist = np.zeros(self.n_states)
        dist[state] = 1.0
        return dist

    def make_valid_distribution(self, distn):
        """Various checks to make sure nothing silly is happening"""
        np_distn = np.array(distn, dtype=float)
        assert np_distn.shape == (self.n_states,)
        assert np.isclose(np_distn.sum(), 1.0)
        return np_distn

    def __repr__(self):
        return '<{}>'.format(self.name)


def make_variables(strings, n_states):
    """Just a shortcut for making lots of variables"""
    varnames = strings.split()
    return [Variable(v, n_states) for v in varnames]


class Distribution(object):
    """A distribution over one or more variables
    """
    P_LABEL = 'Pr'

    def __init__(self, variables, pr=None):
        # Make a copy
        for v in variables:
            assert isinstance(v, Variable)
        # No duplicates!
        assert len(set(variables)) == len(variables)

        self.variables = list(variables)
        self.names = [v.name for v in variables]
        self.probabilities = pr

    def joint(self, *variables):
        """Generate the (sub)joint distribution over N variables

        Note: This uses pandas' MultiIndex to generate the joint distribution
        any number of variables together from the existing `probabilities`.
        """
        for v in variables:
            assert isinstance(v, Variable)
            assert v in self.variables

        # This is the function that does all the work. It is amazingly
        # flexible.
        pr = pd.pivot_table(
            # We need to treat the variables as columns not indexes for this
            # to work!
            self.probabilities.reset_index(),
            values=[Distribution.P_LABEL],
            index=[v.name for v in variables],
            aggfunc=np.sum,
        )

        # Return another (smaller?) distribution.
        return Distribution(variables, pr)

    def iter_assignments(self):
        # Indexes are returned as single values when there is only one..
        is_single = len(self.variables) == 1
        for indexes, columns in self.probabilities.iterrows():
            if is_single:
                assignments = {self.variables[0]: indexes}
            else:
                assignments = dict([(v, val) for v, val in zip(self.variables, indexes)])
            pr = columns[Distribution.P_LABEL]
            yield assignments, pr

    def entropy(self, *variables):
        return self._calc_entropy(self.joint(*variables).probabilities)

    def mutual_info(self, v1, v2, v3=None):
        """calculate the mutual (or conditional mutual) information.

        the simplest way to do this is to calculate the entropy of various
        joint distributions and bung them together. see here:

        https://en.wikipedia.org/wiki/mutual_information
        https://en.wikipedia.org/wiki/conditional_mutual_information
        """
        # Define a local variable to make it lovely and readable
        h = self.entropy

        # Simple version
        if v3 is None:
            return h(v1) + h(v2) - h(v1, v2)

        # Conditional version
        return h(v1, v3) + h(v2, v3) - h(v1, v2, v3) - h(v3)

    def _calc_entropy(self, df):
        """Calculate entropy of all entries in a Dataframe (ignoring shape)

        Assumes there are no zero entries (the way we generate probabilities
        ensures this).
        """
        # Extract the numpy array, and just treat it as flat. Then do the
        # standard information calculation (base 2).
        q = df.values.ravel()
        return -(q * np.log2(q)).sum()

    def _repr_html_(self):
        return self.probabilities._repr_html_()


class JointDist(Distribution):
    """Construct a joint distribution assuming independence
    """
    def __init__(self, assignments):
        super(JointDist, self).__init__(assignments.keys())

        # Update the distributions to make sure they are numpy and valid.
        np_assgn = {}
        for v, a in assignments.items():
            np_assgn[v] = v.make_valid_distribution(a)

        # Create an array with all possible combinations of states of inputs
        # from these variables.
        all_states = cartesian(v.states for v in np_assgn.keys())

        # Inputs are assumed to be independent (conditional on this branch),
        # so we can generate a cartesian product of the probabilities, then
        # use this to calculate the joint probabilities.
        cart = cartesian([a for a in np_assgn.values()])

        # Each row contains the probabilities for that combination. We just
        # multiply them...
        probs = cart.prod(axis=1)

        # Now we construct the DataFrame, setting the index to the variable
        # names
        p = pd.DataFrame(data=probs, columns=[Distribution.P_LABEL])
        s = pd.DataFrame(data=all_states, columns=self.names)
        self.probabilities = pd.concat([s, p], axis=1)
        self.probabilities.set_index(self.names, inplace=True)

        assert self.probabilities.index.names == self.names


class UniformDist(JointDist):
    """A handy class for constructing joint distributions"""
    def __init__(self, *vs):
        assignments = dict([(v, v.uniform()) for v in vs])
        super(UniformDist, self).__init__(assignments)


class ProbabilityTree(object):
    """A container for Probability Branches"""
    def __init__(self):
        self.variables = set()
        self.root = ProbabilityBranch(self, None, 1.0)

    def all_branches(self):
        """Depth first iteration over the entire tree"""
        for b in self._branch_iter(self.root):
            yield b

    def _branch_iter(self, root):
        yield root
        for b in root.branches:
            for b in self._branch_iter(b):
                yield b

    def dump(self):
        # TODO: Sort the assignments into something nicer with labels,
        # ordered.
        for b in self.all_branches():
            print ' ' * b.depth, b.probability, b.assignments


class ProbabilityBranch(object):
    """The variable assignments and their Conditional probability"""
    def __init__(self, tree, parent, prob, assignments={}):
        self.tree = tree
        tree.variables |= set(assignments.keys())
        self.parent = parent

        # This is the probability, conditional on the parents...
        self.c_prob = prob
        self.assignments = assignments

        # Add all previous variables (Note that means prior assignments will
        # overwrite later ones).
        for p in self.ancestors():
            self.assignments.update(p.assignments)

        # We're a leaf till this is filled out.
        self.branches = []

    def add_variables(self, assignments):
        for v, dist in assignments.items():
            if v not in self.assignments:
                self.assignments[v] = dist

    def ancestors(self):
        """Iterator over all ancestors"""
        p = self.parent
        while p is not None:
            yield p
            p = p.parent

    @property
    def depth(self):
        depth = 0
        for p in self.ancestors():
            depth += 1
        return depth

    @property
    def probability(self):
        """The unconditional probability of this branch

        This simply the product of this and all parent branches"""
        prob = self.c_prob
        for p in self.ancestors():
            prob *= p.c_prob
        return prob

    def add_branches(self, distn):
        self.distribution = distn

        # Now append all of these branches.
        for assign, prob in distn.iter_assignments():
            # Prune zero probability branches
            if prob == 0.0:
                continue

            self.branches.append(ProbabilityBranch(self.tree, self, prob, assign))

    @property
    def is_leaf(self):
        """A branch is a leaf if it has no children"""
        return not self.branches


class TreeDistribution(Distribution):
    def __init__(self, tree, ordering=[]):
        super(TreeDistribution, self).__init__(tree.variables)

        # Add a probability column and check they haven't named a variable the
        # same.
        data = dict([(name, []) for name in self.names])
        assert Distribution.P_LABEL not in data
        data[Distribution.P_LABEL] = []

        # Now go through all the leaf branches, as these will have all values
        # calculated for them. Add each leaf branch probabilities as a row in
        # our table of probabilities
        for b in tree.all_branches():
            # If we're a leaf branch, everything has been evaluated
            if b.is_leaf:
                data[Distribution.P_LABEL].append(b.probability)
                for var, val in b.assignments.items():
                    data[var.name].append(val)

        # Try ordering the columns
        if not ordering:
            ordering = [v.name for v in tree.variables]
        else:
            ordering = ordering[:]
        ordering.append(self.P_LABEL)

        # Create the Dataframe that carries all of the information from the
        # leaves
        self.probabilities = pd.DataFrame(data=data, columns=ordering)
        self.probabilities.set_index(self.names, inplace=True)


def test1():
    import mappings
    a, b, c = make_variables('A B C', 2)
    print a
    print b
    d = JointDist({a: [.3, .7], b: b.uniform()})
    print d.probabilities
    print d.joint(b, a).probabilities
    #
    q = JointDist({a: a.uniform()})
    print q.probabilities

    d = JointDist({a: a.with_state(0), b: b.uniform()})
    print d.probabilities

    #
    # Can trim to zero
    print d.probabilities[d.probabilities['Pr'] != 0.0]

    # d = JointDist({a: a.uniform()})
    # print d.probabilities
    # eq = Equation('xor', [a, b], [c], mappings.f_xor)
    # print eq.mapping_table().index.names
    #
    # print JointDist(eq.calculate({a: 1, b: 1})).probabilities
    #
    # print UniformDist(a, b).probabilities
    #
    # print d.probabilities.index.names
    # for r in d.probabilities.iterrows():
    #     print r[0], r[1]['Pr']


def test_tree():
    a, b, c = make_variables('A B C', 2)
    d = JointDist({a: [.8, .2], b: b.uniform()})
    # d = JointDist({a: [0, 1], b: b.uniform()})
    t = ProbabilityTree()
    t.root.add_branches(d)
    for bch in t.root.branches:
        bch.add_branches(JointDist({c: [.6, .4]}))
    t.dump()

    td = TreeDistribution(t)
    print td.probabilities
    print td.joint(a, c).probabilities
#

if __name__ == '__main__':
    # test1()
    test_tree()
