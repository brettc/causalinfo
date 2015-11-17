import numpy as np
import pandas as pd
from util import cartesian


class Namespace(object):
    """Holds all Variables that are defined
    """

    def __init__(self):
        pass

    def _add(self, name, obj):
        setattr(self, name, obj)


NS = Namespace()


class Variable(object):
    """A discrete variable, with a distribution over N states.
    """

    def __init__(self, name, n_states):
        """Name the variable, and say how many states it has. Variables start
        off as unassigned.
        """
        global NS
        assert isinstance(name, str)
        assert name.isalnum()

        self.name = name
        self.n_states = n_states

        # Generate the actual states; this makes it easy to work with.
        self.states = range(n_states)

        NS._add(name, self)

    def uniform(self):
        """Return a uniform distribution for this variable."""
        return np.ones(self.n_states) / float(self.n_states)

    def with_state(self, state):
        """Return a distribution with just this state set."""
        assert state in self.states
        dist = np.zeros(self.n_states)
        dist[state] = 1.0
        return dist

    def make_valid_distribution(self, distn):
        """Convert distribution and check it."""
        valid_dist = np.array(distn, dtype=float)
        assert valid_dist.shape == (self.n_states,)
        assert np.isclose(valid_dist.sum(), 1.0)
        return valid_dist

    def __repr__(self):
        return '({})'.format(self.name)


def expand_variables(vs):
    """Make sure we have a list of Variables

    This will return a flattened list of variables, no matter what you send
    into it.
    """
    try:
        # This won't work if it ain't iterable
        vsi = iter(vs)
        list_of = []
        for v in vsi:
            list_of.extend(expand_variables(v))
        return list_of
    except TypeError:
        # Okay, it must be a single Variable. Return it in a list.
        assert isinstance(vs, Variable)
        return [vs]


def make_variables(strings, n_states):
    """Just a shortcut for making lots of variables"""
    var_names = strings.split()
    return [Variable(v, n_states) for v in var_names]


class Distribution(object):
    """Base class for a distribution over one or more variables
    """
    P_LABEL = 'Pr'

    def __init__(self, variables, pr=None):
        # Everything should be a variable
        for v in variables:
            assert isinstance(v, Variable)
        # No duplicates!
        assert len(set(variables)) == len(variables)

        self.variables = list(variables)
        self.names = [v.name for v in self.variables]
        self.probabilities = pr

    def joint(self, *variables):
        """Generate the (sub)joint distribution over included variables

        Note: This uses pandas' MultiIndex to generate the joint distribution
        any number of variables together from the existing probabilities.
        """
        variables = expand_variables(variables)

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

    def prob_of_state(self, assignments):
        assert set(assignments.keys()) == set(self.variables)
        # Construct a tuple from the assignments
        by_name = dict([(v.name, val) for v, val in assignments.items()])
        state = tuple([by_name[nm] for nm in self.probabilities.index.names])
        return self.probabilities.loc[state].values[0]

    def iter_conditional(self, a, b):
        # Build a lookup for b
        jb = self.joint(b)
        jab = self.joint(a, b)
        for ass, p in jab.iter_assignments():
            yield ass, p / jb.prob_of_state({b: ass[b]})

    def iter_assignments(self):
        # Indexes are returned as single values when there is only one..
        is_single = len(self.variables) == 1
        for indexes, columns in self.probabilities.iterrows():
            if is_single:
                assignments = {self.variables[0]: indexes}
            else:
                assignments = dict(
                    [(v, val) for v, val in zip(self.variables, indexes)])
            pr = columns[Distribution.P_LABEL]
            yield assignments, pr

    def entropy(self, *variables):
        """Calculate the entropy of one or more variables in this
        distribution."""
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

    @staticmethod
    def _calc_entropy(df):
        """Calculate entropy of all entries in a Dataframe (ignoring shape)

        Assumes there are no zero entries (the way we generate probabilities
        ensures this).
        """
        # Extract the numpy array, and just treat it as flat. Then do the
        # standard information calculation (base 2).
        q = df.values.ravel()
        return -(q * np.log2(q)).sum()

    def query_probability(self, query_string):
        """Provide a simple way to calculate the probability of some
        combination of states in the distribution.
        """

        # We need to reset the index to be able to use the query function.
        result = self.probabilities.reset_index().query(query_string)

        # Sum the probabilities
        return result[self.P_LABEL].sum()

    def to_frame(self):
        """Return a Dataframe (this is just probabilities for now!)"""
        return self.probabilities

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.to_frame()._repr_html_()


class JointDist(Distribution):
    """Construct a joint distribution assuming independence
    """

    def __init__(self, assignments):
        super(JointDist, self).__init__(assignments.keys())

        # Update the distributions to make sure they are numpy and valid.
        np_assgn = {}
        for v, a in assignments.items():
            np_assgn[v] = v.make_valid_distribution(a)

        # NOTE: Ensure you keep the order consistent (use self.variables &
        # self.names to construct everything).

        # Create an array with all possible combinations of states of inputs
        # from these variables.
        all_states = cartesian(v.states for v in self.variables)

        # Inputs are assumed to be independent (conditional on this branch),
        # so we can generate a cartesian product of the probabilities, then
        # use this to calculate the joint probabilities.
        cart = cartesian([np_assgn[v] for v in self.variables])

        # Each row contains the probabilities for that combination. We just
        # multiply them...
        probs = cart.prod(axis=1)

        # Now we construct the DataFrame, setting the index to the variable
        # names
        p = pd.DataFrame(data=probs, columns=[Distribution.P_LABEL])
        s = pd.DataFrame(data=all_states, columns=self.names)
        self.probabilities = pd.concat([s, p], axis=1)
        self.probabilities.set_index(self.names, inplace=True)

        # Check we kept the order consistent
        assert self.probabilities.index.names == self.names


class JointDistByState(JointDist):
    """Construct joint distribution where one state has p=1.0"""

    def __init__(self, state_assignments):
        assignments = dict(
            [(v, v.with_state(s)) for v, s in state_assignments.items()])
        super(JointDistByState, self).__init__(assignments)


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
            for sub_b in self._branch_iter(b):
                yield sub_b

    def _dump(self):
        # TODO: Sort the assignments into something nicer with labels,
        # ordered.
        for b in self.all_branches():
            print ' ' * b.depth, b.probability, b.assignments


class ProbabilityBranch(object):
    """The variable assignments and their Conditional probability"""

    def __init__(self, tree, parent, prob, assignments=None):
        if not assignments:
            assignments = {}
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
        for _ in self.ancestors():
            depth += 1
        return depth

    @property
    def probability(self):
        """The unconditional probability of this branch

        The product of the prob of this and all parent branches
        """
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

            self.branches.append(
                ProbabilityBranch(self.tree, self, prob, assign))

    @property
    def is_leaf(self):
        """A branch is a leaf if it has no children"""
        return not self.branches


class TreeDistribution(Distribution):
    """Construct a joint distribution from a ProbabilityTree

    We use the leaf branches of a probability tree (which *should* add to 1.0)
    to construct a joint distribution over all variables in the tree.
    """

    def __init__(self, tree, ordering=None):
        super(TreeDistribution, self).__init__(tree.variables)
        if not ordering:
            ordering = []

        # Let's keep this around
        self.tree = tree

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

        # Create the Dataframe that carries all of the information from the
        # leaves
        pr = pd.DataFrame(data=data)

        # We use the pivot table to squash together all the branches which
        # have identical states (this happens because the "do" variables
        # filter down the tree and override any local settings).
        self.probabilities = pd.pivot_table(
            pr,
            values=[Distribution.P_LABEL],
            index=ordering,
            aggfunc=np.sum,
        )
