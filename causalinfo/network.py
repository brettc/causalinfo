from itertools import product, chain
import networkx as nx
import numpy as np
import pandas as pd
from util import cartesian


class Variable(object):
    """A discrete variable, with a distribution over N states.

    The variable can be assigned a distribution if is being used as input, or
    as an intervention. Otherwise the distribution is calculated.
    """

    def __init__(self, name, n_states, assigned=None):
        """Name the variable, and say how many states it has. Variables start
        off as unassigned.
        """
        assert str(name) == name
        assert name.isalnum()
        self.name = name

        # Generate the actual states; this makes it easy to work with.
        self.n_states = n_states
        self.states = range(n_states)

        # We store two distributions here. One is assigned, and used for input
        # variables and interventions. The other distribution is calculated
        # during the evaluation of the causal graph.
        if assigned is not None:
            self.assigned = self._check_distribution(assigned)
        else:
            self.assigned = None
        self.calculated = np.zeros(self.n_states)

        self.use_assigned = False

    @property
    def current_distribution(self):
        """Get the current distribution

        This depends on how we're using the variable. It could be as input, or
        as an intervention.
        """
        if self.use_assigned:
            assert self.assigned is not None
            return self.assigned

        if not np.isclose(self.calculated.sum(), 1.0):
            raise ValueError("The distribution of {}"
                             " does not sum to 1.0".format(self.name))
        return self.calculated

    def assign_uniform(self):
        self.assign(np.ones(self.n_states) / float(self.n_states))

    def assign_from_joint(self, full_joint):
        self.assign(full_joint.joint(self).values.ravel())

    def _check_distribution(self, distn):
        """Various checks to make sure nothing silly is happening"""
        # Note: this makes a copy if it already an array. Which is good, as we
        # want to make it unwriteable.
        np_distn = np.array(distn, dtype=float)
        assert np_distn.shape == (self.n_states,)
        assert np.isclose(np_distn.sum(), 1.0)

        # Prevent accidentally screwing with this...
        np_distn.flags.writeable = False
        return np_distn

    def assign(self, distn):
        # Make it readonly. Then we can safely reference it from outside.
        self.assigned = self._check_distribution(distn)

    def clear(self):
        self.assigned = None

    def calculate(self, distn):
        # Here, we assign the distribution
        self.calculated = self._check_distribution(distn)

    def __repr__(self):
        return '<{}>'.format(self.name)

    def _repr_html_(self):
        if self.assigned is not None:
            data = self.assigned
        else:
            data = ['?' for _ in self.states]
        return pd.DataFrame(data=data, columns=[self.name])._repr_html_()


def make_variables(strings, n_states):
    """Just a shortcut for making lots of variables"""
    varnames = strings.split()
    return [Variable(v, n_states) for v in varnames]


class Equation(object):
    """A Equation maps 1+ input variables to 1+ output variables"""

    def __init__(self, name, inputs, outputs, strategy_func):
        assert str(name) == name
        assert not [i for i in inputs if not isinstance(i, Variable)]
        assert not [o for o in outputs if not isinstance(o, Variable)]

        self.name = name
        self.inputs = inputs
        self.outputs = outputs

        # Create an array with all possible combinations of states of inputs
        input_states = list(product(*[i.states for i in inputs]))
        self.input_states = input_states

        # We need arrays to hold the results of each output (note: they could
        # be different sizes)
        self.per_state_results = [
            np.zeros((len(input_states), o.n_states)) for o in outputs]

        # Create a lookup table based on the strategy function. Then we can
        # discard the function.
        self.lookup = {}

        for i, states in enumerate(input_states):
            # Get out relevant states to fill out
            results = [c[i] for c in self.per_state_results]

            # Send arguments as (input, input, ..., output, output, ...)
            args = [s for s in states]
            args.extend(results)

            strategy_func(*args)

            # Each of the output distributions must sum to 1.0
            for r in results:
                assert np.isclose(r.sum(), 1.0)

            # Keep this around
            self.lookup[states] = results

    def play_using(self, assignments):
        """Calculate output given variable / state assignments"""
        # Build a tuple of the relevant input states from the set of
        # assignments given.
        states = tuple([assignments[v] for v in self.inputs])

        # Look them up
        results = self.lookup[states]

        # Now, apply the results to the output variables. Note that this is
        # ignored if the variable is currently being manipulated
        for v, r in zip(self.outputs, results):
            v.calculate(r)

    def __repr__(self):
        return "Equation<{}>".format(self.name)

    def mapping_table(self):
        """Output the mapping equation in a nice way

        We do this a long-winded way, but it allows pandas to do the nice
        formatting for us. We generate a row for every single possibility of
        input and outputs states of this variable, the use the pivot_table to
        construct a table for us with nice row/column headings.
        """
        # Create a set of dictionaries/lists for each column
        data = dict([(i_var.name, []) for i_var in self.inputs])
        data.update({'Output': [], 'State': [], self.name: []})

        # A very ugly loop to produce all the probabilities in a nice way.
        # Note that this just reproduces what is already in `self.lookup`. I
        # just could think of a better way to get nice output.
        for i_index, i_state in enumerate(self.input_states):
            for o_var, results in zip(self.outputs, self.per_state_results):
                for o_state, o_p in enumerate(results[i_index]):
                    for i_var, s in zip(self.inputs, i_state):
                        data[i_var.name].append(s)
                    data['Output'].append(o_var.name)
                    data['State'].append(o_state)
                    data[self.name].append(o_p)
        all_data = pd.DataFrame(data=data)

        # The magnificent pivot table function does all the work
        return pd.pivot_table(data=all_data, values=[self.name],
                              index=[i_var.name for i_var in self.inputs],
                              columns=['Output', 'State'])

    def _repr_html_(self):
        return self.mapping_table()._repr_html_()


class CausalNetwork(object):

    def __init__(self, equations):
        assert not [not_p for not_p in equations
                    if not isinstance(not_p, Equation)]

        self.equations = equations

        # Make a network from this. The first is the full network of both
        # equations and variables (a bipartite graph). The second is just the
        # network of causal variables (the project of the bipartite graph).
        full_network = nx.DiGraph()
        causal_network = nx.DiGraph()
        for p in equations:
            for i in p.inputs:
                full_network.add_edge(i, p)
            for o in p.outputs:
                full_network.add_edge(p, o)
            for i in p.inputs:
                for o in p.outputs:
                    causal_network.add_edge(i, o)

        self.full_network = full_network
        self.causal_network = causal_network

        # Nodes are either inputs, outputs, or inner
        self.inputs = set()
        self.outputs = set()
        self.inner = set()

        for n in self.causal_network.nodes():
            preds = self.full_network.predecessors(n)
            sucs = self.full_network.successors(n)
            if not preds:
                self.inputs.add(n)
            if not sucs:
                self.outputs.add(n)
            if preds and sucs:
                self.inner.add(n)

        # Sort all nodes into topological order. This allows us to calculate
        # the probabilities across the nodes in the right order, so that the
        # inputs for each player are always calculated in time (by previous
        # equations).
        self.ordered_variables = nx.topological_sort(self.causal_network)
        self.ordered_nodes = nx.topological_sort(self.full_network)

        self.graphviz_prettify(self.full_network)
        # self.graphviz_prettify(self.causal_network)

    def graphviz_prettify(self, network):
        """This just makes things pretty for graphviz output."""
        graph_settings = {
            'rankdir': 'TB',
            'dpi': 60,
        }
        network.graph.update(graph_settings)

        for n in self.ordered_nodes:
            network.node[n]['label'] = n.name
            if isinstance(n, Equation):
                network.node[n]['shape'] = 'diamond'

    def reset(self):
        # Reset all the interventions...
        for v in self.ordered_variables:
            v.reset()

    def generate_joint(self, do=[]):
        """Make the joint distribution"""
        tree = self.generate_tree(do)
        return FullJoint(tree)

    def generate_tree(self, do=[]):
        """Generate the ProbabilityTree"""

        # Use the assigned distributions
        for var in chain(self.inputs, do):
            if var.assigned is None:
                raise ValueError("{} has no distribution assigned.".format(var.name))
            var.use_assigned = True

        tree = ProbabilityTree(self)

        # Evaluate all the nodes and recursively construct the
        # ProbabilityTree.
        self.evaluate_branch(tree.root, self.ordered_nodes)

        # Reset this now
        for var in chain(self.inputs, do):
            var.use_assigned = False

        return tree

    def evaluate_branch(self, root, remaining_nodes):
        """Recursively evaluate all possibilities, building a probability tree"""
        current_player = None
        next_nodes = []

        # Get the player that we need to evaluate
        for n in remaining_nodes:
            # Node are both equations and variables. We just want the
            # equations.
            if current_player is None:
                if isinstance(n, Equation):
                    current_player = n
            else:
                # Leave whatever is left for the next layer of evaluation.
                next_nodes.append(n)

        # No more equations! We're done
        if current_player is None:
            return

        # Go through each of the branches and evaluate the state.
        for b in root.branches:
            # Let the player assign output
            current_player.play_using(b.assignments)

            # Add the branches, and then evaluate using the next set of
            # remaining nodes.
            b.add_branches(current_player.outputs)
            self.evaluate_branch(b, next_nodes)


class ProbabilityTree(object):

    def __init__(self, network):
        self.network = network

        # Record the assigned distributions (interventions and inputs)
        self.settings = {}
        for v in network.ordered_variables:
            if v.use_assigned:
                self.settings[v] = v.current_distribution.copy()

        self.root = ProbabilityBranch(None, 1.0)

        self.root.add_branches(network.inputs)

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

    def __init__(self, parent, prob, variables=None, states=None):
        self.parent = parent

        # This is the probability, conditional on the parents...
        self.c_prob = prob

        # Record these -- these are the variables that got assigned to this
        # particular branch.
        self.variables = variables
        self.states = states

        # Also, keep a list of everything that has been assigned up to now.
        if variables is None:
            self.assignments = {}
        else:
            self.assignments = dict(zip(variables, states))

        # Add all previous variables.
        for p in self.ancestors():
            self.assignments.update(p.assignments)

        self.branches = []

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

    def add_branches(self, variables):
        # Create an array with all possible combinations of states of inputs
        # from these variables.
        all_states = product(*[v.states for v in variables])

        # Inputs are assumed to be independent (conditional on this branch),
        # so we can generate a cartesian product of the probabilities, then
        # use this to calculate the joint probabilities.
        cart = cartesian([v.current_distribution for v in variables])

        # Each row contains the probabilities for that combination. We just
        # multiply them...
        probs = cart.prod(axis=1)

        # Now append all of these branches.
        for state, prob in zip(all_states, probs):
            # Prune zero probability branches
            if prob == 0.0:
                continue

            self.branches.append(ProbabilityBranch(
                self, prob, variables, state))

    @property
    def is_leaf(self):
        """A branch is a leaf if it has no children"""
        return not self.branches


class FullJoint(object):
    """All non-zero combinations of variables in a causal graph

    This is the main result from calculating the causal graph. We can
    generate all the required probabilities from it. We can also go and re-use
    the graph and this won't change.
    """
    P_LABEL = 'Pr'

    def __init__(self, tree):
        self.tree = tree

        # Begin by making columns for each of the variables
        data = dict([(v.name, []) for v in tree.network.ordered_variables])

        # Add a probability column and check they haven't named a variable the
        # same.
        assert self.P_LABEL not in data
        data[self.P_LABEL] = []

        # Now go through all the leaf branches, as these will have all values
        # calculated for them. Add each leaf branch probabilities as a row in
        # our table of probabilities
        for b in tree.all_branches():
            # If we're a leaf branch, everything has been evaluated
            if b.is_leaf:
                data['Pr'].append(b.probability)
                for var, val in b.assignments.items():
                    data[var.name].append(val)

        # We shouldn't have any empty data!
        if __debug__:
            for k, v in data.items():
                assert v

        # These have been topologically ordered, so it's a nice way to present
        # them.
        ordering = [v.name for v in tree.network.ordered_variables]
        ordering.append(self.P_LABEL)

        # Create the Dataframe that carries all of the information from the
        # leaves
        self.probabilities = pd.DataFrame(data=data, columns=ordering)

    def joint(self, *variables):
        """Generate the joint distribution over N variables

        Note: This uses pandas' MultiIndex to generate the joint distribution
        any number of variables together from the existing `probabilities`.
        """
        for v in variables:
            assert isinstance(v, Variable)
            assert v in self.tree.network.ordered_variables

        # This is the function that does all the work. It is amazingly
        # flexible.
        return pd.pivot_table(
            self.probabilities,
            values=[self.P_LABEL],
            index=[v.name for v in variables],
            aggfunc=np.sum
        )

    def entropy(self, *variables):
        return self._calc_entropy(self.joint(*variables))

    def mutual_info(self, v1, v2, v3=None):
        """Calculate the mutual (or conditional mutual) information.

        The simplest way to do this is to calculate the entropy of various
        joint distributions and bung them together. See here:

        https://en.wikipedia.org/wiki/Mutual_information
        https://en.wikipedia.org/wiki/Conditional_mutual_information
        """
        # Define a local variable to make it lovely and readable
        h = self.entropy
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
