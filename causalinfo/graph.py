from itertools import product

import networkx as nx
import numpy as np
import pandas as pd

from probability import Variable, ProbabilityTree, JointDist, TreeDistribution


class Equation(object):
    """Maps input variable(s) to output variable(s)"""

    INPUT_LABEL = 'Input'
    OUTPUT_LABEL = 'Output'

    def __init__(self, name, inputs, outputs, strategy_func):
        """Use the strategy_func to map inputs to outputs.

        Args:
            name (str): Identifying name of equation.
            inputs (List[Variable]): Variables to map from.
            outputs (List[Variable]): Variables to map to.
            strategy_func (function): Mapping function.

        """
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
            np.zeros((len(input_states), o.n_states),
                     dtype=float) for o in outputs]

        # Create a lookup table based on the strategy function. Then we can
        # discard the function (very useful if we're interested in pickling).
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
                if not np.isclose(r.sum(), 1.0):
                    raise RuntimeError(
                        "Probabilities must add to 1.0: {}".format(r))

            # Keep this around
            self.lookup[states] = results

    def calculate(self, assignments):
        """Calculate output given variable / state assignments"""
        # Build a tuple of the relevant input states from the set of
        # assignments given.
        states = tuple([assignments[v] for v in self.inputs])

        # Look them up
        try:
            results = self.lookup[states]
        except KeyError:
            raise RuntimeError("Error in {} with key {}".format(self, states))

        # Now, construct a mapping over th output variables and return that.
        return dict(zip(self.outputs, results))

    def __repr__(self):
        return "<{}>".format(self.name)

    def to_frame(self):
        """Output the mapping equation in a nice way

        We do this a long-winded way, but it allows pandas to do the nice
        formatting for us. We generate a row for every single possibility of
        input and outputs states of this variable, then use the pivot_table to
        construct a table for us with nice row/column headings.
        """
        # Create a set of dictionaries/lists for each column
        data = dict([(i_var.name, []) for i_var in self.inputs])
        data.update({self.OUTPUT_LABEL: [], self.INPUT_LABEL: [], self.name: []})

        # A very ugly loop to produce all the probabilities in a nice way.
        # Note that this just reproduces what is already in `self.lookup`.
        # Honestly, I just haven't thought of a better way to get nice output.
        for i_index, i_state in enumerate(self.input_states):
            for o_var, results in zip(self.outputs, self.per_state_results):
                for o_state, o_p in enumerate(results[i_index]):
                    for i_var, s in zip(self.inputs, i_state):
                        data[i_var.name].append(s)
                    data[self.OUTPUT_LABEL].append(o_var.name)
                    data[self.INPUT_LABEL].append(o_state)
                    data[self.name].append(o_p)
        all_data = pd.DataFrame(data=data)

        # The magnificent pivot table function does all the work
        return pd.pivot_table(data=all_data, values=[self.name],
                              index=[i_var.name for i_var in self.inputs],
                              columns=[self.OUTPUT_LABEL, self.INPUT_LABEL])

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.to_frame()._repr_html_()


class CausalGraph(object):
    """A Causal graph built using a set of equations relating variables"""

    def __init__(self, equations):
        # Everythings must be an equation
        self.equations = equations
        self.equations_by_name = {}

        for eq in equations:
            if not isinstance(eq, Equation):
                raise RuntimeError("Non Equation found.")

            if eq.name in equations:
                raise RuntimeError("Equations names must be unique within a graph")
            self.equations_by_name[eq.name] = eq

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
        self.graphviz_prettify(self.causal_network)
        
    def get_equation(self, name):
        return self.equations_by_name(name)

    def graphviz_prettify(self, network):
        """This just makes things pretty for graphviz output."""
        graph_settings = {
            'rankdir': 'LR',
            'dpi': 60,
        }
        network.graph.update(graph_settings)

        for n in network.nodes_iter():
            if isinstance(n, Variable):
                network.node[n]['label'] = n.name
            elif isinstance(n, Equation):
                network.node[n]['shape'] = 'diamond'

    def generate_joint(self, root_dist, do_dist=None):
        """Get the joint distribution, given root & do variables"""
        tree = self.generate_tree(root_dist, do_dist)
        return TreeDistribution(tree)

    def generate_tree(self, root_dist, do_dist=None):
        """Generate the ProbabilityTree"""

        tree = ProbabilityTree()

        # This could be much nicer if we had a way to merge distributions
        if do_dist is None:
            tree.root.add_branches(root_dist)

            # Evaluate all the nodes and recursively construct the
            # ProbabilityTree.
            self._evaluate_branch(tree.root, self.ordered_nodes)

        else:
            tree.root.add_branches(do_dist)
            for b in tree.root.branches:
                b.add_branches(root_dist)
                self._evaluate_branch(b, self.ordered_nodes)

        return tree

    def _evaluate_branch(self, branch, remaining_nodes):
        """Recursively evaluate all possibilities, building a tree"""
        current_eq = None
        next_nodes = []

        # Get the player that we need to evaluate
        for n in remaining_nodes:
            # Node are both equations and variables. We just want the
            # equations.
            if current_eq is None:
                if isinstance(n, Equation):
                    current_eq = n
            else:
                # Leave whatever is left for the next layer of evaluation.
                next_nodes.append(n)

        # No more equations! We're done
        if current_eq is None:
            return

        # Go through each of the branches and evaluate the state.
        for b in branch.branches:
            # Let the player assign output
            outputs = current_eq.calculate(b.assignments)

            # Construct a distribution of these outputs
            distn = JointDist(outputs)

            # Add the branches, and then evaluate using the next set of
            # remaining nodes.
            b.add_branches(distn)
            self._evaluate_branch(b, next_nodes)
