"""Drawing helpers for discrete Bayesian networks built with pgmpy.

draw_dag shows the structure, visualize_model puts each node's CPT inside it,
and visualize_marginals shows the posterior marginals with observed nodes shaded.
The figures are graphviz Digraphs, which Jupyter displays inline; they need the
Graphviz program (`dot`) installed, see the README.

Adapted from pgmpy_utils.py in probml-utils (https://github.com/probml/probml-utils),
by murphyk@ and Drishttii@, under the MIT License:

    Copyright (c) 2022 Probabilistic machine learning

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import itertools
from html import escape

import numpy as np
from graphviz import Digraph
from pgmpy.inference import VariableElimination


def draw_dag(model):
    """The structure alone: one ellipse per variable, one arrow per edge."""
    g = Digraph()
    for node in model.nodes():
        g.node(str(node))
    for parent, child in model.edges():
        g.edge(str(parent), str(child))
    return g


def _cell(text):
    return f"<TD>{escape(str(text))}</TD>"


def _table(title, rows, width, shaded=False):
    """A graphviz HTML label: the title across the top, then the rows."""
    shade = ' BGCOLOR="#BEBEBE"' if shaded else ""
    head = f'<TR><TD COLSPAN="{width}"{shade}>{escape(str(title))}</TD></TR>'
    body = "".join("<TR>" + "".join(row) + "</TR>" for row in rows)
    return f"<<TABLE>{head}{body}</TABLE>>"


def visualize_model(model):
    """Each node holds its CPT: one row per parent configuration, one column per state."""
    g = Digraph()
    for cpd in model.get_cpds():
        name, parents = cpd.variable, cpd.variables[1:]
        states = cpd.state_names[name]
        # values has shape (states, *parent cards); a row per parent configuration.
        values = cpd.get_values().T
        if not parents:
            rows = [[_cell(s) for s in states], [_cell(f"{p:.2f}") for p in values[0]]]
            width = len(states)
        else:
            configs = itertools.product(*(cpd.state_names[p] for p in parents))
            rows = [[_cell(" ")] + [_cell(s) for s in states]]
            rows += [[_cell(", ".join(map(str, c)))] + [_cell(f"{p:.2f}") for p in row]
                     for c, row in zip(configs, values)]
            width = len(states) + 1
        g.node(str(name), label=_table(name, rows, width))
    for parent, child in model.edges():
        g.edge(str(parent), str(child))
    return g


def get_marginals(model, evidence=None, inference_engine=None):
    """P(node | evidence) for every node; an observed node gets all its mass on the observed state."""
    evidence = evidence or {}
    inference_engine = inference_engine or VariableElimination(model)
    marginals = {}
    for node in model.nodes():
        if node in evidence:
            states = model.get_cpds(node).state_names[node]
            value = evidence[node]
            probs = np.zeros(len(states))
            probs[states.index(value) if value in states else value] = 1.0
            marginals[node] = probs
        else:
            marginals[node] = inference_engine.query([node], evidence=evidence, show_progress=False).values
    return marginals


def visualize_marginals(model, evidence, marginals):
    """Each node holds its marginal; the observed nodes have a shaded title."""
    g = Digraph()
    for node, probs in marginals.items():
        states = model.get_cpds(node).state_names[node]
        rows = [[_cell(s) for s in states], [_cell(round(float(p), 2)) for p in probs]]
        g.node(str(node), label=_table(node, rows, len(states), shaded=node in evidence))
    for parent, child in model.edges():
        g.edge(str(parent), str(child))
    return g
