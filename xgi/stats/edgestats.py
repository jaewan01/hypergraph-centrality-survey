"""Edge statistics.

This module is part of the stats package, and it defines edge-level statistics.  That
is, each function defined in this module is assumed to define an edge-quantity mapping.
Each callable defined here is accessible via a `Network` object, or a
:class:`~xgi.core.views.EdgeView` object.  For more details, see the `tutorial
<https://xgi.readthedocs.io/en/stable/api/tutorials/focus_6.html>`_.

Examples
--------

>>> import xgi
>>> H = xgi.Hypergraph([[1, 2, 3], [2, 3, 4, 5], [3, 4, 5]])
>>> H.order()
{0: 2, 1: 3, 2: 2}
>>> H.edges.order.asdict()
{0: 2, 1: 3, 2: 2}

"""

import numpy as np

import xgi

__all__ = [
    "attrs",
    "order",
    "size",
    "node_edge_centrality",
    "degree_centrality",
    "line_expansion_degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "harmonic_centrality",
    "eigenvector_centrality",
    "pagerank_centrality",
    "hypercoreness"
]


def attrs(net, bunch, attr=None, missing=None):
    """Access edge attributes.

    Parameters
    ----------
    net : xgi.Hypergraph
        The network.
    bunch : Iterable
        Nodes in `net`.
    attr : str | None (default)
        If None, return all attributes.  Otherwise, return a single attribute with name
        `attr`.
    missing : Any
        Value to impute in case an edge does not have an attribute with name `attr`.
        Default is None.

    Returns
    -------
    dict
        If attr is None, return a nested dict of the form `{edge: {"attr": val}}`.
        Otherwise, return a simple dict of the form `{edge: val}`.

    Notes
    -----
    When requesting all attributes (i.e. when `attr` is None), no value is imputed.

    Examples
    --------
    >>> import xgi
    >>> H = xgi.Hypergraph()
    >>> edges = [
    ...     ([0, 1], 'one', {'color': 'red'}),
    ...     ([1, 2], 'two', {'color': 'black', 'age': 30}),
    ...     ([2, 3, 4], 'three', {'color': 'blue', 'age': 40}),
    ... ]
    >>> H.add_edges_from(edges)

    Access all attributes as different types.

    >>> H.edges.attrs.asdict() # doctest: +NORMALIZE_WHITESPACE
    {'one': {'color': 'red'},
     'two': {'color': 'black', 'age': 30},
     'three': {'color': 'blue', 'age': 40}}
    >>> H.edges.attrs.asnumpy() # doctest: +NORMALIZE_WHITESPACE
    array([{'color': 'red'},
           {'color': 'black', 'age': 30},
           {'color': 'blue', 'age': 40}],
           dtype=object)

    Access a single attribute as different types.

    >>> H.edges.attrs('color').asdict()
    {'one': 'red', 'two': 'black', 'three': 'blue'}
    >>> H.edges.attrs('color').aslist()
    ['red', 'black', 'blue']

    By default, None is imputed when a node does not have the requested attribute.

    >>> H.edges.attrs('age').asdict()
    {'one': None, 'two': 30, 'three': 40}

    Use `missing` to change the imputed value.

    >>> H.edges.attrs('age', missing=100).asdict()
    {'one': 100, 'two': 30, 'three': 40}

    """
    if isinstance(attr, str):
        return {e: net._edge_attr[e].get(attr, missing) for e in bunch}
    elif attr is None:
        return {e: net._edge_attr[e] for e in bunch}
    else:
        raise ValueError('"attr" must be str or None')


def order(net, bunch, degree=None):
    """Edge order.

    The order of an edge is the number of nodes it contains minus 1.

    Parameters
    ----------
    net : xgi.Hypergraph
        The network.
    bunch : Iterable
        Edges in `net`.
    degree : int | None
        If not None (default), count only those member nodes with the specified degree.

    Returns
    -------
    dict

    See Also
    --------
    size

    Examples
    --------
    >>> import xgi
    >>> H = xgi.Hypergraph([[1, 2, 3], [2, 3, 4, 5], [3, 4, 5]])
    >>> H.edges.order.asdict()
    {0: 2, 1: 3, 2: 2}
    >>> H.edges.order(degree=2).asdict()
    {0: 0, 1: 2, 2: 1}

    """
    if degree is None:
        return {e: len(net._edge[e]) - 1 for e in bunch}
    else:
        return {
            e: sum(len(net._node[n]) == degree for n in net._edge[e]) - 1 for e in bunch
        }


def size(net, bunch, degree=None):
    """Edge size.

    The size of an edge is the number of nodes it contains.

    Parameters
    ----------
    net : xgi.Hypergraph
        The network.
    bunch : Iterable
        Edges in `net`.

    Returns
    -------
    dict

    See Also
    --------
    order

    Examples
    --------
    >>> import xgi
    >>> H = xgi.Hypergraph([[1, 2, 3], [2, 3, 4, 5], [3, 4, 5]])
    >>> H.edges.size.asdict()
    {0: 3, 1: 4, 2: 3}

    """
    if degree is None:
        return {e: len(net._edge[e]) for e in bunch}
    else:
        return {
            e: sum(len(net._node[n]) == degree for n in net._edge[e]) for e in bunch
        }


def node_edge_centrality(
    net,
    bunch,
    f=lambda x: np.power(x, 2),
    g=lambda x: np.power(x, 0.5),
    phi=lambda x: np.power(x, 2),
    psi=lambda x: np.power(x, 0.5),
    max_iter=100,
    tol=1e-6,
):
    """Computes edge centralities.

    Parameters
    ----------
    net : Hypergraph
        The hypergraph of interest
    bunch : Iterable
        Edges in `net`
    f : lambda function, default: x^2
        The function f as described in Tudisco and Higham.
        Must accept a numpy array.
    g : lambda function, default: x^0.5
        The function g as described in Tudisco and Higham.
        Must accept a numpy array.
    phi : lambda function, default: x^2
        The function phi as described in Tudisco and Higham.
        Must accept a numpy array.
    psi : lambda function, default: x^0.5
        The function psi as described in Tudisco and Higham.
        Must accept a numpy array.
    max_iter : int, default: 100
        Number of iterations at which the algorithm terminates
        if convergence is not reached.
    tol : float > 0, default: 1e-6
        The total allowable error in the node and edge centralities.

    Returns
    -------
    dict, dict
        The edge centrality where keys are the edge IDs and values are
        associated centralities.

    Notes
    -----
    In the paper from which this was taken, it is more general in that it includes
    general functions for both nodes and edges, nodes and edges may be weighted,
    and one can choose different norms for normalization.

    This method does not output the node centralities even though they are computed.

    References
    ----------
    Node and edge nonlinear eigenvector centrality for hypergraphs,
    Francesco Tudisco & Desmond J. Higham,
    https://doi.org/10.1038/s42005-021-00704-2
    """
    _, c = xgi.node_edge_centrality(net, f, g, phi, psi, max_iter, tol)
    return {e: c[e] for e in c if e in bunch}

def degree_centrality(net, bunch):
    """Compute the degree centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Centrality in affiliation networks,
    Katherine Faust,
    https://doi.org/10.1016/S0378-8733(96)00300-0
    """
    c = xgi.degree_centrality(net, target="edge")
    return {e: c[e] for e in c if e in bunch}

def line_expansion_degree_centrality(net, bunch):
    """Compute the line-expansion degree centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Centrality in affiliation networks,
    Katherine Faust,
    https://doi.org/10.1016/S0378-8733(96)00300-0
    """
    c = xgi.line_expansion_degree_centrality(net)
    return {e: c[e] for e in c if e in bunch}

def closeness_centrality(net, bunch):
    """Compute the closeness centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Centrality in affiliation networks,
    Katherine Faust,
    https://doi.org/10.1016/S0378-8733(96)00300-0
    """
    c = xgi.closeness_centrality(net, target="edge")
    return {e: c[e] for e in c if e in bunch}

def betweenness_centrality(net, bunch):
    """Compute the betweenness centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Centrality in affiliation networks,
    Katherine Faust,
    https://doi.org/10.1016/S0378-8733(96)00300-0
    """
    c = xgi.betweenness_centrality(net, target="edge")
    return {e: c[e] for e in c if e in bunch}

def harmonic_centrality(net, bunch):
    """Compute the harmonic centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Axioms for Centrality,
    P. Boldi, S. Vigna, 
    https://doi.org/10.1080/15427951.2013.865686 
    """
    c = xgi.harmonic_centrality(net, target="edge")
    return {e: c[e] for e in c if e in bunch}

def eigenvector_centrality(net, bunch, max_iter=100, tol=1e-6):
    """Compute the eigenvector centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Centrality in affiliation networks,
    Katherine Faust,
    https://doi.org/10.1016/S0378-8733(96)00300-0
    """
    c = xgi.eigenvector_centrality(net, target="edge", max_iter=max_iter, tol=tol)
    return {e: c[e] for e in c if e in bunch}

def pagerank_centrality(net, bunch, alpha=0.9, max_iter=100, tol=1e-6):
    """Compute the PageRank centrality of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.
    alpha : float, default 0.9
        Teleportation parameter (1 - alpha is the teleport probability).
    max_iter : int, optional
        Number of iterations at which the algorithm terminates
        if convergence is not reached. By default, 100.
    tol : float > 0, optional
        The total allowable error in the node and edge centralities.
        By default, 1e-6.

    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are centralities.

    References
    ----------
    Centrality in affiliation networks,
    Katherine Faust,
    https://doi.org/10.1016/S0378-8733(96)00300-0
    """
    c = xgi.eigenvector_centrality(net, target="edge", max_iter=max_iter, tol=tol)
    return {e: c[e] for e in c if e in bunch}

def hypercoreness(net, bunch):
    """Compute the hypercoreness of a hypergraph.

    Parameters
    ----------
    net : xgi.Hypergraph
        The hypergraph of interest.
    bunch : Iterable
        Nodes in `net`.

    Returns
    -------
    dict

    References
    ----------
    Axioms for Centrality,
    P. Boldi, S. Vigna, 
    https://doi.org/10.1080/15427951.2013.865686 
    """
    c = xgi.hypercoreness(net, target="edge")
    return {e: c[e] for e in c if e in bunch}