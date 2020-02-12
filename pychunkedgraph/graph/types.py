from typing import Dict
from typing import Iterable
from collections import namedtuple

import numpy as np

from .utils import basetypes

empty_1d = np.empty(0, dtype=basetypes.NODE_ID)
empty_2d = np.empty((0, 2), dtype=basetypes.NODE_ID)


class Node:
    """
    Represents a node ID when creating new hierarchy after merge/split.
    This is required because the new IDs are not updated in the backend
    until the operation is complete.
    """

    def __init__(
        self,
        node_id: basetypes.NODE_ID,
        *,
        is_new: bool = True,
        parent_id: basetypes.NODE_ID = None,
        children: Iterable = empty_1d.copy(),
        atomic_cross_edges: Dict = dict(),
    ):
        """
        `is_new` flag to check if a node is to be updated/written to storage.
        if False, meant to be used as cache to avoid costly lookups.
        """
        self.node_id = node_id
        self.is_new = is_new
        self.parent_id = parent_id
        self.children = children
        self.atomic_cross_edges = atomic_cross_edges

    def __str__(self):
        return f"({self.node_id}:{self.parent_id}:{self.children})"

    def __repr__(self):
        return f"({self.node_id}:{self.parent_id}:{self.children})"


"""
An Agglomeration is syntactic sugar for representing
a level 2 ID and it's supervoxels and edges.
`in_edges`
    edges between supervoxels belonging to the agglomeration.
`out_edges`
    edges between supervoxels of agglomeration
    and neighboring agglomeration.
`cross_edges_d`
    dict of cross edges {layer: cross_edges_relevant_on_that_layer}
"""
_agglomeration_fields = (
    "node_id",
    "supervoxels",
    "in_edges",
    "out_edges",
    "cross_edges",
    "cross_edges_d",
)
_agglomeration_defaults = (
    None,
    empty_1d.copy(),
    empty_2d.copy(),
    empty_2d.copy(),
    empty_2d.copy(),
    {},
)
Agglomeration = namedtuple(
    "Agglomeration", _agglomeration_fields, defaults=_agglomeration_defaults,
)