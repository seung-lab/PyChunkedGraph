"""
helper functions for edge stuff
"""

from typing import Dict
from typing import Tuple
from typing import Iterable
from typing import Callable
from typing import Optional

import numpy as np

from . import Edges
from . import EDGE_TYPES
from ..types import empty_2d
from ..utils import basetypes
from ..chunks import utils as chunk_utils
from ..meta import ChunkedGraphMeta


def concatenate_chunk_edges(chunk_edge_dicts: Iterable) -> Dict:
    """combine edge_dicts of multiple chunks into one edge_dict"""
    edges_dict = {}
    for edge_type in EDGE_TYPES:
        sv_ids1 = [np.array([], dtype=basetypes.NODE_ID)]
        sv_ids2 = [np.array([], dtype=basetypes.NODE_ID)]
        affinities = [np.array([], dtype=basetypes.EDGE_AFFINITY)]
        areas = [np.array([], dtype=basetypes.EDGE_AREA)]
        for edge_d in chunk_edge_dicts:
            edges = edge_d[edge_type]
            sv_ids1.append(edges.node_ids1)
            sv_ids2.append(edges.node_ids2)
            affinities.append(edges.affinities)
            areas.append(edges.areas)

        sv_ids1 = np.concatenate(sv_ids1)
        sv_ids2 = np.concatenate(sv_ids2)
        affinities = np.concatenate(affinities)
        areas = np.concatenate(areas)
        edges_dict[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
    return edges_dict


def concatenate_cross_edge_dicts(edges_ds: Iterable[Dict]) -> Dict:
    """Combines cross chunk edge dicts of form {layer id : edge list}."""
    from collections import defaultdict

    result_d = defaultdict(list)

    for edges_d in edges_ds:
        for layer, edges in edges_d.items():
            result_d[layer].append(edges)

    for layer, edge_lists in result_d.items():
        result_d[layer] = np.concatenate(edge_lists)
    return result_d


def merge_cross_edge_dicts(x_edges_d1: Dict, x_edges_d2: Dict) -> Dict:
    """
    Combines two cross chunk dictionaries of form
    {node_id: {layer id : edge list}}.
    """
    node_ids = np.unique(list(x_edges_d1.keys()) + list(x_edges_d2.keys()))
    result_d = {}
    for node_id in node_ids:
        cross_edge_ds = [x_edges_d1.get(node_id, {}), x_edges_d2.get(node_id, {})]
        result_d[node_id] = concatenate_cross_edge_dicts(cross_edge_ds)
    return result_d


def categorize_edges(
    meta: ChunkedGraphMeta, supervoxels: np.ndarray, edges: Edges
) -> Tuple[Edges, Edges, Edges]:
    """
    Find edges and categorize them into:
    `in_edges`
        between given supervoxels
        (sv1, sv2) - sv1 in supervoxels and sv2 in supervoxels
    `out_edges`
        originating from given supervoxels but within chunk
        (sv1, sv2) - sv1 in supervoxels and sv2 not in supervoxels
    `cross_edges`
        originating from given supervoxels but crossing chunk boundary
    """
    mask1 = np.isin(edges.node_ids1, supervoxels)
    mask2 = np.isin(edges.node_ids2, supervoxels)
    in_mask = mask1 & mask2
    out_mask = mask1 & ~mask2

    in_edges = edges[in_mask]
    all_out_edges = edges[out_mask]  # out_edges + cross_edges

    edge_layers = get_cross_chunk_edges_layer(meta, all_out_edges.get_pairs())
    cross_edges_mask = edge_layers > 1
    out_edges = all_out_edges[~cross_edges_mask]
    cross_edges = all_out_edges[cross_edges_mask]
    return (in_edges, out_edges, cross_edges)


def categorize_edges_v2(
    meta: ChunkedGraphMeta,
    edges: Edges,
    get_sv_parents: Callable,
    graph_id:str,
    sv_parent_d:Dict,
) -> Tuple[Edges, Edges, Edges]:
    """Faster version of categorize_edges(), avoids looping over L2 IDs."""

    import fastremap
    from ...logging.log_db import TimeIt

    # TODO (investigate) these 2 lines take 99% for the time
    with TimeIt("1sv_parents", graph_id=graph_id, n_ids=len(edges.node_ids1)):
        node_ids1 = get_sv_parents(edges.node_ids1)

    with TimeIt("2sv_parents", graph_id=graph_id, n_ids=len(edges.node_ids2)):
        node_ids2 = get_sv_parents(edges.node_ids2)

    with TimeIt("1remap", graph_id=graph_id, n_ids=len(edges.node_ids1)):
        node_ids3 = fastremap.remap(edges.node_ids1, sv_parent_d, preserve_missing_labels=True, in_place=True)

    with TimeIt("2remap", graph_id=graph_id, n_ids=len(edges.node_ids1)):
        node_ids4 = fastremap.remap(edges.node_ids2, sv_parent_d, preserve_missing_labels=True, in_place=True)

    assert node_ids1 == node_ids3
    assert node_ids2 == node_ids4

    layer_mask1 = chunk_utils.get_chunk_layers(meta, node_ids1) > 1
    nodes_mask = node_ids1 == node_ids2

    in_edges = edges[nodes_mask]
    all_out_ = edges[layer_mask1 & ~nodes_mask]

    cx_layers = get_cross_chunk_edges_layer(meta, all_out_.get_pairs())

    cx_mask = cx_layers > 1
    out_edges = all_out_[~cx_mask]
    cross_edges = all_out_[cx_mask]
    return (in_edges, out_edges, cross_edges)


def get_cross_chunk_edges_layer(meta: ChunkedGraphMeta, cross_edges: Iterable):
    """Computes the layer in which a cross chunk edge becomes relevant.
    I.e. if a cross chunk edge links two nodes in layer 4 this function
    returns 3.
    :param cross_edges: n x 2 array
        edges between atomic (level 1) node ids
    :return: array of length n
    """
    if len(cross_edges) == 0:
        return np.array([], dtype=int)
    cross_chunk_edge_layers = np.ones(len(cross_edges), dtype=int)
    coords0 = chunk_utils.get_chunk_coordinates_multiple(meta, cross_edges[:, 0])
    coords1 = chunk_utils.get_chunk_coordinates_multiple(meta, cross_edges[:, 1])

    for _ in range(2, meta.layer_count):
        edge_diff = np.sum(np.abs(coords0 - coords1), axis=1)
        cross_chunk_edge_layers[edge_diff > 0] += 1
        coords0 = coords0 // meta.graph_config.FANOUT
        coords1 = coords1 // meta.graph_config.FANOUT
    return cross_chunk_edge_layers


def filter_min_layer_cross_edges(
    meta: ChunkedGraphMeta, cross_edges_d: Dict, node_layer: int = 2
) -> Tuple[int, Iterable]:
    """
    Given a dict of cross chunk edges {layer: edges}
    Return the first layer with cross edges.
    """
    for layer in range(node_layer, meta.layer_count):
        edges_ = cross_edges_d.get(layer, empty_2d)
        if edges_.size:
            return (layer, edges_)
    return (meta.layer_count, edges_)


def filter_min_layer_cross_edges_multiple(
    meta: ChunkedGraphMeta, l2id_atomic_cross_edges_ds: Iterable, node_layer: int = 2
) -> Tuple[int, Iterable]:
    """
    Given a list of dicts of cross chunk edges [{layer: edges}]
    Return the first layer with cross edges.
    """
    min_layer = meta.layer_count
    for edges_d in l2id_atomic_cross_edges_ds:
        layer_, _ = filter_min_layer_cross_edges(meta, edges_d, node_layer=node_layer)
        min_layer = min(min_layer, layer_)
    edges = [empty_2d]
    for edges_d in l2id_atomic_cross_edges_ds:
        edges.append(edges_d.get(min_layer, empty_2d))
    return min_layer, np.concatenate(edges)


def get_edges_status(cg, edges: Iterable, time_stamp: Optional[float] = None):
    from ...utils.general import in2d

    coords0 = chunk_utils.get_chunk_coordinates_multiple(cg.meta, edges[:, 0])
    coords1 = chunk_utils.get_chunk_coordinates_multiple(cg.meta, edges[:, 1])

    coords = np.concatenate([np.array(coords0), np.array(coords1)])
    bbox = [np.min(coords, axis=0), np.max(coords, axis=0)]
    bbox[1] += 1

    root_ids = set(
        cg.get_roots(edges.ravel(), assert_roots=True, time_stamp=time_stamp)
    )
    sg_edges = cg.get_subgraph(
        root_ids,
        bbox=bbox,
        bbox_is_coordinate=False,
        edges_only=True,
    )
    existence_status = in2d(edges, sg_edges)
    edge_layers = cg.get_cross_chunk_edges_layer(edges)
    active_status = []
    for layer in np.unique(edge_layers):
        layer_edges = edges[edge_layers == layer]
        edges_parents = cg.get_roots(
            layer_edges.ravel(), time_stamp=time_stamp, stop_layer=layer + 1
        ).reshape(-1, 2)
        mask = edges_parents[:, 0] == edges_parents[:, 1]
        active_status.extend(mask)
    active_status = np.array(active_status, dtype=bool)
    return existence_status, active_status
