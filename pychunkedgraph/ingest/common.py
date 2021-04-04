from typing import Dict
from typing import Tuple
from typing import Sequence

from .manager import IngestionManager
from .ran_agglomeration import read_raw_edge_data
from .ran_agglomeration import read_raw_agglomeration_data
from ..io.edges import get_chunk_edges
from ..io.components import get_chunk_components


def get_atomic_chunk_data(
    imanager: IngestionManager, coord: Sequence[int]
) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        read_raw_edge_data(imanager, coord)
        if imanager.config.USE_RAW_EDGES
        else get_chunk_edges(imanager.cg_meta.data_source.EDGES, [coord])
    )
    mapping = (
        read_raw_agglomeration_data(imanager, coord)
        if imanager.config.USE_RAW_COMPONENTS
        else get_chunk_components(
            imanager.cg_meta.data_source.COMPONENTS, coord
        )
    )
    return chunk_edges, mapping
