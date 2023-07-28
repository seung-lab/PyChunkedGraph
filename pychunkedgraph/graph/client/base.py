from datetime import datetime
from abc import ABC
from abc import abstractmethod
from typing import Iterable, Union, Optional, Any, Dict

import numpy as np


class SimpleClient(ABC):
    """
    Abstract class for interacting with backend data store where the chunkedgraph is stored.
    E.g., BigTableClient for using big table as storage.
    """
    
    @abstractmethod
    def create_graph(self) -> None:
        """Initialize the graph and store associated meta."""
    
    @abstractmethod
    def add_graph_version(self, version):
        """Add a version to the graph."""
    
    @abstractmethod
    def read_graph_version(self):
        """Read stored graph version."""
    
    @abstractmethod
    def update_graph_meta(self, meta):
        """Update stored graph meta."""
    
    @abstractmethod
    def read_graph_meta(self):
        """Read stored graph meta."""
    
    @abstractmethod
    def read_nodes(
            self,
            start_id=None,
            end_id=None,
            node_ids=None,
            properties=None,
            start_time=None,
            end_time=None,
            end_time_inclusive=False,
    ):
        """
        Read nodes and their properties.
        Accepts a range of node IDs or specific node IDs.
        """
    
    @abstractmethod
    def read_node(
            self,
            node_id,
            properties=None,
            start_time=None,
            end_time=None,
            end_time_inclusive=False,
    ):
        """Read a single node and it's properties."""
    
    @abstractmethod
    def write_nodes(self, nodes):
        """Writes/updates nodes (IDs along with properties)."""
    
    @abstractmethod
    def write(
            self,
            rows: Iterable[dict[str, Union[bytes, dict[str, Iterable[Any]]]]],
            root_ids: Optional[
                Union[np.uint64, Iterable[np.uint64]]
            ] = None,
            operation_id: Optional[np.uint64] = None,
            slow_retry: bool = True,
            block_size: int = 2000,
    ):
        """Writes a list of mutated rows in bulk
        WARNING: If <rows> contains the same row (same row_key) and column
        key two times only the last one is effectively written to the backend data store
        (even when the mutations were applied to different columns)
        --> no versioning!
        :param rows: list
            list of mutated rows
        :param root_ids: list if uint64
        :param operation_id: uint64 or None
            operation_id (or other unique id) that *was* used to lock the root
            the bulk write is only executed if the root is still locked with
            the same id.
        :param slow_retry: bool
        :param block_size: int
        """
    
    @abstractmethod
    def mutate_row(
            self,
            row_key: bytes,
            val_dict: Dict,
            time_stamp: Optional[datetime] = None,
    ) -> dict[str, Union[bytes, dict[str, Iterable[Any]]]]:
        """Mutates a single row (doesn't write to the backend storage, just returns the row with mutated
        data without writing to the backend storage)."""
    
    @abstractmethod
    def lock_root(self, node_id, operation_id):
        """Locks root node with operation_id to prevent race conditions."""
    
    @abstractmethod
    def lock_roots(self, node_ids, operation_id):
        """Locks root nodes to prevent race conditions."""
    
    @abstractmethod
    def lock_root_indefinitely(self, node_id, operation_id):
        """Locks root node with operation_id to prevent race conditions."""
    
    @abstractmethod
    def lock_roots_indefinitely(self, node_ids, operation_id):
        """
        Locks root nodes indefinitely to prevent structural damage to graph.
        This scenario is rare and needs asynchronous fix or inspection to unlock.
        """
    
    @abstractmethod
    def unlock_root(self, node_id, operation_id):
        """Unlocks root node that is locked with operation_id."""
    
    @abstractmethod
    def unlock_indefinitely_locked_root(self, node_id, operation_id):
        """Unlocks root node that is indefinitely locked with operation_id."""
    
    @abstractmethod
    def renew_lock(self, node_id, operation_id):
        """Renews existing node lock with operation_id for extended time."""
    
    @abstractmethod
    def renew_locks(self, node_ids, operation_id):
        """Renews existing node locks with operation_id for extended time."""
    
    @abstractmethod
    def get_lock_timestamp(self, node_ids, operation_id):
        """Reads timestamp from lock row to get a consistent timestamp."""
    
    @abstractmethod
    def get_consolidated_lock_timestamp(self, root_ids, operation_ids):
        """Minimum of multiple lock timestamps."""
    
    @abstractmethod
    def get_compatible_timestamp(self, time_stamp):
        """Datetime time stamp compatible with client's services."""


class ClientWithIDGen(SimpleClient):
    """
    Abstract class for client to backend data store that has support for generating IDs.
    If not, something else can be used but these methods need to be implemented.
    Eg., Big Table row cells can be used to generate unique IDs.
    """
    
    @abstractmethod
    def create_node_ids(self, chunk_id, size):
        """Generate a range of unique IDs in the chunk."""
    
    @abstractmethod
    def create_node_id(self, chunk_id):
        """Generate a unique ID in the chunk."""
    
    @abstractmethod
    def get_max_node_id(self, chunk_id):
        """Gets the current maximum node ID in the chunk."""
    
    @abstractmethod
    def create_operation_id(self):
        """Generate a unique operation ID."""
    
    @abstractmethod
    def get_max_operation_id(self):
        """Gets the current maximum operation ID."""


class OperationLogger(ABC):
    """
    Abstract class for interacting with backend data store where the operation logs are stored.
    E.g., BigTableClient can be used to store logs in Google BigTable.
    """
    
    # TODO add functions for writing
    
    @abstractmethod
    def read_log_entry(self, operation_id: int) -> None:
        """Read log entry for a given operation ID."""
    
    @abstractmethod
    def read_log_entries(self, operation_ids) -> None:
        """Read log entries for given operation IDs."""
