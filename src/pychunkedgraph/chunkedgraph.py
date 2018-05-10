import numpy as np
import time
import datetime
import os
import networkx as nx
import pytz

from google.cloud import bigtable

# global variables
HOME = os.path.expanduser("~")
N_DIGITS_UINT64 = len(str(np.iinfo(np.uint64).max))
UTC = pytz.UTC

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = HOME + "/.cloudvolume/secrets/google-secret.json"


def serialize_node_id(node_id):
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    s_node_id = "%.20d" % node_id
    s_node_id = serialize_key(s_node_id)
    return s_node_id


def serialize_key(key):
    """ Serializes a key to be ingested by a bigtable table row

    :param key: str
    :return: str
    """
    return key.encode("utf-8")


def mutate_row(table, row_key, column_family_id, val_dict, time_stamp=None):
    """

    :param table: bigtable table instance
    :param row_key: serialized bigtable row key
    :param column_family_id: str
        serialized column family id
    :param val_dict: dict
    :param time_stamp: None or datetime
    :return: list
    """
    row = table.row(row_key)

    for column, value in val_dict.items():
        row.set_cell(column_family_id=column_family_id, column=column,
                     value=value, timestamp=time_stamp)
    return row


def get_chunk_id_from_node_id(node_id):
    """ Extracts z, y, x, l

    :param node_id: int
    :return: list of ints
    """

    return np.frombuffer(np.uint64(node_id), dtype=np.uint8)[4:]


def test_if_nodes_are_in_same_chunk(node_ids):
    """ Test whether two nodes are in the same chunk

    :param node_ids: list of two ints
    :return: bool
    """
    assert len(node_ids) == 2

    return np.frombuffer(node_ids[0], dtype=np.uint32)[1] == \
           np.frombuffer(node_ids[1], dtype=np.uint32)[1]


class ChunkedGraph(object):
    def __init__(self, instance_id="pychunkedgraph",
                 project_id="neuromancer-seung-import",
                 chunk_size=(512, 512, 64), dev_mode=False):

        self._client = bigtable.Client(project=project_id, admin=True)
        self._instance = self.client.instance(instance_id)

        if dev_mode:
            self._table = self.instance.table("pychgtable_dev")
        else:
            self._table = self.instance.table("pychgtable")

        self._fan_out = 2
        self._chunk_size = np.array(chunk_size)

    @property
    def client(self):
        return self._client

    @property
    def instance(self):
        return self._instance

    @property
    def table(self):
        return self._table

    @property
    def family_id(self):
        return "0"

    @property
    def fan_out(self):
        return self._fan_out

    @property
    def chunk_size(self):
        return self._chunk_size

    def get_cg_id_from_rg_id(self, atomic_id):
        """ Extracts ChunkedGraph id from RegionGraph id

        :param atomic_id: int
        :return: int
        """
        # There might be multiple chunk ids for a single rag id because
        # rag supervoxels get split at chunk boundaries. Here, only one
        # chunk id needs to be traced to the top to retrieve the
        # agglomeration id that they both belong to
        r = self.table.read_row(serialize_node_id(atomic_id))
        return np.frombuffer(r.cells[self.family_id][serialize_key("cg_id")][0].value,
                             dtype=np.uint64)[0]

    def find_unique_node_id(self, chunk_id):
        """ Finds a unique node id for the given chunk

        :param chunk_id: uint32
        :return: uint64
        """

        randint = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
        node_id = (chunk_id << 32).astype(np.uint64) + randint

        while self.table.read_row(serialize_node_id(node_id)) is not None:
            randint = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
            node_id = (chunk_id << 32).astype(np.uint64) + randint

        return node_id

    def read_row(self, node_id, key, idx=0, dtype=np.uint64):
        row = self.table.read_row(serialize_node_id(node_id))
        return np.frombuffer(row.cells[self.family_id][serialize_key(key)][idx].value, dtype=dtype)

    def read_rows(self, node_ids, key, dtype=np.uint64):
        results = []

        for node_id in node_ids:
            results.append(np.frombuffer(self.table.read_row(
                serialize_node_id(node_id).cells[self.family_id][
                serialize_key(key)]), dtype=dtype))

        return results

    def add_atomic_edges_in_chunks(self, edge_ids, cross_edge_ids, edge_affs,
                                   cross_edge_affs, cg2rg_dict, rg2cg_dict,
                                   time_stamp=None):
        """ Creates atomic edges between supervoxels and first
            abstraction layer """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # Catch trivial case
        if len(edge_ids) == 0:
            return 0

        # Write rg2cg mapping to table
        rows = []
        for rg_id in rg2cg_dict.keys():
            # Create node
            val_dict = {"cg_id": np.array([rg2cg_dict[rg_id]]).tobytes()}

            rows.append(mutate_row(self.table, serialize_node_id(rg_id),
                                   self.family_id, val_dict))
        status = self.table.mutate_rows(rows)

        # Make parent id creation easier
        z, y, x, l = get_chunk_id_from_node_id(edge_ids[0, 0])
        parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, l+1],
                                                dtype=np.uint8),
                                       dtype=np.uint32)

        # Get connected component within the chunk
        chunk_g = nx.from_edgelist(edge_ids)
        chunk_g.add_nodes_from(np.unique(cross_edge_ids[:, 0]))
        ccs = list(nx.connected_components(chunk_g))

        # print("%d ccs detected" % (len(ccs)))

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the print / speed measurement
        time_start = time.time()
        for i_cc, cc in enumerate(ccs):
            if node_c > 0:
                dt = time.time() - time_start
                print("%5d at %5d - %.5fs             " %
                      (i_cc, node_c, dt / node_c), end="\r")

            rows = []

            node_ids = np.array(list(cc))

            # Create parent id
            parent_id = parent_id_base.copy()
            parent_id[0] = i_cc
            parent_id = np.frombuffer(parent_id, dtype=np.uint64)
            parent_id_b = parent_id.tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # print("Node:", node_id)
                # Extract edges relevant to this node
                edge_col1_mask = edge_ids[:, 0] == node_id
                edge_col2_mask = edge_ids[:, 1] == node_id

                # Cross edges are ordered to always point OUT of the chunk
                cross_edge_mask = cross_edge_ids[:, 0] == node_id

                parent_cross_edges = np.concatenate([parent_cross_edges,
                                                     cross_edge_ids[cross_edge_mask]])

                connected_partner_ids = np.concatenate([edge_ids[edge_col1_mask][:, 1],
                                                        edge_ids[edge_col2_mask][:, 0],
                                                        cross_edge_ids[cross_edge_mask][:, 1]]).tobytes()

                connected_partner_affs = np.concatenate([edge_affs[np.logical_or(edge_col1_mask, edge_col2_mask)],
                                                         cross_edge_affs[cross_edge_mask]]).tobytes()

                # Create node
                val_dict = {"atomic_partners": connected_partner_ids,
                            "atomic_affinities": connected_partner_affs,
                            "parents": parent_id_b,
                            "rg_id": np.array([cg2rg_dict[node_id]]).tobytes()}

                rows.append(mutate_row(self.table, serialize_node_id(node_id),
                                       self.family_id, val_dict))
                node_c += 1

            # Create parent node
            val_dict = {"children": node_ids.tobytes(),
                        "atomic_cross_edges": parent_cross_edges.tobytes()}

            rows.append(mutate_row(self.table, serialize_node_id(parent_id),
                                   self.family_id, val_dict))

            node_c += 1

            status = self.table.mutate_rows(rows)

        try:
            dt = time.time() - time_start
            print("Average time: %.5fs / node; %.5fs / edge - Number of edges: %6d, %6d" %
                  (dt / node_c, dt / len(edge_ids), len(edge_ids), len(cross_edge_ids)))
        except:
            print("WARNING: NOTHING HAPPENED")

    def add_layer(self, layer_id, child_chunk_coords, time_stamp=None):
        """ Creates all hierarchy layers above the first abstract layer """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        # 1 ----------
        # The first part is concerned with reading data from the child nodes
        # of this layer and pre-processing it for the second part

        atomic_child_ids = np.array([], dtype=np.uint64)    # ids in lowest layer
        child_ids = np.array([], dtype=np.uint64)   # ids in layer one below this one
        atomic_partner_id_dict = {}
        atomic_child_id_dict = {}

        leftover_atomic_edges = {}

        for chunk_coord in child_chunk_coords:
            # Get start and end key
            x, y, z = chunk_coord
            node_id_base = np.array([0, 0, 0, 0, z, y, x, layer_id - 1], dtype=np.uint8)
            node_id_base_next = node_id_base.copy()
            node_id_base_next[-2] += 1

            start_key = serialize_node_id(np.frombuffer(node_id_base, dtype=np.uint64)[0])
            end_key = serialize_node_id(np.frombuffer(node_id_base_next, dtype=np.uint64)[0])

            print(start_key, end_key)

            # Set up read
            range_read = self.table.read_rows(start_key=start_key,
                                              end_key=end_key,
                                              end_inclusive=False)
            # Execute read
            range_read.consume_all()

            # Loop through nodes from this chunk
            for row_key, row_data in range_read.rows.items():
                atomic_edges = np.frombuffer(row_data.cells[self.family_id]["atomic_cross_edges".encode("utf-8")][0].value, dtype=np.uint64).reshape(-1, 2)
                atomic_partner_id_dict[int(row_key)] = atomic_edges[:, 1]
                atomic_child_id_dict[int(row_key)] = atomic_edges[:, 0]

                atomic_child_ids = np.concatenate([atomic_child_ids, atomic_edges[:, 0]])
                child_ids = np.concatenate([child_ids, np.array([row_key] * len(atomic_edges[:, 0]), dtype=np.uint64)])

        # Extract edges from remaining cross chunk edges
        # and maintain unused cross chunk edges
        edge_ids = np.array([], np.uint64).reshape(0, 2)

        u_atomic_child_ids = np.unique(atomic_child_ids)
        atomic_partner_id_dict_keys = list(atomic_partner_id_dict.keys())
        time_start = time.time()

        time_segs = [[], [], []]
        for i_child_key, child_key in enumerate(atomic_partner_id_dict_keys):
            if i_child_key % 20 == 1:
                dt = time.time() - time_start
                eta = dt / i_child_key * len(atomic_partner_id_dict_keys) - dt
                print("%5d - dt: %.3fs - eta: %.3fs - %.4fs - %.4fs - %.4fs           " %
                      (i_child_key, dt, eta, np.mean(time_segs[0]), np.mean(time_segs[1]), np.mean(time_segs[2])), end="\r")

            this_atomic_partner_ids = atomic_partner_id_dict[child_key]
            this_atomic_child_ids = atomic_child_id_dict[child_key]

            time_seg = time.time()

            leftover_mask = ~np.in1d(this_atomic_partner_ids, u_atomic_child_ids)

            time_segs[0].append(time.time() - time_seg)
            time_seg = time.time()
            leftover_atomic_edges[child_key] = np.concatenate([this_atomic_child_ids[leftover_mask, None],
                                                               this_atomic_partner_ids[leftover_mask, None]], axis=1)

            time_segs[1].append(time.time() - time_seg)
            time_seg = time.time()

            partners = np.unique(child_ids[np.in1d(atomic_child_ids, this_atomic_partner_ids)])
            these_edges = np.concatenate([np.array([child_key] * len(partners), dtype=np.uint64)[:, None], partners[:, None]], axis=1)

            edge_ids = np.concatenate([edge_ids, these_edges])

            time_segs[2].append(time.time() - time_seg)

        # 2 ----------
        # The second part finds connected components, writes the parents to
        # BigTable and updates the childs

        # Make parent id creation easier
        x, y, z = np.min(child_chunk_coords, axis=0)
        parent_id_base = np.frombuffer(np.array([0, 0, 0, 0, z, y, x, layer_id], dtype=np.uint8), dtype=np.uint32)

        # Extract connected components
        chunk_g = nx.from_edgelist(edge_ids)
        chunk_g.add_nodes_from(atomic_partner_id_dict_keys)

        ccs = list(nx.connected_components(chunk_g))

        # Add rows for nodes that are in this chunk
        # a connected component at a time
        node_c = 0  # Just a counter for the print / speed measurement
        time_start = time.time()
        for i_cc, cc in enumerate(ccs):
            if node_c > 0:
                dt = time.time() - time_start
                print("%5d at %5d - %.5fs             " %
                      (i_cc, node_c, dt / node_c), end="\r")

            rows = []

            node_ids = np.array(list(cc))

            # Create parent id
            parent_id = parent_id_base.copy()
            parent_id[0] = i_cc
            parent_id = np.frombuffer(parent_id, dtype=np.uint64)
            parent_id_b = parent_id.tobytes()

            parent_cross_edges = np.array([], dtype=np.uint64).reshape(0, 2)

            # Add rows for nodes that are in this chunk
            for i_node_id, node_id in enumerate(node_ids):
                # Extract edges relevant to this node
                parent_cross_edges = np.concatenate([parent_cross_edges,
                                                     leftover_atomic_edges[node_id]])

                # Create node
                val_dict = {"parents": parent_id_b}

                rows.append(mutate_row(self.table, serialize_node_id(node_id),
                                       self.family_id, val_dict))
                node_c += 1

            # Create parent node
            val_dict = {"children": node_ids.tobytes(),
                        "atomic_cross_edges": parent_cross_edges.tobytes()}

            rows.append(mutate_row(self.table, serialize_node_id(parent_id),
                                   self.family_id, val_dict))

            node_c += 1

            status = self.table.mutate_rows(rows)

        try:
            dt = time.time() - time_start
            print("Average time: %.5fs / node; %.5fs / edge - Number of edges: %6d" %
                  (dt / node_c, dt / len(edge_ids), len(edge_ids)))
        except:
            print("WARNING: NOTHING HAPPENED")

    def get_parent(self, node_id, time_stamp=None):
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        parent_key = serialize_key("parents")

        row = self.table.read_row(serialize_node_id(node_id))

        if parent_key in row.cells[self.family_id]:
            for parent_entry in row.cells[self.family_id][parent_key]:
                if parent_entry.timestamp > time_stamp:
                    continue
                else:
                    return np.frombuffer(parent_entry.value, dtype=np.uint64)[0]
        else:
            return None

        raise Exception("Did not find a valid parent for %d with"
                        " the given time stamp" % node_id)

    def get_children(self, node_id):
        return self.read_row(node_id, "children", dtype=np.uint64)

    def get_root(self, atomic_id, time_stamp=None, is_cg_id=False):
        """ Takes an atomic id and returns the associated agglomeration ids

        :param atomic_id: int
        :param time_stamp: None or datetime
        :return: int
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        if not is_cg_id:
            atomic_id = self.get_cg_id_from_rg_id(atomic_id)

        parent_id = atomic_id

        while True:
            temp_parent_id = self.get_parent(parent_id, time_stamp)
            if temp_parent_id is None:
                break
            else:
                parent_id = temp_parent_id

        return parent_id

    def read_agglomeration_id_history(self, agglomeration_id, time_stamp=None):
        """ Returns all agglomeration ids agglomeration_id was part of

        :param agglomeration_id: int
        :param time_stamp: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole history
        :return: array of int
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.min

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        id_working_set = [agglomeration_id]
        id_history = [agglomeration_id]

        former_parent_key = serialize_key("former_parents")
        new_parent_key = serialize_key("new_parents")

        while len(id_working_set) > 0:
            next_id = id_working_set[0]
            del id_working_set[0]

            r = self.table.read_row(serialize_node_id(next_id))
            if new_parent_key in r.cells[self.family_id]:
                new_parent_ids = np.frombuffer(
                    r.cells[self.family_id][new_parent_key][0].value,
                    dtype=np.uint64)
                id_working_set.extend(new_parent_ids)
                id_history.extend(new_parent_ids)

            if former_parent_key in r.cells[self.family_id]:
                if time_stamp < r.cells[self.family_id][new_parent_key][0].timestamp:
                    former_parent_ids = np.frombuffer(
                        r.cells[self.family_id][new_parent_key][0].value,
                        dtype=np.uint64)
                    id_working_set.extend(former_parent_ids)
                    id_history.extend(former_parent_ids)

        return np.array(id_history)

    def add_edge(self, atomic_edge, affinity=None, is_cg_id=False):
        """ Adds an atomic edge to the ChunkedGraph

        :param atomic_edge: list of two ints
        :param affinity: float
        :param is_cg_id: bool
        """
        time_stamp = datetime.datetime.now()
        time_stamp = UTC.localize(time_stamp)

        if affinity is None:
            affinity = 1

        rows = []

        if not is_cg_id:
            atomic_edge = [self.get_cg_id_from_rg_id(atomic_edge[0]),
                           self.get_cg_id_from_rg_id(atomic_edge[1])]

        # Walk up the hierarchy until a parent in the same chunk is found
        parent_ids = [self.get_parent(atomic_edge[0]),
                      self.get_parent(atomic_edge[1])]

        while not test_if_nodes_are_in_same_chunk(parent_ids):
            parent_ids = [self.get_parent(parent_ids[0]),
                          self.get_parent(parent_ids[1])]

        original_parents = [self.get_root(parent_ids[0], is_cg_id=True),
                            self.get_root(parent_ids[1], is_cg_id=True)]

        # Find a new node id and update all children
        circumnvented_nodes = parent_ids.copy()

        chunk_id = np.frombuffer(parent_ids[0], dtype=np.uint32)[1]
        new_parent_id = self.find_unique_node_id(chunk_id)
        new_parent_id_b = np.array(new_parent_id).tobytes()
        while parent_ids[0] is not None and parent_ids[1] is not None:
            combined_child_ids = np.array([], dtype=np.uint64)
            for prior_parent_id in parent_ids:
                r = self.table.read_row(serialize_node_id(prior_parent_id))
                child_ids = np.frombuffer(r.cells[self.family_id][serialize_key("children")][0].value,
                                          dtype=np.uint64)
                child_ids = child_ids[~np.in1d(child_ids, circumnvented_nodes)]
                combined_child_ids = np.concatenate([combined_child_ids, child_ids])

                for child_id in child_ids:
                    val_dict = {"parents": new_parent_id_b}
                    rows.append(mutate_row(self.table, serialize_node_id(child_id),
                                           self.family_id, val_dict, time_stamp))

            # Create new parent node
            val_dict = {"children": combined_child_ids.tobytes()}

            parent_ids = [self.get_parent(parent_ids[0]),
                          self.get_parent(parent_ids[1])]

            current_node_id = new_parent_id
            if parent_ids[0] is not None and parent_ids[1] is not None:
                chunk_id = np.frombuffer(parent_ids[0], dtype=np.uint32)[1]
                new_parent_id = self.find_unique_node_id(chunk_id)
                new_parent_id_b = np.array(new_parent_id).tobytes()

                val_dict["parents"] = new_parent_id_b
            else:
                val_dict["former_parents"] = np.array(original_parents).tobytes()

                rows.append(mutate_row(self.table,
                                       serialize_node_id(original_parents[0]),
                                       self.family_id,
                                       {"new_parents": new_parent_id_b}))

                rows.append(mutate_row(self.table,
                                       serialize_node_id(original_parents[1]),
                                       self.family_id,
                                       {"new_parents": new_parent_id_b}))

            rows.append(mutate_row(self.table,
                                   serialize_node_id(current_node_id),
                                   self.family_id, val_dict))

        # Atomic edge
        for i_atomic_id in range(2):
            val_dict = {"atomic_partners": np.array([atomic_edge[(i_atomic_id + 1) % 2]]).tobytes(),
                        "atomic_affinities": np.array([affinity]).tobytes()}
            rows.append(mutate_row(self.table, serialize_node_id(atomic_edge[i_atomic_id]),
                                   self.family_id, val_dict, time_stamp))

        status = self.table.mutate_rows(rows)

    def get_subgraph(self, agglomeration_id, bounding_box=None,
                     bb_is_coordinate=False, time_stamp=None):
        """ Returns all edges between supervoxels belonging to the specified
            agglomeration id within the defined bouning box

        :param agglomeration_id: int
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param time_stamp: datetime or None
        :return: edge list
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        if bb_is_coordinate:
            bounding_box = np.array(bounding_box, dtype=np.float32) / self.chunk_size
            bounding_box[0] = np.floor(bounding_box[0])
            bounding_box[1] = np.ceil(bounding_box[1])
            
        # bounding_box = np.array(bounding_box, dtype=np.int)

        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        affinities = np.array([], dtype=np.float32)
        child_ids = [agglomeration_id]

        while len(child_ids) > 0:
            new_childs = []
            layer = get_chunk_id_from_node_id(child_ids[0])[-1]

            for child_id in child_ids:
                if layer == 2:
                    this_edges, this_affinities = self.get_subgraph_chunk(child_id, time_stamp=time_stamp)

                    affinities = np.concatenate([affinities, this_affinities])
                    edges = np.concatenate([edges, this_edges])
                else:
                    this_children = self.read_row(child_id, "children",
                                                  dtype=np.uint64)

                    # cids_min = np.frombuffer(this_children, dtype=np.uint8).reshape(-1, 8)[:, 4:-1][:, ::-1] * self.fan_out ** np.max([0, (layer - 2)])
                    # cids_max = cids_min + self.fan_out * np.max([0, (layer - 2)])
                    #
                    # child_id_mask_min_upper = np.all(cids_min <= bounding_box[1], axis=1)
                    # child_id_mask_max_lower = np.all(cids_max > bounding_box[0], axis=1)
                    #
                    # m = np.logical_and(child_id_mask_min_upper, child_id_mask_max_lower)
                    # this_children = this_children[m]

                    new_childs.extend(this_children)

            child_ids = new_childs

        return edges, affinities

    def get_subgraph_chunk(self, parent_id, time_stamp=None):
        """ Takes an atomic id and returns the associated agglomeration ids

        :param parent_id: int
        :param time_stamp: None or datetime
        :return: edge list
        """
        if time_stamp is None:
            time_stamp = datetime.datetime.now()

        if time_stamp.tzinfo is None:
            time_stamp = UTC.localize(time_stamp)

        child_ids = self.read_row(parent_id, "children", dtype=np.uint64)
        edge_key = serialize_key("atomic_partners")
        affinity_key = serialize_key("atomic_affinities")

        edges = np.array([], dtype=np.uint64).reshape(0, 2)
        affinities = np.array([], dtype=np.float32)
        for child_id in child_ids:
            node_edges = np.array([], dtype=np.uint64)
            node_affinities = np.array([], dtype=np.float32)

            r = self.table.read_row(serialize_node_id(child_id))
            for i_edgelist in range(len(r.cells[self.family_id][edge_key])):
                if time_stamp > r.cells[self.family_id][edge_key][i_edgelist].timestamp:
                    edge_batch = np.frombuffer(r.cells[self.family_id][edge_key][i_edgelist].value, dtype=np.uint64)
                    affinity_batch = np.frombuffer(r.cells[self.family_id][affinity_key][i_edgelist].value, dtype=np.float32)
                    edge_batch_m = ~np.in1d(edge_batch, node_edges)

                    affinity_batch = affinity_batch[:len(edge_batch)] # TEMPORARY HACK

                    node_edges = np.concatenate([node_edges, edge_batch[edge_batch_m]])
                    node_affinities = np.concatenate([node_affinities,
                                                      affinity_batch[edge_batch_m]])

            node_edge_m = node_affinities > 0
            node_edges = node_edges[node_edge_m]
            node_affinities = node_affinities[node_edge_m]

            if len(node_edges) > 0:
                node_edges = np.concatenate([np.ones((len(node_edges), 1), dtype=np.uint64) * child_id,  node_edges[:, None]], axis=1)

                edges = np.concatenate([edges, node_edges])
                affinities = np.concatenate([affinities, node_affinities])

        return edges, affinities

    def remove_edge(self, atomic_edges, is_cg_id=False):
        pass
        # time_stamp = datetime.datetime.now()
        # time_stamp = UTC.localize(time_stamp)
        #
        # if not is_cg_id:
        #     for i_atomic_edge in range(len(atomic_edges)):
        #         atomic_edges[i_atomic_edge] = [self.get_cg_id_from_rg_id(atomic_edges[i_atomic_edge][0]),
        #                                        self.get_cg_id_from_rg_id(atomic_edges[i_atomic_edge][1])]
        #
        # atomic_edges = np.array(atomic_edges)
        #
        # # Remove atomic edge
        # rows = []
        # for atomic_edge in atomic_edges:
        #     for i_atomic_id in range(2):
        #         atomic_id = atomic_edge[i_atomic_id]
        #
        #         val_dict = {"atomic_partners": np.array([atomic_edge[(i_atomic_id + 1) % 2]]).tobytes(),
        #                     "atomic_affinities": np.array([-1]).tobytes()}
        #         rows.append(mutate_row(self.table, serialize_node_id(atomic_id),
        #                                self.family_id, val_dict, time_stamp))
        # # self.table.mutate_rows(rows)
        #
        # chunk_ids = np.frombuffer(atomic_edges, dtype=np.uint32)[1::2].reshape(-1, 2)
        # u_chunk_ids = np.unique(chunk_ids)
        #
        # # Connected component if removed edge is within an atomic chunk
        # for u_chunk_id in u_chunk_ids:
        #     node_id = atomic_edges[chunk_ids == u_chunk_id][0]
        #     parent_id = self.get_parent(node_id)
        #     edges, affinities = self.get_subgraph_chunk(parent_id)
        #
        #     g = nx.from_edgelist(edges)
        #     nx.connected_components(g)
        #
        # return atomic_edges, chunk_ids
