# pylint: disable=protected-access,missing-function-docstring,invalid-name,wrong-import-position

from os import environ
from datetime import timedelta

environ["BIGTABLE_PROJECT"] = "<>"
environ["BIGTABLE_INSTANCE"] = "<>"
environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path>"

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Concurrency
from pychunkedgraph.graph.operation import GraphEditOperation


def repair_operation(cg, log_d, operation_id):
    operation = GraphEditOperation.from_operation_id(
        cg, operation_id, multicut_as_split=False, privileged_mode=True
    )
    ts = log_d["timestamp"]
    result = operation.execute(
        operation_id=operation_id,
        parent_ts=ts - timedelta(seconds=0.1),
        override_ts=ts + timedelta(microseconds=(ts.microsecond % 1000) + 10),
    )
    old_roots = operation._update_root_ids()
    print("roots", old_roots, result.new_root_ids)
    print("result op ID", result.operation_id)
    print("result L2 IDs", result.new_lvl2_ids)

    for root_ in old_roots:
        cg.client.unlock_root(root_, result.operation_id)
        cg.client.unlock_indefinitely_locked_root(root_, result.operation_id)


if __name__ == "__main__":
    op_ids_to_retry = [...]
    locked_roots = [...]

    cg = ChunkedGraph(graph_id="<graph_id>")
    node_attrs = cg.client.read_nodes(node_ids=locked_roots)
    for node_id, attrs in node_attrs.items():
        if Concurrency.IndefiniteLock in attrs:
            locked_op = attrs[Concurrency.IndefiniteLock][0].value
            op_ids_to_retry.append(locked_op)
            print(f"{node_id} indefinitely locked by op {locked_op}")
    print(f"total to retry: {len(op_ids_to_retry)}")

    logs = cg.client.read_log_entries(op_ids_to_retry)
    for op_id, log in logs.items():
        print(f"repairing {op_id}")
        repair_operation(cg, log, op_id)
