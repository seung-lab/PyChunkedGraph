## Manifest formats

### Legacy format
Legacy format has the standard format defined by neuroglancer - `{segment_id}:{lod}:{bounding_box}`.

### Sharded Graphene format
For large chunkedgraph datasets like `minnie65` sharding is necessary to reduce number of mesh files to control storage costs.

There can be two types of mesh fragments in a manifest: `initial` and `dynamic`.
* `initial`

   Initials IDs are segment IDs generated at the time of chunkedgraph creation. Meshes for these are generated in the form of [shards](https://github.com/seung-lab/cloud-volume/wiki/Graphene#meshing). The following formats are used for mesh fragments of initial IDs, depending on whether existence of these shards is verified when generating the manifest. `~` is used to denote sharded format.

   With verification, `~{layer}/{shard_file}:{offset}:{size}`, this is unique for a segment ID.

   eg: `~2/425884686-0.shard:165832:217`

   Without verification, `~{segment_id}:{layer}:{chunk_id}:{fname}:{minishard_number}`, segment ID included to ensure unique fragment ID.

   eg: `~2:173395595644370944:425884686-0.shard:1`

   If verification is needed, the manifest includes fragment ID for a semgent ID only if it's mesh fragment exists. Without verification, fragment ID is included in manifest and assumed to exist.

* `dynamic`

   For segment IDs generated by edit operations during proofreading, legacy format is used to name mesh fragments.


For all formats, `prepend_seg_ids=true` query parameter can be used to add `~{segment_id}:` as prefix to fragment ID in the manifest. This can be used in neuroglancer to map segment ID to mesh fragment in cache, this helps avoid redownloading unaffected fragments after an edit operation.