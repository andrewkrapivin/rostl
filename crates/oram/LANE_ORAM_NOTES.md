# Lane ORAM Development Notes

These notes apply to the lane ORAM work on the `lane_orama` branch. They are
design notes for the partial implementation and are not general requirements
for every ORAM in the repository.

## Lane layout

- Treat each fixed slot across all buckets as an independent lane: slot 0 in
  every bucket is lane 0, slot 1 in every bucket is lane 1, and so on.
- For now, loaded paths remain bucket-major/interleaved: all `Z` slots from the
  root bucket, then all `Z` slots from the next bucket, and so on.
- We may later evaluate a lane-major path buffer containing every block from
  lane 0 contiguously, followed by lane 1, and so on. Do not change to that
  layout without making the choice explicit.
- Bucket width `Z` is a public compile-time configuration and may differ between
  use cases.
- Tree branching factor `B` is a public compile-time configuration restricted
  to powers of two. Path indexing may rely on bit masks instead of division or
  remainder.

## Current implementation direction

- Focus on `Block32` first while keeping the explicit `Block64` layout available.
- Blocks are exactly one 64-byte cache line and have 64-byte alignment.
- Default blocks are entirely `0xff`, so every SIMD lane initially contains its
  maximum value. For now, `u32::MAX` and `u64::MAX` are reserved as dummy keys
  for `Block32` and `Block64`, respectively. This ensures a dummy block cannot
  match a valid requested key using the current key-only comparison.
- Removing a block replaces the entire cache line with the canonical all-`0xff`
  dummy block in one masked move. Leaving the old key in a dummy block would
  allow a later key-only scan to match that stale slot.
- Reserving the maximum key prevents use of the full key space. If full-key-space
  support is needed, conditional reads must additionally require that the block
  position is not the corresponding dummy position.
- AVX-512 implementations are selected at compile time. Keep matching
  non-AVX-512 entry points as explicit `todo!()` stubs until those fallbacks are
  designed.
- Stash and loaded-path blocks are represented as one contiguous array. The
  stash offset is public metadata; scans that cover both regions do not need the
  offset.
- `LaneORAM<Z, S, B>` stores the rounded logical capacity, a `B`-ary wide tree,
  and `S + height * Z` blocks of combined stash/path storage.
