# label_propagation

Synchronous Label Propagation (DolphinDB programming challenge).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requires a C++20 compiler and OpenMP (e.g. g++ 13+). No third-party dependencies.

## Run

```bash
./build/label_propagation <input.csv>
```

Reads the input graph CSV, runs synchronous label propagation to
convergence, and writes `output.csv` (node_id,final_label, sorted by
node_id) to the current working directory, then exits.

## Implementation notes

- Whole-file `mmap`, line-aligned byte-range parallel parsing
- Numeric node ids resolved through a dedicated (value,length)-keyed
  open-addressing interner; other ids through a string interner
- Row-order CSR adjacency; label ranks assigned lexicographically so the
  tie-break is a min over integer ranks
- Active-frontier iteration: full rounds until changes drop below N/4,
  afterwards only nodes adjacent to changed nodes are recomputed
- Output order: already-sorted detection, LSD radix (numeric ids) or
  std::sort fallback; single-buffer write

Set `LP_TIMING=1` to print per-phase timings to stderr.
