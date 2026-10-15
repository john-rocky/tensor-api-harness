# models

One directory per model that went through the skill table.

- `skill_table.md` — the filled copy of the root table: status and evidence per row.
- `OPS.md` — the architecture and op inventory (row 1): blocks, ops, the Tensor API op for each, the rewrites.
- `ENVIRONMENT.md` — the environment record (row 15): LiteRT commit, toolchain, weights sha256, flags.
- `iterations/ITERATION_<N>.md` — one record per iteration: the instruction it ran on, the progress lines, the five close sections.
- `graph/` — the C++ graph, weights loader and host loop, laid out as a LiteRT `tensor/examples/<name>` directory (`tools/overlay.sh` links it in).
- `reference/` — the script that dumps the PyTorch reference and the manifest it writes. The dumps themselves are not committed.
- `results/` — the checks' outputs, the timing JSON, the delegation log. Small text files only.
