# timestral (Zig runner)

Standalone CPU inference for Timestral checkpoints, in Zig 0.17. Everything except the matrix
products is Zig standard library: files and memory-mapped weights (`std.Io`), the Hub download
(`std.http.Client`, `std.json`), threads (`std.Io.Threaded`) and SIMD (`@Vector`). Matrix products can
run on a C BLAS (Apple Accelerate by default on macOS, OpenBLAS optionally) bound through
[translate-c](https://codeberg.org/ziglang/translate-c) (pinned to its `zig-0.17.x` branch); the
builtin SIMD kernels need no C library at all. Elementwise work (softmax, SiLU, RMSNorm) uses `@Vector`, and
the layers run on a `std.Io.Threaded` pool: linear layers split by output columns, attention by
(row, head), and the forward and mirrored passes of symmetric averaging run concurrently. Output is
identical for any thread count.

```bash
zig build --release=fast
./zig-out/bin/timestral mistralai/Timestral-28M series.csv --horizon 96 > forecast.csv
```

`<model>` is a directory holding `config.json` and `model.safetensors`, or a Hub repo id. A repo id
is looked up in the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME`, `~/.cache/huggingface/hub`) and
downloaded there if missing (`HF_TOKEN` is used for gated repos; `--offline` or
`HF_HUB_OFFLINE=1` forbids the network). F32, F16 and BF16 checkpoints load.

## Input

`series.csv` has one line per variate; values are comma or whitespace separated and `nan` / `NA` / an
empty field marks a missing value. **Several series** go in one file, separated by a blank line (all
with the same number of variates and steps). Covariates use the same blocks, one block per series and
one line per covariate:

| flag | rows | steps per line |
|---|---|---|
| `--past FILE` | past-only covariates | as the series |
| `--known FILE` | known-future covariates | series + horizon |
| `--cat-past FILE` | categorical codes (numbers) | as the series |
| `--cat-known FILE` | categorical codes (numbers) | series + horizon |

Categorical codes are target-encoded per series for a univariate target and used raw otherwise, as in
the Python package. Output is CSV: `series,variate,step,q<level>...`.

Other options: `--threads N` (default: one per logical CPU; `1` is single-threaded), `--max-context N`, `--block-patches N`, `--no-symmetric`, `--no-clip`, `--revision REV`,
`--time` (load and forecast wall time on stderr).

## Build options

| option | values | default |
|---|---|---|
| `-Dblas=` | `accelerate`, `openblas`, `builtin` | `accelerate` on macOS, `builtin` elsewhere |

`builtin` is about 2x slower than Accelerate on the 28M model (17 vs 9 ms); use `-Dblas=openblas` on Linux for speed. Cross-compiling works with the builtin kernels (Linux x86_64/aarch64, musl, Windows, macOS):
`zig build --release=fast -Dblas=builtin -Dtarget=x86_64-linux-gnu`.

## Checked against PyTorch

`tests/reference.py` takes the same arguments and file formats and runs the Python package;
`tests/compare.py` diffs two result files. On the 28M model, with `uv sync --frozen` for PyTorch,
the Zig output matched to about 1e-6 (relative to max(|value|, 1)) for: a single series, several
series, multivariate targets, past, known, numeric and categorical covariates in every combination,
missing values, multi-block horizons, a context cap, fp16 and bf16 checkpoints, and the builtin kernels.
All four model sizes (5M to 209M) matched on a plain forecast.

Not verified: running on Linux or Windows (only cross-compiled), and the OpenBLAS build.

## License

Apache License 2.0, see [LICENSE](LICENSE). This is a port of the inference code in
[mistralai/timestral](https://github.com/mistralai/timestral), which is Apache 2.0. The Timestral model
weights are not covered by it: they are distributed on the Hugging Face Hub under a Modified MIT
license, available in each model repository.
