# CLAUDE.md

Guidance for working in this **tt-metal** checkout (Tenstorrent's low-level
TT-Metalium kernel framework and the TT-NN op library / Python bindings).

## Environment setup (from source)

This machine (Ubuntu 22.04, clang-20, Tenstorrent devices at `/dev/tenstorrent`)
is set up by running, from the repo root, in order:

```bash
git submodule update --init --recursive      # umd, tracy, llama
sudo ./install_dependencies.sh                # apt build deps (use --validate to check)
./build_metal.sh                              # clang-20 toolchain, Release build
./create_venv.sh                              # uv-based python_env + editable ttnn install
```

A reusable wrapper that runs all four steps lives at `~/setup.py`
(`python3 ~/setup.py --repo /path/to/tt-metal`; `--skip-*` flags to skip stages).

### Activating the environment for a session

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

Verify: `python3 -m ttnn.examples.usage.run_op_on_device`

## Build

- `./build_metal.sh` — default Release build. Output goes to `build_Release/`,
  with a `build` symlink pointing at the active build dir.
- Default toolchain: `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake`.
- Useful flags: `--build-tests` / `--build-ttnn-tests` / `--build-metal-tests`
  (test binaries), `--enable-ccache`, `--development` (RelWithDebInfo), `--debug`,
  `-e`/`--export-compile-commands`, `--build-programming-examples`, `--clean`.
- `./build_metal.sh --help` lists all options.
- After a C++ change, rebuild incrementally with `cmake --build build` (or just
  re-run `./build_metal.sh`). Compiled test binaries land in `build/test/...`.

## Running tests

Activate the venv and set `TT_METAL_HOME`/`PYTHONPATH` first.

- C++ (gtest), needs `--build-tests`:
  `./build/test/tt_metal/unit_tests_api --gtest_filter="..."`
- Python (pytest): `pytest tests/ttnn/... -vvv` or a specific file.
- Helper suites: `./tests/scripts/run_cpp_unit_tests.sh`,
  `./tests/scripts/run_python_api_unit_tests.sh`.
- See `CONTRIBUTING.md` for the full test matrix and model perf tests.

## Layout

- `tt_metal/` — Metalium core (host runtime, kernels, HAL, dispatch).
- `ttnn/` — TT-NN op library and Python bindings.
- `tt_stl/`, `tt-train/` — STL utilities and the C++ training framework.
- `models/` — reference / demo models.
- `tests/` — C++ and Python tests, sweep framework.
- `tt_metal/third_party/{umd,tracy}` — submodules (driver, profiler).

## Conventions

- C++ is formatted with clang-format (`.clang-format`) and linted via clang-tidy
  (`.clang-tidy`); `pre-commit` hooks are installed by `create_venv.sh`.
- Match the style of surrounding code. See `CONTRIBUTING.md` and
  `METALIUM_GUIDE.md` for deeper architectural guidance.
- Don't commit the `build_Release/`, `build/`, or `python_env/` directories.
