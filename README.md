# ww3-dia-gpu-v2: the whole W3SNL1, inside WAVEWATCH III, bit for bit

> [!WARNING]
> **This project is archived and no longer maintained.**
> Work on the GPU port continues in **ww3-gpu**, which is now closed source and private, so the
> links to `h0ffmann/ww3-gpu` below no longer resolve publicly. For news on it, see
> [@h0ffmann](https://github.com/h0ffmann).
>
> **Licence change.** This archived repository stays under the MIT License ([`LICENSE`](LICENSE)),
> exactly as published. That licence covers only the contents of this repository; it does not
> extend to ww3-gpu or to any closed-source successor, which are not licensed under it. Code that
> was already published elsewhere keeps the licence it was published under.

The discrete interaction approximation (DIA) of WAVEWATCH III® (WW3), `W3SNL1` with its table
builder `INSNL1`, translated to C++ with [Kokkos](https://kokkos.org) (one source for Serial,
OpenMP and CUDA), called from WW3 itself, and checked against the Fortran with WW3's own
regression tests. This repository documents that work and compares it with
[ww3-dia-gpu](https://github.com/h0ffmann/ww3-dia-gpu) (a fork of
[consigcody94/ww3-dia-gpu](https://github.com/consigcody94/ww3-dia-gpu), the code of
[NOAA-EMC/WW3#1583](https://github.com/NOAA-EMC/WW3/issues/1583)). The code lives in
[h0ffmann/ww3-gpu](https://github.com/h0ffmann/ww3-gpu), under
[`kokkos/`](https://github.com/h0ffmann/ww3-gpu/tree/main/kokkos); nothing is copied here. The
port report is [kokkos/port_report/W3SNL1_KOKKOS.md](https://github.com/h0ffmann/ww3-gpu/blob/main/kokkos/port_report/W3SNL1_KOKKOS.md).

Every statement below about WW3 cites `model/src/<file>.F90:<line>` at `WW3@6d1bcdfe`, the
commit of the h0ffmann/WW3 branch `hoff/kokkos-snl1` that the port is built against.

## WAVEWATCH III regression tests: v1 against v2

Every entry of WW3's `regtests/bin/matrix.base` whose switch has `NL1` (311 at `WW3@6d1bcdfe`),
run with the Fortran DIA and with the candidate DIA from one executable, compared byte for byte
by WW3's `test.comp`.

| | entries runnable | byte-identical to the Fortran | failed |
|---|---|---|---|
| v1 (ww3-dia-gpu) | 0 of 311: not callable from WW3 (no shim, no caller in `W3SRCE`), and it computes `SA1`/`SA2`, not the `S` and `D` WW3 integrates | — | — |
| v2, run in progress (2026-10-15 19:07 UTC, gfortran 16.2) | 157 of 311 finished so far | 157 of 157 | 0 |
| v2, earlier runs (2026-10-15, gfortran 15.3) | 125 (100 serial, 25 OpenMP) | 125 of 125 | 0 |

16 of the 157 link the port but never call it (GQM grids, `ww3_tc1`); they are counted, and
reported separately in the final record. This table is replaced by the final count of the
311-entry run when it ends.

## What v1 is

`ww3_dia_benchmark.cu` (259 lines) times two CUDA kernels that compute `SA1` and `SA2`, part of
section 3 of `W3SNL1`, on synthetic data: random energies, four index arrays of the form
`(isp ± NTH [+1]) mod n`, made-up weights, `CONS = 3.0e7`. Its 16.8x compares one GPU thread per
spectral bin against one GPU thread per grid point, both on an RTX 3060, and its "bit-identical"
compares those two kernels with each other. The issue's own comment calls the index arrays
simplified and the speed-up an upper bound, and lists integration into WW3 as not done.

## What v2 is

| `W3SNL1` at `WW3@6d1bcdfe` | v1 | v2 |
|---|---|---|
| 1. depth scaling, `CONS(KDMEAN)` (`w3snl1md.F90:338-342`) | a constant | per point |
| 2. `UE = A/CONX`, the high-frequency tail with `FACHFE`, padding (`:348-376`) | random `UE` | yes |
| `INSNL1`: 32 index arrays, `AWG`, `SWG`, `AF11` (`:483-786`) | 4 synthetic arrays | built from the grid |
| 3.a four-point interpolation of four members, 16 loads (`:382-391`) | two-point, 2 members reused | yes |
| 3.b `SA1`, `SA2` and the six diagonal terms `DA*`, over the extended spectrum (`:394-414`) | `SA1`, `SA2` | yes |
| 4. the source `S` and the diagonal `D`, gathered through `IC11..IC82` (`:416-440`) | no | yes |
| called by WW3's `W3SRCE` | no | yes, behind the `KOKKOS` switch |
| compared with | itself | the Fortran DIA, inside WW3 |

## How it was achieved

1. **Translate, don't improve.** Each line of the Fortran has its C++ counterpart in the same
   order, `float` for WW3's `REAL`, no algebraic rewrite. One team (a CUDA block) per grid point,
   one thread per spectral bin, the same mapping as v1; the Fortran's working arrays (`UE`, `SA*`,
   `DA*`, `CON`) live in team scratch, shared memory on a GPU, with barriers between sections,
   since section 4 reads what other bins wrote in section 3.
2. **Reproduce the executable, not the source read by eye.** WW3 is built at `-O3`, and gfortran
   expands `AF11 = (SIG*TPIINV)**11` (`:741`, `:750`) through its power tree, `x5*x6`, where
   `-O0` calls `__powisf2`, `x3*x8`: the two disagree in the last bit for 64% of inputs. The port
   takes the `-O3` chain; runtime exponents (`XFR**IFRP`) follow `__powisf2`. Found by reading
   WW3's optimised GIMPLE (`-fdump-tree-optimized`), line by line against the C++.
3. **No contraction.** `-ffp-contract=off` on the CPU and `--fmad=false` on CUDA: a fused
   multiply-add rounds once where the Fortran rounds twice. With GCC's default the OpenMP build
   drifted 1.1e-5 relative from the Fortran.
4. **Wired into WW3, not beside it.** A `bind(C)` shim ([`kokkos/src/fortran_iface/snl1_shim.cpp`](https://github.com/h0ffmann/ww3-gpu/blob/main/kokkos/src/fortran_iface/snl1_shim.cpp))
   and a small caller patch on the WW3 fork: `W3INIT` builds the tables of each grid (`IMOD`),
   `W3SRCE` calls the kernel for the grid `W3SETG` selected (`IGRID`), and `WW_KOKKOS_SNL1=0|1`
   chooses the Fortran or the Kokkos DIA at run time in one executable.
5. **What WW3 runs it in.** One context per grid, so `ww3_multi` works; inside `W3WAVE`'s OpenMP
   region the shim runs the same kernel on `Kokkos::Serial`, so nested parallelism needs no
   runtime setting; MPI ranks each hold their own.

## How it was validated

There are no unit tests of the port. The tests are WW3's: every entry of WW3's
`regtests/bin/matrix.base` whose switch has `NL1`, 311 at `WW3@6d1bcdfe` (100 serial, 50
OpenMP, 161 MPI, including PDLIB unstructured grids and `ww3_multi`), run twice from one
executable, with the Fortran DIA and with the Kokkos DIA, and compared by WW3's `test.comp`
(bytes) and a per-field NetCDF comparison. The verdict is bit for bit: a file that differs fails
the entry. Files that two runs of the Fortran alone already write differently (PDLIB restart
files under MPI) are measured and excused, and only those.

What is measured so far, on an i9-14900, CPU only:

- the 100 serial and 25 OpenMP `NL1` entries: every output file byte-identical, gfortran 15.3
  ([serial record](https://github.com/h0ffmann/ww3-gpu/blob/main/docs/data/hoff_matrix_2026-10-15.tsv),
  [OpenMP record](https://github.com/h0ffmann/ww3-gpu/blob/main/docs/data/hoff_matrix_omp_2026-10-15.tsv));
- `ww3_multi`, MPI and hybrid MPI/OpenMP entries: byte-identical in the pilot runs;
- inside the Ubuntu image below (gfortran 15.2): byte-identical on `ww3_ts1`;
- the full 311-entry run on gfortran 16.2 is in progress; its record will be added to
  ww3-gpu's `docs/data/` and linked here.

How the five failures before the `AF11` fix were traced is in
[docs/log/2026-10-14-snl1-bitwise-af11.md](https://github.com/h0ffmann/ww3-gpu/blob/main/docs/log/2026-10-14-snl1-bitwise-af11.md).

## Performance, and what it is not

Recorded 2026-09-15 on an i9-14900 and an RTX 4090, 1 000 points per call, NK=25, NTH=24
([`kokkos/PORT_STATUS.md`](https://github.com/h0ffmann/ww3-gpu/blob/main/kokkos/PORT_STATUS.md) in ww3-gpu):

| backend | through the shim, ms/call | kernel, ms/call |
|---|---|---|
| one CPU core | 25.88 | 24.96 |
| OpenMP, 32 threads | 5.95 | 5.02 |
| CUDA, RTX 4090 | 0.75 | 0.047 |

Per spectral bin, the CUDA kernel computes the full `S` and `D` at 1.3e10 bins/s; v1 computes
`SA1`/`SA2` at 8.9e9 bins/s on an RTX 3060. The two numbers are not comparable: different GPUs,
a fraction of the work per bin in v1, synthetic index arrays with regular memory access. Neither
is a speed-up of WW3. In WW3 today the shim is called once per sea point and copies data on
every call, so WW3 sees neither the kernel column nor v1's number. Keeping the spectrum on the
GPU and calling the kernel once for all points is the next phase, and its timings will be
published only with a passed regression run on the same commit.

## Run it

Ubuntu image, no Nix, CPU only (the image holds the compiled programs, not the sources; its
`docker/` and `just docker-*` recipes arrive in ww3-gpu with the pull request of the 311-entry run):

```console
git clone --recurse-submodules https://github.com/h0ffmann/ww3-gpu.git && cd ww3-gpu
just docker-build
just docker-run --hoff smoke     # Kokkos DIA against Fortran DIA, same executable
just docker-run --ww3 full
```

With the pinned Nix toolchain: `just smoke-hoff`, `just full-hoff`, `just hoff-matrix 8 all`
(see ww3-gpu's [`README.md`](https://github.com/h0ffmann/ww3-gpu#readme)).

## Licence and credits

This repository: MIT ([`LICENSE`](LICENSE)); see the archive notice at the top for ww3-gpu. The kernels in ww3-gpu are derived works of WW3 and carry
`LGPL-3.0-or-later` in the versions published under it. v1 is consigcody94's, BSD-3-Clause. WAVEWATCH III® is a registered trademark
of NOAA's National Weather Service, used here only to refer to that software; this work is not
affiliated with, sponsored by or endorsed by NOAA.
