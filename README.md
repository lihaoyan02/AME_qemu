# AME QEMU: AME Instruction Set Extension for QEMU

AME QEMU is an open-source emulation and validation environment built on QEMU,
a generic machine and userspace emulator and virtualizer. It adds support for
the Attached Matrix Extension (AME), based on Ztt v0.6, to QEMU's RISC-V Tiny
Code Generator (TCG) backend. Dynamic translation allows AME programs to run
on a host without AME hardware.

The project supports AME functional verification, compiler and runtime
development, and operator validation through system and Linux user emulation.
System emulation provides a complete RISC-V machine for bare-metal programs
and guest operating systems; user emulation runs RISC-V Linux programs on the
host. Community contributions to the AME implementation and examples are
welcome.

## Key Features

- **AME Ztt v0.6 instructions:** resource, datatype, and state management;
  memory access, register moves, and data conversion; elementwise arithmetic
  and math functions; bitwise operations, comparisons, and predication;
  permutations; matrix multiply-accumulate; reductions and prefixes.
- **Configurable matrix resources:** matrix size, unit width, and M/Acc
  register counts, selected through CPU options.
- **Large Language Model (LLM) inference operators:** FP32, INT8, and
  mixed-precision GEMM; RMSNorm, GELU, and Softmax; RoPE, MHA, and FFN.

## Quick Start

Run the following commands from the repository root on a Linux host. The
examples use RV64 bare-metal programs.

### Prerequisites

On Ubuntu/Debian, install the common build dependencies:

```sh
sudo apt-get update
sudo apt-get install -y \
  build-essential ninja-build pkg-config python3 python3-venv \
  libglib2.0-dev libpixman-1-dev zlib1g-dev libfdt-dev
```

The operator examples also require an AME-enabled LLVM toolchain and RISC-V
GNU support libraries, provided by the prebuilt toolchain below.

### Build QEMU

```sh
mkdir -p build
cd build
../configure \
  --target-list=riscv64-softmmu,riscv64-linux-user \
  --enable-fdt=system \
  --disable-docs
cd ..
ninja -C build qemu-system-riscv64 qemu-riscv64
```

This produces `build/qemu-system-riscv64` for system emulation and
`build/qemu-riscv64` for Linux user emulation. AME is enabled at runtime with
`-cpu max,xame=true`.

### Install the toolchain

Install the prebuilt AME LLVM toolchain into `toolchain/install`:

```sh
./scripts/build_toolchain.sh
```

Alternatively, build it from source with CMake and a RISC-V GNU toolchain
providing `libgcc.a`:

```sh
sudo apt-get install -y cmake gcc-riscv64-unknown-elf
./scripts/build_toolchain.sh --src
```

### Run examples and tests

Build and test FP32 GEMM:

```sh
make -C example gemm_fp32
```

Build and test all nine operators:

```sh
make -C example test
```

The examples enable AME automatically, compare QEMU results with reference
outputs, and audit the required AME instructions in the disassembly. The
combined report is written to `example/build/reports/EXAMPLE_REPORT.md`.

General QEMU documentation is introduced in [README.rst](README.rst).

## License

QEMU as a whole is released under the GNU General Public License, version 2.
See [LICENSE](LICENSE) for licensing details and [COPYING](COPYING) for the
license text. Individual files and bundled components retain their own
license notices.
