# mmushell

## Description

MMUShell is an OS-Agnostic memory morensics tool, a proof of concept for techniques developed by Andrea Oliveri and Davide Balzarotti in ["In the Land of MMUs: Multiarchitecture OS-Agnostic Virtual Memory Forensics"](https://doi.org/10.1145/3528102).

The first step required to perform any analysis of a physical memory image is the reconstruction of the virtual address spaces, which allows translating virtual addresses to their corresponding physical offsets. However, this phase is often overlooked, and the challenges related to it are rarely discussed in the literature. Practical tools solve the problem by using a set of custom heuristics tailored on a very small number of well-known operating systems (OSs) running on few architectures.

In the whitepaper, we look for the first time at all the different ways the virtual to physical translation can be operated in 10 different CPU architectures. In each case, we study the inviolable constraints imposed by the memory management unit that can be used to build signatures to recover the required data structures from memory without any knowledge about the running OS.

This tool allows to experiment with the extraction of virtual address spaces, showing the challenges of performing an OS-agnostic virtual to physical address translation in real-world scenarios.
It was tested on a large set of 26 different OSs, 6 architectures and a use case on a real hardware device.

## Installation

On a standard Linux distribution :
```shell
$ python -m venv --system-site-packages --symlinks venv
$ venv/bin/pip install -r requirements.txt
```

On Nix/NixOS :
```shell
$ nix develop
# or with direnv
$ direnv allow .devenv
```

## Usage

Help :

```shell
$ mmushell.py
usage: mmushell.py [-h] [--gtruth GTRUTH] [--session SESSION] [--debug] MACHINE_CONFIG
mmushell.py: error: the following arguments are required: MACHINE_CONFIG
```

1. Dump all the RAM areas of the machine that you want to analyze in raw format, one file per physical memory area.
2. Create a YAML file describing the hardware configuration of the machine (see the examples available in the dataset)
    The format is the following :
    ```yaml
    cpu:
        # Architecture type
        architecture: (aarch64|arm|intel|mips|ppc|riscv)

        # Endianness level
        endianness: (big|little)

        # Bits used by architecture
        bits: (32|64)

    mmu:
        # MMU mode varying from architectures
        mode: (ia64|ia32|pae|sv32|sv39|sv48|ppc32|mips32|Short|Long) # any class that inherits from MMU
        #        ^^^^^^^^^     ^^^^^^^^^^    ^^    ^^^     ^^^^^^
        #          intel         riscv      ppc    mips      arm

    memspace:
        # Physical RAM space region
        ram:
          - start: 0x0000000080000000 # ram start address
            end:   0x000000017fffffff # ram end address
            dumpfile: linux.dump

        # Physical memory regions that are not RAM
        not_ram:
          - start: 0x0000000000000000
            end: 0x0000000000011fff

          - start: 0x0000000000100000
            end: 0x0000000000101023
        # ...
    ```

3. Launch mmushell with the configuration file. Example with the provided RISC-V SV39 memory dump :
    ```shell
    $ mmushell.py dataset/riscv/sv39/linux/linux.yaml
    MMUShell.   Type help or ? to list commands.

    [MMUShell riscv]# ?

    Documented commands (type help <topic>):
    ========================================
    exit              help     parse_memory  show_radix_trees
    find_radix_trees  ipython  save_data     show_table
    ```
    Use the interactive shell to find MMU registers, Radix-Trees, Hash tables etc. and explore them. The `help` command lists all the possible actions available for the selected CPU architecture.

> [!Note]
> The folder `qemu/` contains the patch for QEMU 5.0.0 in order to collect the ground truth values of the MMU registers during OSs execution. Please read the concerned [README](qemu/README.md).

### Dataset

[Here](https://www.s3.eurecom.fr/datasets/datasets_old_www/mmushell_dataset.tar) part of the dataset containing the memory dumps of the OSs used in the paper (only the open-source ones, due to license restrictions).

