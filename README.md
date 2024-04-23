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
$ direnv allow .
```

## Organisation

- `mmushell/architectures/` : various architectures parsers and a generic one
- `mmushell/mmushell.py` : main script allowing to reconstruct virtual address spaces from a memory dump, more instructions below
- `mmushell/exporter.py` : this is a POC showing the possible use of techniques to perform a preliminary analysis of a dump by exporting each virtual address space as a self-contained ELF Core dump file. See section [TOWARDS OS AGNOSTIC MEMORY FORENSICS](https://www.s3.eurecom.fr/docs/tops22_oliveri.pdf).
- `converter.py` : export dump to be used in [Fossil](https://github.com/eurecom-s3/fossil). It adds CPU registers and convert the kernel physical address space in virtual address space one. **Note**: you can ignore this script, is not part of mmushell
- `qemu/` : contains scripts and patch necessary to get ground truth registers values from an emulated system

## Usage

### Dataset

[Here](https://www.s3.eurecom.fr/datasets/datasets_old_www/mmushell_dataset.tar) part of the dataset containing the memory dumps of the OSs used in the paper (only the open-source ones, due to license restrictions).

In each archive there are a minimum of 4 files and require at least 4GB of free space (decompressed):
- `XXX.regs` : contains the values of the registers collected by QEMU during the execution (the ground truth), pickle format, to be used (optionally) with --gtruth option of mmushell
- `XXX.yaml` : contains the hardware configuration of the machine which has run the OS, YAML file, to be used as argument of mmushell
- `XXX.dump.Y` : chunk of the RAM dump of the machine
- `XXX.lzma` : an mmushell session file, it contains the output of mmushell, pickle lzma format, to be used (optionally) with --session option of mmushell

The use of `XXX.lzma` allows to avoid reexecuting the parsing and data structure reconstructing phase, gaining time!

### CLI

mmushell must be run in the folder containing dump/configuration files as all the paths are relatives.

> [!Warning]
> Some OSs require a minimum of 32GB of RAM to be parsed (the Intel 32bit ones, in particular HaikuOS) or a minimum of 1 hour of execution (independently by the number of the CPU cores, Intel 32/PAE/IA64 OSs)
> Consider using the session file for them to gain time.

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
        # Example: reserved regions for MMIO, ROM, ... See https://en.wikipedia.org/wiki/Memory-mapped_I/O_and_port-mapped_I/O#Examples
        # Those portions are needed because page tables also maps these special physical addresses, so the CPU can use these associated
        # virtual addresses to write or read from them. We need to distinguish them otherwise we can misinterpret some page tables as data pages.
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

### Ground truth

`XXXX_gtruth` commands are available only if you load a `XXX.regs` file as they compare found results with the ground truth.
These commands have an equivalent command which show only the results found by MMUShell without comparing them with the ground truth.

> [!Note]
> The folder `qemu/` contains the patch for QEMU 5.0.0 in order to collect the ground truth values of the MMU registers during OSs execution. Please read the concerned [README](qemu/README.md).

### Notes and procedures

As mmushell available commands differs from one architecture to another, here are different steps needed to be performed in order. **Note** : steps prefixed with "*" are necessary only if you don't use session files (e.g: `XXX.lzma`).

**RISC-V**

1. *`parse_memory` : find MMU tables
2. *`find_radix_trees` : reconstruct radix trees
3. `show_radix_trees_gtruth` : compare radix trees found with the ground truth

**MIPS**

1. *`parse_memory` -> find MMU opcodes
2. *`find_registers_values` -> perform dataflow analysis and retrieve registers values
3. `show_registers_gtruth` -> compare registers found with the ground truth

**PowerPC**

1. *`parse_memory` -> find MMU opcodes and hash tables
2. *`find_registers_values` -> perform dataflow analisys and retrieve registers values
3. `show_hashtables_gtruth` -> compare the hash table found with the ground truth
4. `show_registers_gtruth` -> compare the registers found with the ground truth

> [!Note]
> For Linux Debian, Mac OS X, Morphos : `show_hashtables_gtruth` shows another Hash Table not retrieved by MMUShell, but it is a table used during startup (as shown by the timestamp) and we ignore it because it does not used during normal OS operation.
> For Mac OS X : ignore BAT registers values in `show_registers_gtruth` as it uses different values for each process (as shown by the ground truth), the falses and positives results are purely a coincidence.

**Intel**

1. *`parse_memory` -> look for tables and IDT
2. *`find_radix_trees` -> reconstruct radix trees (could be slow)
3. *`show_idtrs_gtruth` -> show a comparization between true and found IDTs (note: some OSs define multiple IDTs, one per core). We deliberately ignore IDT table used during the boot phase (see PowerPC notes)
4. `show_radix_trees_gtruth XXXX` -> where XXX must be the PHYSICAL address of the last IDT used by the system shown by show_idtrs_gtruth". Shows a comparization between true and found radix trees which resolve the IDT PHYSICAL ADDRESS XXXX (obtained by the previous command, for our statistics we always use a true positive one)

> [!Note]
> For BarrellfishOS, OmniOS : those allocate a different IDT for every single CPU core. Some processes are core specific and are able to resolve only the IDT of the same core. For each IDT found, MMUShell shows different proccesses as FN. They are not real false negatives because are the per-core processes which are not able to resolve that specific IDT but are found by MMUShell using one among the other IDT.
> For RCore : if you enter the physical address of the real IDT used by the system in `show_radix_trees_gtruth`, MMUShell does not show any entry because it has found a different IDT (a FP) and has no valid radix trees for the real IDT. Please use `show_idtrs` to show the IDT found (YYY) and `show_radix_trees XXXX` to show the radix trees associated (all FP).

**ARM**

1. *`find_registers_values` -> perform dataflow analysis to recover TTBCR value
2. *`show_registers_gtruth` -> compare the values retrieved with the ground truth
3. *`set_ttbcr XXX` -> use REAL value of the TTBCR shown by the previous command
4. *`find_tables` -> find MMU tables
5. *`find_radix_trees` -> reconstruct radix trees
6. `show_radix_trees_gtruth` -> compare radix trees found with the ground truth

**AArch64**

1. *`find_registers_values` -> perform dataflow analysis to recover TCR value
2. *`show_registers_gtruth` -> compare the values retrieved with the ground truth
3. *`set_tcr XXX` -> use REAL value of the TCR shown by the previous command
4. *`find_tables` -> find MMU tables
5. *`find_radix_trees` -> reconstruct radix trees
6. `show_radix_trees_gtruth` -> compare radix trees found with the ground truth
