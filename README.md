# mmushell

## Description

MMUShell is an OS-Agnostic memory morensics tool, a proof of concept for techniques developed by Andrea Oliveri and Davide Balzarotti in ["In the Land of MMUs: Multiarchitecture OS-Agnostic Virtual Memory Forensics"](https://doi.org/10.1145/3528102).

The first step required to perform any analysis of a physical memory image is the reconstruction of the virtual address spaces, which allows translating virtual addresses to their corresponding physical offsets. However, this phase is often overlooked, and the challenges related to it are rarely discussed in the literature. Practical tools solve the problem by using a set of custom heuristics tailored on a very small number of well-known operating systems (OSs) running on few architectures.

In the whitepaper, we look for the first time at all the different ways the virtual to physical translation can be operated in 10 different CPU architectures. In each case, we study the inviolable constraints imposed by the memory management unit that can be used to build signatures to recover the required data structures from memory without any knowledge about the running OS.

This tool allows to experiment with the extraction of virtual address spaces, showing the challenges of performing an OS-agnostic virtual to physical address translation in real-world scenarios.
It was tested on a large set of 26 different OSs, 6 architectures and a use case on a real hardware device.

## Quick installation

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

## Documentation

You can find the updated documentation [here](https://memoscopy.github.io/mmushell), where you will find tutorials, how-to guides, references and explanations on this project.
