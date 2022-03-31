# mmushell
MMUShell OS-Agnostic Memory Forensics Tool

Proof of concept for techniques developed by Andrea Oliveri and Davide Balzarotti in 

["In the Land of MMUs: Multiarchitecture OS-Agnostic Virtual Memory Forensics"](https://doi.org/10.1145/3528102)

Installation:
```
pip install -r requirements.txt
```

Usage:
- Dump all the RAM areas of the machine that you want to analyze in raw format, one file per physical memory area.
- Create a YAML file describing the hardware configuration of the machine (see the examples available in the dataset).
- ```mmushell machine.yaml```
- Use the interactive shell to find MMU registers, Radix-Trees, Hash tables etc. and explore them. The ```help``` command lists all the possible actions available for the selected CPU architecture.
- [Here](http://crazyivan.s3.eurecom.fr:8888/mmushell_dataset.tar) part of the dataset containing the memory dumps of the OSs used in the paper (only the open-source ones, due to license restrictions).
- ```/qemu/``` contains the patch for QEMU 5.0.0 in order to collect the ground truth values of the MMU registers during OSs execution.
