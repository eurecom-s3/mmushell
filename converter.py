#!/usr/bin/env python3

import yaml
import argparse
from sortedcontainers import SortedDict
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("MACHINE_CONFIG", help="YAML file describing the machine", type=argparse.FileType("r"))
    args = parser.parse_args()

    # Load machine config
    try:
        machine_config = yaml.load(args.MACHINE_CONFIG, Loader=yaml.FullLoader)
        args.MACHINE_CONFIG.close()
    except Exception as e:
        print(f"Malformed YAML file: {e}")
        exit(1)

    # Call the exporter
    exporter(machine_config)

def exporter(machine_config):
        """Convert dump set into an ELF file containg the physical address space"""
        
        architecture = machine_config["cpu"]["architecture"]
        bits = machine_config["cpu"]["bits"]
        endianness = machine_config["cpu"]["endianness"]
        prefix = machine_config["memspace"]["ram"][0]["dumpfile"].split(".")[0]

        with open(prefix + ".elf", "wb") as elf_fd:
            # Create the ELF header and write it on the file
            machine_data = {
                "QEMUArchitecture": architecture,
                "Uptime": -1,
                "CPURegisters": [],
                "MemoryMappedDevices": [(x["start"], x["end"] + 1) for x in machine_config["memspace"]["not_ram"]],
                "MMUMode": machine_config["mmu"]["mode"]
                }

            # Create ELF main header
            if architecture == "aarch64":
                e_machine = 0xB7
            elif architecture == "arm":
                e_machine = 0x28
            elif architecture == "riscv":
                e_machine = 0xF3
            elif architecture == "intel":
                if bits == 64:
                    e_machine = 0x3E
                else:
                    e_machine = 0x03
                machine_data["CPUSpecifics"] = {"MAXPHYADDR": machine_config["cpu"]["processor_features"]["m_phy"]}
            else:
                raise Exception("Unsupported architecture")

            e_ehsize = 0x40
            e_phentsize = 0x38
            elf_h = bytearray(e_ehsize)
            elf_h[0x00:0x04] = b'\x7fELF'                                   # Magic
            elf_h[0x04] = 2                                                 # Elf type
            elf_h[0x05] = 1 if endianness == "little" else 2                # Endianness
            elf_h[0x06] = 1                                                 # Version
            elf_h[0x10:0x12] = 0x4.to_bytes(2, endianness)                  # e_type
            elf_h[0x12:0x14] = e_machine.to_bytes(2, endianness)            # e_machine
            elf_h[0x14:0x18] = 0x1.to_bytes(4, endianness)                  # e_version
            elf_h[0x34:0x36] = e_ehsize.to_bytes(2, endianness)             # e_ehsize
            elf_h[0x36:0x38] = e_phentsize.to_bytes(2, endianness)          # e_phentsize
            elf_fd.write(elf_h)
            
            regions = SortedDict()
            for region in machine_config["memspace"]["ram"]:
                regions[(region["start"], region["end"] + 1)] = region["dumpfile"]
            for region in machine_config["memspace"]["not_ram"]:
                regions[(region["start"], region["end"] + 1)] = None
            
            # Write segments in the new file and fill the program header
            p_offset = len(elf_h)
            offset2p_offset = {}
            
            for (begin, end), dump_file in regions.items():
                # Not write not RAM regions
                if dump_file is None:
                    offset2p_offset[(begin, end)] = -1
                    continue
                
                # Write physical RAM regions
                offset2p_offset[(begin, end)] = p_offset
                with open(dump_file, "rb") as region_fd:
                    elf_fd.write(region_fd.read())
                p_offset += (end - begin)

            # Create FOSSIL NOTE segment style
            pad = 4
            name = "FOSSIL"
            n_type = 0xDEADC0DE
            name_b = name.encode()
            name_b += b"\x00"
            namesz = len(name_b).to_bytes(pad, endianness)
            name_b += bytes(pad - (len(name_b) % pad))

            descr_b = json.dumps(machine_data).encode()
            descr_b += b"\x00"
            descr_b += bytes(pad - (len(descr_b) % pad))
            descrsz = len(descr_b).to_bytes(pad, endianness)

            machine_note = namesz + descrsz + n_type.to_bytes(pad, endianness) + name_b + descr_b
            len_machine_note = len(machine_note)
            elf_fd.write(machine_note)

            # Create the program header
            # Add FOSSIL NOTE entry style
            p_header = bytes()
            note_entry = bytearray(e_phentsize)
            note_entry[0x00:0x04] = 0x4.to_bytes(4, endianness)         # p_type
            note_entry[0x08:0x10] = p_offset.to_bytes(8, endianness)    # p_offset
            note_entry[0x20:0x28] = len_machine_note.to_bytes(8, endianness)     # p_filesz

            p_offset += len_machine_note
            e_phoff = p_offset
            p_header += note_entry

            # Add all the segments (ignoring not in RAM pages)
            for (begin, end), offset in offset2p_offset.items():
                if offset == -1:
                    p_filesz = 0
                    pmask = 6
                    offset = 0
                else:
                    p_filesz = end - begin
                    pmask = 7

                segment_entry = bytearray(e_phentsize)
                segment_entry[0x00:0x04] = 0x1.to_bytes(4, endianness)          # p_type
                segment_entry[0x04:0x08] = pmask.to_bytes(4, endianness)        # p_flags
                segment_entry[0x10:0x18] = begin.to_bytes(8, endianness)        # p_vaddr
                segment_entry[0x18:0x20] = begin.to_bytes(8, endianness)        # p_paddr Original offset
                segment_entry[0x28:0x30] = (end - begin).to_bytes(8, endianness)     # p_memsz
                segment_entry[0x08:0x10] = offset.to_bytes(8, endianness)       # p_offset
                segment_entry[0x20:0x28] = p_filesz.to_bytes(8, endianness)     # p_filesz

                p_header += segment_entry

            # Write the segment header
            elf_fd.write(p_header)
            s_header_pos = elf_fd.tell() # Last position written (used if we need to write segment header)
            e_phnum = len(regions) + 1 

            # Modify the ELF header to point to program header
            elf_fd.seek(0x20)
            elf_fd.write(e_phoff.to_bytes(8, endianness))             # e_phoff

            # If we have more than 65535 segments we have create a special Section entry contains the
            # number of program entry (as specified in ELF64 specifications)
            if e_phnum < 65536:
                elf_fd.seek(0x38)
                elf_fd.write(e_phnum.to_bytes(2, endianness))         # e_phnum
            else:
                elf_fd.seek(0x28)
                elf_fd.write(s_header_pos.to_bytes(8, endianness))    # e_shoff
                elf_fd.seek(0x38)
                elf_fd.write(0xFFFF.to_bytes(2, endianness))          # e_phnum
                elf_fd.write(0x40.to_bytes(2, endianness))            # e_shentsize
                elf_fd.write(0x1.to_bytes(2, endianness))             # e_shnum

                section_entry = bytearray(0x40)
                section_entry[0x2C:0x30] = e_phnum.to_bytes(4, endianness)  # sh_info
                elf_fd.seek(s_header_pos)
                elf_fd.write(section_entry)

if __name__ == '__main__':
    main()
