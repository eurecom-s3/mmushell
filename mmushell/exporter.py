#!/usr/bin/env python3

import json
import argparse
import traceback
import numpy as np

from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from compress_pickle import load as load_c
from collections import defaultdict
from bisect import bisect
from pickle import load
from struct import iter_unpack
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("PHY_ELF", help="Dump file in ELF format", type=str)
    parser.add_argument(
        "MMU_DATA",
        help="List of DTBs and MMU configuration registers",
        type=argparse.FileType("rb"),
    )
    args = parser.parse_args()

    # Load session file
    try:
        mmu_data = load(args.MMU_DATA)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Load ELF file
    elf_dump = ELFDump(args.PHY_ELF)

    # Dump processes
    for idx, process_mmu_data in enumerate(tqdm(mmu_data)):
        try:
            virtspace = get_virtspace(elf_dump, process_mmu_data)
            virtspace.export_virtual_memory_elf(f"process.{idx}.elf")
        except Exception as e:
            print(f"Error during process exporting: {e}")
            # print(traceback.format_exc())


class IMSimple:
    """Fast search in intervals (begin), (end).
    
        Description:
        - Represents a class for efficient search in intervals.
    
        Attributes:
        - keys: List of interval beginnings.
        - values: List of interval ends.
    
        Methods:
        - __init__(self, keys, values): Constructor method to initialize IMSimple instance.
        - __getitem__(self, x): Method to get the difference between x and the nearest interval beginning.
        - contains(self, x, size): Method to check if a given address and size fit within intervals.
        - get_values(self): Method to get values of intervals.
        - get_extremes(self): Method to get the first and last interval boundaries.
    
        Purpose:
        - To provide fast search functionality in intervals.
    """

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def __getitem__(self, x):
        idx = bisect(self.keys, x) - 1
        begin = self.keys[idx]
        if begin <= x < self.values[idx]:
            return x - begin
        else:
            return -1

    def contains(self, x, size):
        idx = bisect(self.keys, x) - 1
        begin = self.keys[idx]
        end = self.values[idx]
        if not (begin <= x < end) or x + size >= end:
            return -1
        else:
            return x - begin

    def get_values(self):
        return zip(self.keys, self.values)

    def get_extremes(self):
        return self.keys[0], self.values[-1]


class IMData:
    """Fast search in intervals (begin), (end, associated data).
    
        Description:
        - Represents a class for efficient search in intervals with associated data.
        
        Attributes:
        - keys: List of interval beginnings.
        - values: List of tuples containing interval ends and associated data.
        
        Methods:
        - __init__(self, keys, values): Constructor method to initialize IMSimple instance.
        - __getitem__(self, x): Method to get the associated data for a specific point.
        - contains(self, x, size): Method to check if a given address and size fit within intervals.
        - get_values(self): Method to get values of intervals.
        - get_extremes(self): Method to get the first and last interval boundaries.
        
        Purpose:
        - To provide fast search functionality in intervals with associated data
    """

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def __getitem__(self, x):
        """Return the index of the interval containing the value x.

        Args:
        - x: The value to check if it falls within any interval.
        - size: The size of the interval to find.

        Returns:
        - Index of the interval containing x if x falls within the interval and the size of the interval is not exceeded by adding the specified size; otherwise, returns -1.

        Description:
        - The method checks if the value x falls within any interval in the mapping.
        - If x is within an interval and adding the specified size does not exceed the interval size, it returns the index of the interval containing x.
        - Otherwise, it returns -1.

        Purpose:
        - To determine if a value falls within any interval in the mapping and if adding a specified size to the value exceeds the interval size.

        Technical Explanation:
        - The method utilizes binary search to find the index of the interval containing the value x.
        - It checks if x falls within the interval and if adding the specified size does not exceed the interval size.
        - If the conditions are met, it returns the index of the interval; otherwise, it returns -1.
    """
        idx = bisect(self.keys, x) - 1
        begin = self.keys[idx]
        end, data = self.values[idx]
        if begin <= x < end:
            return data
        else:
            return -1

    def contains(self, x, size):
        idx = bisect(self.keys, x) - 1
        begin = self.keys[idx]
        end, data = self.values[idx]
        if not (begin <= x < end) or x + size >= end:
            return -1
        else:
            return data

    def get_values(self):
        return zip(self.keys, self.values)

    def get_extremes(self):
        return self.keys[0], self.values[-1][0]


class IMOffsets:
    """Fast search in intervals (begin), (end, associated offset).
    
        Description:
        - Represents a class for efficient search in intervals.
        - Initializes with keys and values representing interval boundaries and associated offsets.
    
        Attributes:
        - keys: List of interval beginnings.
        - values: List of tuples containing interval ends and associated offsets.
    
        Methods:
        - __init__(self, keys, values): Constructor method to initialize IMOffsets instance.
        - __getitem__(self, x): Method to get the offset associated with a specific point.
        - contains(self, x, size): Method to check if a given address and size fit within intervals.
        - get_values(self): Method to get values of intervals.
        - get_extremes(self): Method to get the first and last interval boundaries.
    
        Purpose:
        - To provide fast search functionality in intervals with associated offsets.
    """

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def __getitem__(self, x):
        idx = bisect(self.keys, x) - 1
        begin = self.keys[idx]
        end, data = self.values[idx]
        if begin <= x < end:
            return x - begin + data
        else:
            return -1

    def contains(self, x, size):
        """Return the maximum size and the list of intervals.

            Args:
            - x: The value to check if it falls within any interval.
            - size: The size of the interval to find.
    
            Returns:
            - Maximum size available within intervals and the list of intervals.
    
            Description:
            - The method checks if the value x falls within any interval in the mapping.
            - If x is within an interval, it calculates the maximum size available starting from x and returns the list of intervals that cover the requested size.
    
            Purpose:
            - To find intervals containing a specific value and to determine the maximum size available within those intervals.
    
            Technical Explanation:
            - The method utilizes binary search to find the index of the interval containing the value x.
            - It then iterates through the intervals starting from the identified index and calculates the maximum size available.
            - The method returns the maximum size and the list of intervals that cover the requested size.
    """
        idx = bisect(self.keys, x) - 1
        begin = self.keys[idx]
        end, data = self.values[idx]
        if not (begin <= x < end):
            return 0, []

        intervals = [(x, min(end - x, size), x - begin + data)]
        if end - x >= size:
            return size, intervals

        # The address space requested is bigger than a single interval
        start = end
        remaining = size - (end - x)
        idx += 1
        print(start, remaining, idx)
        while idx < len(self.values):
            begin = self.keys[idx]
            end, data = self.values[idx]

            # Virtual addresses must be contigous
            if begin != start:
                return size - remaining, intervals

            interval_size = min(end - begin, remaining)
            intervals.append((start, interval_size, data))
            remaining -= interval_size
            if not remaining:
                return size, intervals
            start += interval_size
            idx += 1

    def get_values(self):
        return zip(self.keys, self.values)

    def get_extremes(self):
        return self.keys[0], self.values[-1][0]


class IMOverlapping:
    """Fast search in overlapping intervals (begin), (end, [associated offsets]).
    
        Description:
        - Represents a class for efficiently searching overlapping intervals.
        - Initializes with a list of intervals.
    
        Attributes:
        - intervals: List of intervals (begin), (end, [associated offsets]).
        - limits: Sorted list of interval limits.
        - results: List of results corresponding to intervals.
        
        Methods:
        - __init__(self, intervals): Constructor method to initialize IMOverlapping instance.
        - __getitem__(self, x): Method to get values associated with a specific point.
        - get_values(self): Method to get values of intervals.
    
        Purpose:
        - To provide fast search functionality in overlapping intervals.)
    """

    def __init__(self, intervals):
        """Initialize the class instance with a list of intervals.
            Args:
            - intervals: A list of intervals represented as tuples (l, r, v), where:
              - l: The left endpoint of the interval.
              - r: The right endpoint of the interval.
              - v: The value associated with the interval.
    
            Description:
            - The constructor initializes the class instance with a list of intervals.
            - It organizes the intervals based on their left endpoints and computes the changes in offsets.
            - The results are stored for efficient access during interval queries.
    
            Purpose:
            - To initialize the class instance with a list of intervals and precompute results for interval queries.
    
            Technical Explanation:
            - The constructor takes a list of intervals and sorts them based on their left endpoints.
            - It computes the changes in offsets for each interval and stores the results for efficient access during interval queries.
    
            Example:
            ```python
            intervals = [(0, 3, [10]), (1, 4, [20]), (2, 5, [30])]
            instance = ClassName(intervals) ```
        """
        limit2changes = defaultdict(lambda: ([], []))
        for idx, (l, r, v) in enumerate(intervals):
            assert l < r
            limit2changes[l][0].append(v)
            limit2changes[r][1].append(v)
        self.limits, changes = zip(*sorted(limit2changes.items()))

        self.results = [[]]
        s = set()
        offsets = {}
        res = []
        for idx, (arrivals, departures) in enumerate(changes):
            s.difference_update(departures)
            for i in departures:
                offsets.pop(i)

            for i in s:
                offsets[i] += self.limits[idx] - self.limits[idx - 1]

            s.update(arrivals)
            for i in arrivals:
                offsets[i] = 0

            res.clear()
            for k, v in offsets.items():
                res.extend([i + v for i in k])
            self.results.append(res.copy())

    def __getitem__(self, x):
        idx = bisect(self.limits, x)
        k = x - self.limits[idx - 1]
        return [k + p for p in self.results[idx]]

    def get_values(self):
        return zip(self.limits, self.results)


class ELFDump:
    """Represents a class for parsing ELF files and extracting machine data.
    
        Description:
        - ELFDump is a class designed to read and parse ELF files, extracting necessary information such as machine data, endianness, architecture, and memory mapped devices.
    
        Attributes:
        - filename: Name of the ELF file.
        - machine_data: Dictionary containing machine configuration extracted from ELF segments.
        - p2o: Mapping of physical addresses to file offsets.
        - o2p: Mapping of file offsets to physical addresses.
        - p2mmd: Mapping of physical addresses to Memory Mapped Devices (MMD) intervals.
        - elf_buf: Buffer containing the contents of the ELF file.
        - elf_filename: Name of the ELF file.

        Methods:
        - __init__(self, elf_filename): Constructor method to initialize ELFDump instance.
        - __read_elf_file(self, elf_fd): Reads and parses the ELF file from the provided file descriptor.
        - _compact_intervals_simple(self, intervals): Compacts intervals for contiguous pointer values.
        - _compact_intervals(self, intervals): Compacts intervals for contiguous pointer and pointed values.
        - in_ram(self, paddr, size=1): Returns True if the interval is completely in RAM.
        - in_mmd(self, paddr, size=1): Returns True if the interval is completely in Memory mapped devices space.
        - get_data(self, paddr, size): Returns the data at physical address (interval).
        - get_data_raw(self, offset, size=1): Returns the data at the offset in the ELF (interval).
        - get_machine_data(self): Returns a dict containing machine configuration.
        - get_ram_regions(self): Returns all the RAM regions of the machine and the associated offset.
        - get_mmd_regions(self): Returns all the Memory mapped devices intervals of the machine and the associated offset.

        Purpose:
        - To parse ELF files, extract machine data, and provide methods for accessing data and information from the ELF file.


    """
    def __init__(self, elf_filename):
        self.filename = elf_filename
        self.machine_data = {}
        self.p2o = None  # Physical to RAM (ELF offset)
        self.o2p = None  # RAM (ELF offset) to Physical
        self.p2mmd = None  # Physical to Memory Mapped Devices (ELF offset)
        self.elf_buf = np.zeros(0, dtype=np.byte)
        self.elf_filename = elf_filename

        with open(self.elf_filename, "rb") as elf_fd:
            # Load the ELF in memory
            self.elf_buf = np.fromfile(elf_fd, dtype=np.byte)
            elf_fd.seek(0)

            # Parse the ELF file
            self.__read_elf_file(elf_fd)

    def __read_elf_file(self, elf_fd):
        """Functioning:
            - Reads and parses the dump in ELF format from the provided file descriptor.
            - Extracts machine data, endianness, and architecture information from ELF segments.
            - Fills arrays for translating physical addresses to file offsets.
            - Compacts intervals for efficient processing.
            
            Args:
                self: The instance of the calling class.
                elf_fd: File descriptor for the ELF file to be read and parsed.
            
            Expected Results:
                Extract machine data, endianness, and architecture information from ELF segments.
                Fill arrays for translating physical addresses to file offsets.
                Compact intervals for efficient processing.
            
            Purpose:
                To read and parse an ELF file, extracting necessary information and preparing data structures for further processing.
            
            Technical Explanation:
            - The function begins by initializing lists to store information related to physical addresses, offsets, and memory-mapped devices.
            - It then utilizes the `ELFFile` class from the `elftools` module to parse the ELF file.
            - For each segment in the ELF file, it checks if it is a NoteSegment. If it is, it iterates through notes and extracts machine data, endianness, and architecture information.
            - If the segment is not a NoteSegment, it calculates the start and end addresses of the segment and fills arrays needed to translate physical addresses to file offsets.
            - These arrays are then compacted for efficiency using the `_compact_intervals` method.
            - Finally, the compacted intervals are assigned to attributes of the class instance for future reference and usage.
            """
        o2p_list = []
        p2o_list = []
        p2mmd_list = []
        elf_file = ELFFile(elf_fd)

        for segm in elf_file.iter_segments():
            # NOTES
            if isinstance(segm, NoteSegment):
                for note in segm.iter_notes():
                    # Ignore NOTE genrated by other softwares
                    if note["n_name"] != "FOSSIL":
                        continue

                    # At moment only one type of note
                    if note["n_type"] != 0xDEADC0DE:
                        continue

                    # Suppose only one deadcode note
                    self.machine_data = json.loads(note["n_desc"].rstrip("\x00"))
                    self.machine_data["Endianness"] = (
                        "little"
                        if elf_file.header["e_ident"].EI_DATA == "ELFDATA2LSB"
                        else "big"
                    )
                    self.machine_data["Architecture"] = "_".join(
                        elf_file.header["e_machine"].split("_")[1:]
                    )
            else:
                # Fill arrays needed to translate physical addresses to file offsets
                r_start = segm["p_vaddr"]
                r_end = r_start + segm["p_memsz"]

                if segm["p_filesz"]:
                    p_offset = segm["p_offset"]
                    p2o_list.append((r_start, (r_end, p_offset)))
                    o2p_list.append((p_offset, (p_offset + (r_end - r_start), r_start)))
                else:
                    # device_name = "" # UNUSED
                    for device in self.machine_data[
                        "MemoryMappedDevices"
                    ]:  # Possible because NOTES always the first segment
                        if device[0] == r_start:
                            # device_name = device[1] # UNUSED
                            break
                    p2mmd_list.append((r_start, r_end))

        # Debug
        # self.p2o_list = p2o_list
        # self.o2p_list = o2p_list
        # self.p2mmd_list = p2mmd_list

        # Compact intervals
        p2o_list = self._compact_intervals(p2o_list)
        o2p_list = self._compact_intervals(o2p_list)
        p2mmd_list = self._compact_intervals_simple(p2mmd_list)

        self.p2o = IMOffsets(*list(zip(*sorted(p2o_list))))
        self.o2p = IMOffsets(*list(zip(*sorted(o2p_list))))
        self.p2mmd = IMSimple(*list(zip(*sorted(p2mmd_list))))

    def _compact_intervals_simple(self, intervals):
        """Functioning:
            - Compacts intervals if pointer values are contiguous.
            - Merges adjacent intervals into a single interval.
            
            Args:
                self: The instance of the calling class.
                intervals: List of intervals to be compacted.
            
            Expected Results:
                Compact intervals if pointer values are contiguous.
            
            Purpose:
                To merge adjacent intervals into a single interval for efficient processing.
            
            Technical Explanation:
            - The function takes a list of intervals as input, where each interval consists of a begin and end value.
            - It initializes an empty list to store the compacted intervals.
            - It then iterates through each interval in the input list.
            - If the end value of the current interval is equal to the begin value of the previous interval, it extends the previous interval to include the end value of the current interval.
            - If the end value of the current interval is not equal to the begin value of the previous interval, it adds the previous interval to the list of compacted intervals and starts a new interval.
            - Finally, it appends the last interval to the list of compacted intervals.
            - The function returns the list of compacted intervals.
            
            Example:
                Input: [(0, 1), (1, 3), (4, 6), (6, 8)]
                Output: [(0, 3), (4, 8)]
        """
        fused_intervals = []
        prev_begin = prev_end = -1
        for interval in intervals:
            begin, end = interval
            if prev_end == begin:
                prev_end = end
            else:
                fused_intervals.append((prev_begin, prev_end))
                prev_begin = begin
                prev_end = end

        if prev_begin != begin:
            fused_intervals.append((prev_begin, prev_end))
        else:
            fused_intervals.append((begin, end))

        return fused_intervals[1:]

    def _compact_intervals(self, intervals):
        """Functioning:
            - Compacts intervals if pointer and pointed values are contiguous.
            - Merges adjacent intervals into a single interval.
            
            Args:
                self: The instance of the calling class.
                intervals: List of intervals to be compacted, where each interval consists of a start address and a tuple containing the end address and physical address.
            
            Expected Results:
                Compact intervals if pointer and pointed values are contiguous.
            
            Purpose:
                To merge adjacent intervals into a single interval for efficient processing.
            
            Technical Explanation:
            - The function takes a list of intervals as input, where each interval consists of a start address and a tuple containing the end address and physical address.
            - It initializes an empty list to store the compacted intervals.
            - It then iterates through each interval in the input list.
            - If the end address of the current interval is equal to the start address of the previous interval, and the physical address of the current interval is contiguous with the previous physical address, it extends the previous interval to include the end address and physical address of the current interval.
            - If the end address of the current interval is not equal to the start address of the previous interval or the physical address is not contiguous, it adds the previous interval to the list of compacted intervals and starts a new interval.
            - Finally, it appends the last interval to the list of compacted intervals.
            - The function returns the list of compacted intervals.
            
            Example:
                Input: [(0, (1, 100)), (1, (3, 101)), (4, (6, 102)), (6, (8, 103))]
                Output: [(0, (3, 100)), (4, (8, 102))]
        """
        fused_intervals = []
        prev_begin = prev_end = prev_phy = -1
        for interval in intervals:
            begin, (end, phy) = interval
            if prev_end == begin and prev_phy + (prev_end - prev_begin) == phy:
                prev_end = end
            else:
                fused_intervals.append((prev_begin, (prev_end, prev_phy)))
                prev_begin = begin
                prev_end = end
                prev_phy = phy

        if prev_begin != begin:
            fused_intervals.append((prev_begin, (prev_end, prev_phy)))
        else:
            fused_intervals.append((begin, (end, phy)))

        return fused_intervals[1:]

    def in_ram(self, paddr, size=1):
        """Return True if the interval is completely in RAM"""
        return self.p2o.contains(paddr, size)[0] == size

    def in_mmd(self, paddr, size=1):
        """Return True if the interval is completely in Memory mapped devices space"""
        return True if self.p2mmd.contains(paddr, size) != -1 else False

    def get_data(self, paddr, size):
        """Return the data at physical address (interval)"""
        size_available, intervals = self.p2o.contains(paddr, size)
        if size_available != size:
            return bytes()

        ret = bytearray()
        for interval in intervals:
            _, interval_size, offset = interval
            ret.extend(self.elf_buf[offset : offset + interval_size].tobytes())

        return ret

    def get_data_raw(self, offset, size=1):
        """Return the data at the offset in the ELF (interval)"""
        return self.elf_buf[offset : offset + size].tobytes()

    def get_machine_data(self):
        """Return a dict containing machine configuration"""
        return self.machine_data

    def get_ram_regions(self):
        """Return all the RAM regions of the machine and the associated offset"""
        return self.p2o.get_values()

    def get_mmd_regions(self):
        """Return all the Memory mapped devices intervals of the machine and the associated offset"""
        return self.p2mmd.get_values()


def get_virtspace(phy, mmu_values):
    """Get Virtual Address Space from a Physical Address Space.
        Functioning:
        - Returns a virtual address space based on the physical address space and MMU values.
        - Determines the architecture of the physical address space and selects the appropriate translator to create the virtual address space.
        
        Args:
            phy: Physical address space object.
            mmu_values: MMU (Memory Management Unit) values associated with the physical address space.
        
        Returns:
            A virtual address space object.
        
        Raises:
            Exception: If the architecture of the physical address space is unknown.
        
        Purpose:
            To create a virtual address space from a physical address space based on the architecture of the system.
"""
    architecture = phy.get_machine_data()["Architecture"].lower()
    if "riscv" in architecture:
        return RISCVTranslator.factory(phy, mmu_values)
    elif "x86" in architecture or "386" in architecture:
        return IntelTranslator.factory(phy, mmu_values)
    else:
        raise Exception("Unknown architecture")


class AddressTranslator:
   """ Represents a base class for address translation.
        Description:
        - AddressTranslator is a base class providing functionalities for translating addresses.
    
        Attributes:
        - dtb: Device tree blob.
        - phy: Physical memory instance.
        - wordsize: Size of the machine word (4 or 8 bytes).
        - word_type: Data type of the machine word (np.uint32 or np.uint64).
        - word_fmt: Format of the machine word for packing and unpacking.
        - v2o: Mapping of virtual to offset.
        - o2v: Mapping of offset to virtual.
        - pmasks: Permission masks.
        - minimum_page: Minimum page size.
        
        Purpose:
        - To provide a base class for address translation and related functionalities.
    """
    def __init__(self, dtb, phy):
        """Initialize the AddressTranslator instance with a device tree blob and physical memory instance.
    
            Args:
            - dtb: Device tree blob.
            - phy: Physical memory instance.
    
            Description:
            - This constructor initializes the AddressTranslator instance with a device tree blob and physical memory instance.
            - It sets machine-specific attributes such as word_type, word_fmt based on the machine's word size and endianness.
    
            Purpose:
            - To initialize the AddressTranslator instance with necessary attributes and machine specifics.
    """
        self.dtb = dtb
        self.phy = phy

        # Set machine specifics
        if self.wordsize == 4:
            self.word_type = np.uint32
            if self.phy.machine_data["Endianness"] == "big":
                self.word_fmt = ">u4"
            else:
                self.word_fmt = "<u4"
        else:
            self.word_type = np.uint64
            if self.phy.machine_data["Endianness"] == "big":
                self.word_fmt = ">u8"
            else:
                self.word_fmt = "<u8"

        self.v2o = None
        self.o2v = None
        self.pmasks = None
        self.minimum_page = 0

    def _read_entry(self, idx, entry, lvl):
        """Decode radix tree entry"""
        raise NotImplementedError

    def _reconstruct_permissions(self, pmask):
        """Reconstruct permission masks from radix tree entry"""
        raise NotImplementedError

    def _finalize_virt_addr(self, virt_addr, permissions):
        """Apply architecture specific virtual address modifications"""
        raise NotImplementedError

    def get_data_virt(self, vaddr, size=1):
        """Return data starting from a virtual address"""
        size_available, intervals = self.v2o.contains(vaddr, size)
        if size_available != size:
            return bytes()

        ret = bytearray()
        for interval in intervals:
            _, interval_size, offset = interval
            ret.extend(self.elf_buf[offset : offset + interval_size].tobytes())

        return ret

    def get_data_phy(self, paddr, size):
        """Return data starting from a physical address"""
        return self.phy.get_data(paddr, size)

    def get_data_raw(self, offset, size):
        """Return data starting from an ELF offset"""
        return self.phy.get_data_raw(offset, size)

    def _explore_radixtree(
        self, table_addr, mapping, reverse_mapping, lvl=0, prefix=0, upmask=list()
    ):
        """Explore a radix tree to obtain virtual <-> physical mappings.
        
        Args:
            self: The instance of the calling class.
            table_addr: The address of the radix tree table.
            mapping: A dictionary to store virtual <-> physical mappings.
            reverse_mapping: A dictionary to store reverse mappings from physical to virtual addresses.
            lvl (optional): The level of the radix tree being explored (default is 0).
            prefix (optional): The prefix used for constructing virtual addresses (default is 0).
            upmask (optional): A list containing the permission mask (default is an empty list).
        
        Expected Results:
            Exploration of a radix tree to obtain mappings.
        
        Purpose:
            To traverse a radix tree and obtain virtual <-> physical mappings.
        
        Functioning:
            - The function starts by retrieving the data of the radix tree table.
            - It iterates through each entry of the table, checking for validity.
            - If the entry is valid, it computes the virtual address and permission mask.
            - If it's the last level of the radix tree or a leaf node, it checks if the page is in RAM or memory-mapped devices.
            - If the page is in RAM, it constructs the virtual address and adds it to the mappings dictionary.
            - If it's not the last level, it recursively calls itself to explore lower-level entries.
        """


        table = self.phy.get_data(table_addr, self.table_sizes[lvl])
        if not table:
            print(
                f"Table {hex(table_addr)} size:{self.table_sizes[lvl]} at level {lvl} not in RAM"
            )
            return

        for index, entry in enumerate(iter_unpack(self.unpack_fmt, table)):
            is_valid, pmask, phy_addr, page_size = self._read_entry(
                index, entry[0], lvl
            )

            if not is_valid:
                continue

            virt_addr = prefix | (index << self.shifts[lvl])
            pmask = upmask + pmask

            if (lvl == self.total_levels - 1) or page_size:  # Last radix level or Leaf
                # Ignore pages not in RAM (some OSs map more RAM than available) and not memory mapped devices
                in_ram = self.phy.in_ram(phy_addr, page_size)
                in_mmd = self.phy.in_mmd(phy_addr, page_size)
                if not in_ram and not in_mmd:
                    continue

                permissions = self._reconstruct_permissions(pmask)
                virt_addr = self._finalize_virt_addr(virt_addr, permissions)
                mapping[permissions].append((virt_addr, page_size, phy_addr, in_mmd))

                # Add only RAM address to the reverse translation P2V
                if in_ram and not in_mmd:
                    if permissions not in reverse_mapping:
                        reverse_mapping[permissions] = defaultdict(list)
                    reverse_mapping[permissions][(phy_addr, page_size)].append(
                        virt_addr
                    )
            else:
                # Lower level entry
                self._explore_radixtree(
                    phy_addr,
                    mapping,
                    reverse_mapping,
                    lvl=lvl + 1,
                    prefix=virt_addr,
                    upmask=pmask,
                )

    def _compact_intervals_virt_offset(self, intervals):
        """Functioning:
            - Compacts intervals if virtual addresses and offset values are contiguous (virt -> offset).
            - Iterates through intervals and merges contiguous ones.
            - Appends the merged intervals to the `fused_intervals` list.
            
            Args:
                self: The instance of the calling class.
                intervals: List of intervals containing virtual addresses, end addresses, physical pages, and permission masks.
            
            Expected Results:
                Compact intervals for efficient processing of virtual addresses and offset values.
            
            Purpose:
                To compact intervals if virtual addresses and offset values are contiguous, enhancing processing efficiency.
        """
        fused_intervals = []
        prev_begin = prev_end = prev_offset = -1
        for interval in intervals:
            begin, end, phy, _ = interval

            offset = self.phy.p2o[phy]
            if offset == -1:
                continue

            if prev_end == begin and prev_offset + (prev_end - prev_begin) == offset:
                prev_end = end
            else:
                fused_intervals.append((prev_begin, (prev_end, prev_offset)))
                prev_begin = begin
                prev_end = end
                prev_offset = offset

        if prev_begin != begin:
            fused_intervals.append((prev_begin, (prev_end, prev_offset)))
        else:
            offset = self.phy.p2o[phy]
            if offset == -1:
                print(f"ERROR!! {phy}")
            else:
                fused_intervals.append((begin, (end, offset)))
        return fused_intervals[1:]

    def _compact_intervals_permissions(self, intervals):
        """Functioning:
            - Compacts intervals if virtual addresses are contiguous and permissions are equal.
            - Iterates through intervals and merges contiguous ones with equal permissions.
            - Appends the merged intervals to the `fused_intervals` list.
            
            Args:
                self: The instance of the calling class.
                intervals: List of intervals containing virtual addresses, end addresses, physical pages, and permission masks.
            
            Expected Results:
                Compact intervals for efficient processing of contiguous virtual addresses with equal permissions.
            
            Purpose:
                To compact intervals if virtual addresses are contiguous and permissions are equal, enhancing processing efficiency.s
        """
        fused_intervals = []
        prev_begin = prev_end = -1
        prev_pmask = (0, 0)
        for interval in intervals:
            begin, end, _, pmask = interval
            if prev_end == begin and prev_pmask == pmask:
                prev_end = end
            else:
                fused_intervals.append((prev_begin, (prev_end, prev_pmask)))
                prev_begin = begin
                prev_end = end
                prev_pmask = pmask

        if prev_begin != begin:
            fused_intervals.append((prev_begin, (prev_end, prev_pmask)))
        else:
            fused_intervals.append((begin, (end, pmask)))

        return fused_intervals[1:]

    def _reconstruct_mappings(self, table_addr, upmask):
        """
         Args:
            self: The instance of the calling class.
            table_addr: The address of the radix tree table.
            upmask: The permission mask.
        
        Expected Results:
            Reconstruction of memory mappings for efficient processing.
        
        Purpose:
            To reconstruct memory mappings using a radix tree traversal approach and organize them for further processing.
        
        Exploring Radix Tree:
        - Initiates mappings and calls `_explore_radixtree()` with necessary parameters.
        - Populates `mapping` and `reverse_mapping` with mappings obtained from the radix tree.
        
        ELF Virtual Mapping Reconstruction:
        - Sets `reverse_mapping` and `mapping` attributes for potential future reference.
        - Likely essential for reconstructing ELF (Executable and Linkable Format) virtual mappings.
        
        Collecting Intervals:
        - Collects intervals based on mappings, excluding user-inaccessible pages.
        - Compiles intervals with start address, end address, physical page, and permission mask.
        
        Interval Fusion:
        - Performs operations to compact intervals for efficiency.
        - Reduces the number of elements for faster processing.
        
        Offset to Virtual Mapping:
        - Translates physical offsets to virtual addresses.
        - Creates intervals for physical offsets to virtual addresses based on `reverse_mapping`.
        
        Sorting Intervals:
        - Sorts the intervals obtained from the previous steps.
        - Essential for organized and efficient processing.
        
        Creating Resolution Objects:
        - Constructs resolution objects using collected and sorted interval data.
        - Includes objects such as `IMOffsets`, `IMOverlapping`, `IMData`.
        
        Conclusion:
        - The function aims to reconstruct memory mappings using a radix tree traversal approach.
        """
        # Explore the radix tree
        mapping = defaultdict(list)
        reverse_mapping = {}
        self._explore_radixtree(table_addr, mapping, reverse_mapping, upmask=upmask)

        # Needed for ELF virtual mapping reconstruction
        self.reverse_mapping = reverse_mapping
        self.mapping = mapping

        # Collect all intervals (start, end+1, phy_page, pmask)
        intervals = []
        for pmask, mapping_p in mapping.items():
            if pmask[1] == 0:  # Ignore user not accessible pages
                print(pmask)
                continue
            intervals.extend(
                [(x[0], x[0] + x[1], x[2], pmask) for x in mapping_p if not x[3]]
            )  # Ignore MMD
        intervals.sort()

        if not intervals:
            raise Exception
        # Fuse intervals in order to reduce the number of elements to speed up
        fused_intervals_v2o = self._compact_intervals_virt_offset(intervals)
        fused_intervals_permissions = self._compact_intervals_permissions(intervals)

        # Offset to virtual is impossible to compact in a easy way due to the
        # multiple-to-one mapping. We order the array and use bisection to find
        # the possible results and a partial
        intervals_o2v = []
        for pmasks, d in reverse_mapping.items():
            if pmasks[1] != 0:  # Ignore user accessible pages
                continue
            for k, v in d.items():
                # We have to translate phy -> offset
                offset = self.phy.p2o[k[0]]
                if offset == -1:  # Ignore unresolvable pages
                    continue
                intervals_o2v.append((offset, k[1] + offset, tuple(v)))
        intervals_o2v.sort()

        # Fill resolution objects
        self.v2o = IMOffsets(*list(zip(*fused_intervals_v2o)))
        self.o2v = IMOverlapping(intervals_o2v)
        self.pmasks = IMData(*list(zip(*fused_intervals_permissions)))

    def export_virtual_memory_elf(self, elf_filename):
        
        """Create an ELF file containing the virtual address space of the process.
        Args:
            self: The instance of the class invoking the function.
            elf_filename: Name of the file to create.

        Raises:
            IOError: An error occurred while accessing the ELF file.
        
        Objectives:
            - Create an ELF file containing the virtual address space of the process.
        
        Functioning:
            - The function begins by creating the ELF header and writing it to the file.
            - It then iterates through intervals of memory mappings to compact them for efficiency.
            - Next, it writes segments in the new file and fills the program header.
            - Finally, it modifies the ELF header to point to the program header for consistency and validity of the ELF file.
        """

        with open(elf_filename, "wb") as elf_fd:
            # Create the ELF header and write it on the file
            machine_data = self.phy.get_machine_data()
            endianness = machine_data["Endianness"]
            machine = machine_data["Architecture"].lower()

            # Create ELF main header
            if "aarch64" in machine:
                e_machine = 0xB7
            elif "arm" in machine:
                e_machine = 0x28
            elif "riscv" in machine:
                e_machine = 0xF3
            elif "x86_64" in machine:
                e_machine = 0x3E
            elif "386" in machine:
                e_machine = 0x03
            else:
                raise Exception("Unknown architecture")

            e_ehsize = 0x40
            e_phentsize = 0x38
            elf_h = bytearray(e_ehsize)
            elf_h[0x00:0x04] = b"\x7fELF"  # Magic
            elf_h[0x04] = 2  # Elf type
            elf_h[0x05] = 1 if endianness == "little" else 2  # Endianness
            elf_h[0x06] = 1  # Version
            elf_h[0x10:0x12] = 0x4.to_bytes(2, endianness)  # e_type
            elf_h[0x12:0x14] = e_machine.to_bytes(2, endianness)  # e_machine
            elf_h[0x14:0x18] = 0x1.to_bytes(4, endianness)  # e_version
            elf_h[0x34:0x36] = e_ehsize.to_bytes(2, endianness)  # e_ehsize
            elf_h[0x36:0x38] = e_phentsize.to_bytes(2, endianness)  # e_phentsize
            elf_fd.write(elf_h)

            # For each pmask try to compact intervals in order to reduce the number of segments
            intervals = defaultdict(list)
            for (kpmask, pmask), intervals_list in self.mapping.items():
                print(kpmask, pmask)

                if pmask == 0:  # Ignore pages not accessible by the process
                    continue

                intervals[pmask].extend(
                    [(x[0], x[0] + x[1], x[2]) for x in intervals_list if not x[3]]
                )  # Ignore MMD
                intervals[pmask].sort()

                if len(intervals[pmask]) == 0:
                    intervals.pop(pmask)
                    continue

                # Compact them
                fused_intervals = []
                prev_begin = prev_end = prev_offset = -1
                for interval in intervals[pmask]:
                    begin, end, phy = interval

                    offset = self.phy.p2o[phy]
                    if offset == -1:
                        continue

                    if (
                        prev_end == begin
                        and prev_offset + (prev_end - prev_begin) == offset
                    ):
                        prev_end = end
                    else:
                        fused_intervals.append([prev_begin, prev_end, prev_offset])
                        prev_begin = begin
                        prev_end = end
                        prev_offset = offset

                if prev_begin != begin:
                    fused_intervals.append([prev_begin, prev_end, prev_offset])
                else:
                    offset = self.phy.p2o[phy]
                    if offset == -1:
                        print(f"ERROR!! {phy}")
                    else:
                        fused_intervals.append([begin, end, offset])
                intervals[pmask] = sorted(
                    fused_intervals[1:], key=lambda x: x[1] - x[0], reverse=True
                )

            # Write segments in the new file and fill the program header
            p_offset = len(elf_h)
            offset2p_offset = (
                {}
            )  # Slow but more easy to implement (best way: a tree sort structure able to be updated)
            e_phnum = 0

            for pmask, interval_list in intervals.items():
                e_phnum += len(interval_list)
                for idx, interval in enumerate(interval_list):
                    begin, end, offset = interval
                    size = end - begin
                    if offset not in offset2p_offset:
                        elf_fd.write(self.phy.get_data_raw(offset, size))
                        if not self.phy.get_data_raw(offset, size):
                            print(hex(offset), hex(size))
                        new_offset = p_offset
                        p_offset += size
                        for page_idx in range(0, size, self.minimum_page):
                            offset2p_offset[offset + page_idx] = new_offset + page_idx
                    else:
                        new_offset = offset2p_offset[offset]
                    interval_list[idx].append(
                        new_offset
                    )  # Assign the new offset in the dest file

            # Create the program header containing all the segments (ignoring not in RAM pages)
            e_phoff = elf_fd.tell()
            p_header = bytes()
            for pmask, interval_list in intervals.items():
                for begin, end, offset, p_offset in interval_list:
                    p_filesz = end - begin

                    # Back convert offset to physical page
                    p_addr = self.phy.o2p[offset]
                    assert p_addr != -1

                    segment_entry = bytearray(e_phentsize)
                    segment_entry[0x00:0x04] = 0x1.to_bytes(4, endianness)  # p_type
                    segment_entry[0x04:0x08] = pmask.to_bytes(4, endianness)  # p_flags
                    segment_entry[0x10:0x18] = begin.to_bytes(8, endianness)  # p_vaddr
                    segment_entry[0x18:0x20] = p_addr.to_bytes(
                        8, endianness
                    )  # p_paddr Original physical address
                    segment_entry[0x28:0x30] = p_filesz.to_bytes(
                        8, endianness
                    )  # p_memsz
                    segment_entry[0x08:0x10] = p_offset.to_bytes(
                        8, endianness
                    )  # p_offset
                    segment_entry[0x20:0x28] = p_filesz.to_bytes(
                        8, endianness
                    )  # p_filesz

                    p_header += segment_entry

            # Write the segment header
            elf_fd.write(p_header)
            s_header_pos = (
                elf_fd.tell()
            )  # Last position written (used if we need to write segment header)

            # Modify the ELF header to point to program header
            elf_fd.seek(0x20)
            elf_fd.write(e_phoff.to_bytes(8, endianness))  # e_phoff

            # If we have more than 65535 segments we have create a special Section entry contains the
            # number of program entry (as specified in ELF64 specifications)
            if e_phnum < 65536:
                elf_fd.seek(0x38)
                elf_fd.write(e_phnum.to_bytes(2, endianness))  # e_phnum
            else:
                elf_fd.seek(0x28)
                elf_fd.write(s_header_pos.to_bytes(8, endianness))  # e_shoff
                elf_fd.seek(0x38)
                elf_fd.write(0xFFFF.to_bytes(2, endianness))  # e_phnum
                elf_fd.write(0x40.to_bytes(2, endianness))  # e_shentsize
                elf_fd.write(0x1.to_bytes(2, endianness))  # e_shnum

                section_entry = bytearray(0x40)
                section_entry[0x2C:0x30] = e_phnum.to_bytes(4, endianness)  # sh_info
                elf_fd.seek(s_header_pos)
                elf_fd.write(section_entry)


class IntelTranslator(AddressTranslator):
    @staticmethod
    def derive_mmu_settings(mmu_class, regs_dict, mphy):
        if mmu_class is IntelAMD64:
            dtb = ((regs_dict["cr3"] >> 12) & ((1 << (mphy - 12)) - 1)) << 12

        elif mmu_class is IntelIA32:
            dtb = ((regs_dict["cr3"] >> 12) & (1 << 20) - 1) << 12
            mphy = min(mphy, 40)

        else:
            raise NotImplementedError

        return {
            "dtb": dtb,
            "wp": True,
            "ac": False,
            "nxe": True,
            "smep": False,
            "smap": False,
            "mphy": mphy,
        }

    @staticmethod
    def derive_translator_class(mmu_mode):
        if mmu_mode == "ia64":
            return IntelAMD64
        elif mmu_mode == "pae":
            return NotImplementedError
        elif mmu_mode == "ia32":
            return IntelIA32
        else:
            raise NotImplementedError

    @staticmethod
    def factory(phy, mmu_values):
        machine_data = phy.get_machine_data()
        mmu_mode = machine_data["MMUMode"]
        mphy = machine_data["CPUSpecifics"]["MAXPHYADDR"]

        translator_c = IntelTranslator.derive_translator_class(mmu_mode)
        mmu_settings = IntelTranslator.derive_mmu_settings(
            translator_c, mmu_values, mphy
        )
        return translator_c(phy=phy, **mmu_settings)

    def __init__(
        self, dtb, phy, mphy, wp=False, ac=False, nxe=False, smap=False, smep=False
    ):
        super(IntelTranslator, self).__init__(dtb, phy)
        self.mphy = mphy
        self.wp = wp
        self.ac = ac  # UNUSED by Fossil
        self.smap = smap
        self.nxe = nxe
        self.smep = smep
        self.minimum_page = 0x1000

        print("Creating resolution trees...")
        self._reconstruct_mappings(self.dtb, upmask=[[False, True, True]])

    def _finalize_virt_addr(self, virt_addr, permissions):
        return virt_addr


class IntelIA32(IntelTranslator):
    def __init__(
        self, dtb, phy, mphy, wp=True, ac=False, nxe=False, smap=False, smep=False
    ):
        self.unpack_fmt = "<I"
        self.total_levels = 2
        self.prefix = 0x0
        self.table_sizes = [0x1000, 0x1000]
        self.shifts = [22, 12]
        self.wordsize = 4

        super(IntelIA32, self).__init__(dtb, phy, mphy, wp, ac, nxe, smap, smep)

    def _read_entry(self, idx, entry, lvl):
        # Return (is_Valid, Permissions flags, Table Address, Size)

        # Empty entry
        if not (entry & 0x1):
            return False, tuple(), 0, 0

        else:
            perms_flags = [
                [not bool(entry & 0x4), bool(entry & 0x2), True]  # K  # W  # X
            ]

            # Upper tables pointers
            if not (entry & 0x80) and (lvl == 0):
                addr = ((entry >> 12) & ((1 << 20) - 1)) << 12
                return True, perms_flags, addr, 0

            # Leaf
            else:
                if lvl == 0:
                    addr = (((entry >> 13) & ((1 << (self.mphy - 32)) - 1)) << 32) | (
                        ((entry >> 22) & ((1 << 10) - 1)) << 22
                    )
                else:
                    addr = ((entry >> 12) & ((1 << 20) - 1)) << 12
                return True, perms_flags, addr, 1 << self.shifts[lvl]

    def _reconstruct_permissions(self, pmask):
        k_flags, w_flags, _ = zip(*pmask)

        # Kernel page in user mode
        if any(k_flags):
            r = True
            w = all(w_flags) if self.wp else True
            return r << 2 | w << 1 | 1, 0

        # User page in user mode
        else:
            r = True
            w = all(w_flags)
            return 0, r << 2 | w << 1 | 1


class IntelAMD64(IntelTranslator):
    def __init__(
        self, dtb, phy, mphy, wp=True, ac=False, nxe=True, smap=False, smep=False
    ):
        self.unpack_fmt = "<Q"
        self.total_levels = 4
        self.prefix = 0xFFFF800000000000
        self.table_sizes = [0x1000] * 4
        self.shifts = [39, 30, 21, 12]
        self.wordsize = 8

        super(IntelAMD64, self).__init__(dtb, phy, mphy, wp, ac, nxe, smap, smep)

    def _read_entry(self, idx, entry, lvl):
        # Return (is_Valid, Permissions flags, Table Address, Size)

        # Empty entry
        if not (entry & 0x1):
            return False, tuple(), 0, 0

        else:
            perms_flags = [
                [
                    not bool(entry & 0x4),  # K
                    bool(entry & 0x2),  # W
                    not bool(entry & 0x8000000000000000),  # X
                ]
            ]

            # Upper tables pointers
            if (not (entry & 0x80) and lvl < 3) or lvl == 0:  # PTL4 does not have leaf
                addr = ((entry >> 12) & ((1 << (self.mphy - 12)) - 1)) << 12
                return True, perms_flags, addr, 0

            # Leaf
            else:
                addr = (
                    (entry >> self.shifts[lvl])
                    & ((1 << (self.mphy - self.shifts[lvl])) - 1)
                ) << self.shifts[lvl]
                return True, perms_flags, addr, 1 << self.shifts[lvl]

    def _reconstruct_permissions(self, pmask):
        k_flags, w_flags, x_flags = zip(*pmask)

        # Kernel page in user mode
        if any(k_flags):
            r = True
            w = all(w_flags) if self.wp else True
            x = all(x_flags) if self.nxe else True

            return r << 2 | w << 1 | int(x), 0

        # User page in user mode
        else:
            r = True
            w = all(w_flags)
            x = all(x_flags) if self.nxe else True

            return 0, r << 2 | w << 1 | int(x)

    def _finalize_virt_addr(self, virt_addr, permissions):
        # Canonical address form
        if virt_addr & 0x800000000000:
            return self.prefix | virt_addr
        else:
            return virt_addr


class RISCVTranslator(AddressTranslator):
    """Description:
        - Represents a translator for converting physical addresses to virtual addresses in RISC-V architecture.
        - Provides methods for deriving MMU settings, selecting translator class, creating translator instances, and handling virtual address mapping.
       Purpose:
        - To provide functionality for translating physical addresses to virtual addresses in RISC-V architecture.
"""
    @staticmethod
    def derive_mmu_settings(mmu_class, regs_dict):
        """derive_mmu_settings(mmu_class, regs_dict):
            - Static method to derive MMU (Memory Management Unit) settings based on the MMU class and register dictionary.
            - Returns MMU settings including the DTB (Translation Base Register) and control flags.
        """
        dtb = regs_dict["satp"]
        return {"dtb": dtb, "Sum": False, "mxr": False}

    @staticmethod
    def derive_translator_class(mmu_mode):
        if mmu_mode == "sv39":
            return RISCVSV39
        else:
            return RISCVSV32

    @staticmethod
    def factory(phy, mmu_values):
        machine_data = phy.get_machine_data()
        mmu_mode = machine_data["MMUMode"]
        translator_c = RISCVTranslator.derive_translator_class(mmu_mode)
        mmu_settings = RISCVTranslator.derive_mmu_settings(translator_c, mmu_values)
        return translator_c(phy=phy, **mmu_settings)

    def __init__(self, dtb, phy, Sum=True, mxr=True):
        super(RISCVTranslator, self).__init__(dtb, phy)
        self.Sum = Sum
        self.mxr = mxr
        self.minimum_page = 0x1000

        print("Creating resolution trees...")
        self._reconstruct_mappings(self.dtb, upmask=[[False, True, True, True]])

    def _finalize_virt_addr(self, virt_addr, permissions):
        return virt_addr

    def _reconstruct_permissions(self, pmask):
        """_reconstruct_permissions(self, pmask):
            - Method to reconstruct permissions based on the permission mask.
            - Extracts permission flags (read, write, execute) from the permission mask.
            - Sets permission bits for kernel and user modes.
            - Returns permission settings for kernel and user modes.
    """
        k_flag, r_flag, w_flag, x_flag = pmask[-1]  # No hierarchy

        r = r_flag
        if self.mxr:
            r |= x_flag

        w = w_flag
        x = x_flag

        # Kernel page in user mode
        if k_flag:
            return r << 2 | w << 1 | int(x), 0

        # User page in user mode
        else:
            return 0, r << 2 | w << 1 | int(x)


class RISCVSV32(RISCVTranslator):
    """Description:
        - Represents a translator for RISC-V SV32 addressing mode.
        - Inherits from the RISCVTranslator class.
    
        Attributes:
            - unpack_fmt: Format string for unpacking binary data.
            - total_levels: Total levels in the radix tree.
            - prefix: Prefix for virtual addresses.
            - table_sizes: Sizes of radix tree tables at each level.
            - shifts: Shift values for computing virtual addresses.
            - wordsize: Size of a word in bytes.
    
        Methods:
            - __init__(self, dtb, phy, Sum, mxr): Constructor method to initialize RISCVSV32 instance.
            - _read_entry(self, idx, entry, lvl): Method to read a radix tree entry.
    
        Purpose:
            - To provide functionality for translating physical addresses to virtual addresses in RISC-V SV32 addressing mode.
    """
    def __init__(self, dtb, phy, Sum, mxr):
        self.unpack_fmt = "<I"
        self.total_levels = 2
        self.prefix = 0x0
        self.table_sizes = [0x1000, 0x1000]
        self.shifts = [22, 12]
        self.wordsize = 4

        super(RISCVSV32, self).__init__(dtb, phy, Sum, mxr)

    def _read_entry(self, idx, entry, lvl):
        # Return (is_Valid, Permissions flags, Table Address, Size)

        # Empty entry
        if not (entry & 0x1):
            return False, tuple(), 0, 0

        else:
            k = not bool(entry & 0x10)
            r = bool(entry & 0x2)
            w = bool(entry & 0x4)
            x = bool(entry & 0x8)
            perms_flags = [[k, r, w, x]]

            addr = ((entry >> 10) & ((1 << 22) - 1)) << 12
            # Leaf
            if r or w or x or lvl == 1:
                return True, perms_flags, addr, 1 << self.shifts[lvl]
            else:
                # Upper tables pointers
                return True, perms_flags, addr, 0


class RISCVSV39(RISCVTranslator):
    def __init__(self, dtb, phy, Sum, mxr):
        """Initialize the RISCVSV39 instance with specific attributes for RV39 address translation.

        Args:
        - dtb: Device tree blob.
        - phy: Physical memory instance.
        - Sum: Sum attribute (unspecified type).
        - mxr: Mxr attribute (unspecified type).

        Description:
        - This constructor initializes the RISCVSV39 instance with specific attributes required for RV39 address translation.

        Attributes:
        - unpack_fmt: Format string for unpacking binary data.
        - total_levels: Total levels of the page table hierarchy.
        - prefix: Prefix value for address translation.
        - table_sizes: Sizes of the page tables at each level.
        - shifts: Shift values for calculating addresses.
        - wordsize: Size of the machine word.
    """
        self.unpack_fmt = "<Q"
        self.total_levels = 3
        self.prefix = 0x0
        self.table_sizes = [0x1000, 0x1000, 0x1000]
        self.shifts = [30, 21, 12]
        self.wordsize = 8

        super(RISCVSV39, self).__init__(dtb, phy, Sum, mxr)

    def _read_entry(self, idx, entry, lvl):
        """Decode the entry in the page table.

        Args:
        - idx: Index of the entry.
        - entry: Entry value in the page table.
        - lvl: Level of the page table hierarchy.

        Returns:
        - is_Valid: Boolean indicating if the entry is valid.
        - Permissions flags: Tuple containing permissions flags for the entry.
        - Table Address: Address of the table.
        - Size: Size of the table entry.

        Description:
        - This method decodes an entry in the page table and returns information such as validity, permissions flags, address, and size.

        Purpose:
        - To decode entries in the page table for address translation.

        Technical Explanation:
        - It checks if the entry is empty and returns appropriate values if it is.
        - If the entry is not empty, it extracts permissions flags and calculates the address and size of the entry.
        - For leaf entries, it returns the address and size directly. For non-leaf entries, it returns the address and a size of 0.
        """

        # Empty entry
        if not (entry & 0x1):
            return False, tuple(), 0, 0

        else:
            k = not bool(entry & 0x10)
            r = bool(entry & 0x2)
            w = bool(entry & 0x4)
            x = bool(entry & 0x8)
            perms_flags = [[k, r, w, x]]

            addr = ((entry >> 10) & ((1 << 44) - 1)) << 12
            # Leaf
            if r or w or x or lvl == 2:
                return True, perms_flags, addr, 1 << self.shifts[lvl]
            else:
                # Upper tables pointers
                return True, perms_flags, addr, 0


if __name__ == "__main__":
    main()
