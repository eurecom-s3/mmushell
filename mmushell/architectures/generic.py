import sys
import gc
import os
import random
import logging
import portion
import importlib
import multiprocessing as mp

from miasm.jitter.VmMngr import Vm
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, PAGE_EXEC
from miasm.core.locationdb import LocationDB
from miasm.core.bin_stream import bin_stream_vm
from miasm.ir.translators.python import TranslatorPython
from miasm.ir.symbexec import SymbolicExecutionEngine
from miasm.expression.expression import ExprInt, ExprId
from miasm.core.bin_stream import bin_stream_file
from miasm.analysis.depgraph import DependencyGraph

from mmap import mmap, MAP_SHARED, PROT_READ, MADV_HUGEPAGE
from tqdm import tqdm
from itertools import chain
from compress_pickle import load, dump
from cmd import Cmd
from IPython import embed
from collections import defaultdict, deque
from dataclasses import dataclass, field
from copy import deepcopy
from pickle import load as Load
from enum import IntEnum
from typing import Any, Dict
from time import sleep
from random import uniform
from struct import iter_unpack, unpack
from copy import deepcopy, copy


logger = logging.getLogger(__name__)


# Disable print() from MIASM
class DisableLogs:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CPUReg:
    """Represents a CPU register"""

    def is_valid(self, value) -> bool:
        """Check if the value is valid for the register

        Being a valid register is very specific to the architecture, so this function is aimed to be overloaded by child classes

        Args:
            value: the value to check

        Returns:
            True if the value is valid, False otherwise
        """
        return True

    def is_mmu_equivalent_to(self, other_reg):
        """Check if the register is equivalent to another register

        Args:
            other_reg: the other register to compare

        See child classes for more details
        """
        raise NotImplementedError

    def __hash__(self):
        """Hash of the contained value from register"""
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class TableEntry:
    """Represents a table Page Table Entry

    It holds the mapping between a virtual address of a page and the address of a physical frame.
    There is also auxiliary information about the page such as a present bit, a dirty or modified bit, address space or process ID information, amongst others.

    Aimed to be inherited by child classes to represent different architectures

    Attributes:
        address: the virtual address of the page
        flags: the flags of the page
    """

    def __init__(self, address, flags, *args):
        """Initialize the Table Entry

        Args:
            address: the virtual address of the page
            flags: the flags of the page
            *args: additional arguments
        """
        self.address = address
        self.flags = flags


class PageTable:
    """Represents a Page Table

    A page table is the data structure used by a virtual memory system in a computer operating system to store the mapping between virtual addresses and physical addresses.

    Aimed to be inherited by child classes to represent different architectures

    Attributes:
        address: the address of the page table
        size: the size of the page table
        entries: the entries of the page table
        levels: the levels of the page table
    """

    entry_size = 0

    def __init__(self, address, size, entries, levels, *args):
        """Initialize the Page Table

        Args:
            address: the address of the page table
            size: the size of the page table
            entries: the entries of the page table
            levels: the levels of the page table
            *args: additional arguments
        """
        self.address = address
        self.size = size
        self.entries = entries
        self.levels = levels

    def apply_on_entries(self, f: function, args):
        """Run a function to all the entries of the page table.

        The provided function should take an entry and the arguments as input and return the result of the operation.

        Args:
            f: the function to apply
            args: the arguments to pass to the function

        Returns:
            a list with the results of the function applied to all the entries
        """
        res = []
        for entry in self.entries:
            res.append(f(entry, args))
        return res


def perms_bool_to_string(kr, kw, kx, r, w, x):
    """Convert a set of permissions from boolean to string

    Args:
        kr: read permission for the kernel
        kw: write permission for the kernel
        kx: execute permission for the kernel
        r: read permission for the user
        w: write permission for the user
        x: execute permission for the user

    Returns:
        a string with the permissions
    """
    perm_s = "R" if kr else "-"
    perm_s += "W" if kw else "-"
    perm_s += "X" if kx else "-"
    perm_s += "r" if r else "-"
    perm_s += "w" if w else "-"
    perm_s += "x" if x else "-"
    return perm_s


class RadixTree:
    labels = [
        "Radix address",
        "First level",
        "Kernel size (Bytes)",
        "User size (Bytes)",
    ]
    addr_fmt = "0x{:016x}"

    def __init__(self, top_table, init_level, pas, vas, kernel=True, user=True):
        self.top_table = top_table
        self.init_level = init_level
        self.pas = pas
        self.vas = vas
        self.kernel = kernel
        self.user = user

    def __repr__(self):
        e_resume = self.entry_resume_stringified()
        return str(
            [self.labels[i] + ": " + str(e_resume[i]) for i in range(len(self.labels))]
        )

    def entry_resume(self):
        return [
            self.top_table,
            self.init_level,
            self.pas.get_kernel_size(),
            self.pas.get_user_size(),
        ]

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[0] = self.addr_fmt.format(res[0])
        for idx, r in enumerate(res[1:], start=1):
            res[idx] = str(r)
        return res


class VAS(defaultdict):
    def __init__(self, *args, **kwargs):
        super(VAS, self).__init__()
        self.default_factory = portion.empty

    def __repr__(self):
        s = ""
        for k in self:
            k_str = perms_bool_to_string(*k)
            s += k_str + "\n"
            for interval in self[k]:
                s += f"\t[{hex(interval.lower)}, {hex(interval.upper)}]\n"
        return s

    def hierarchical_extend(self, other, uperms):
        for perm in other:
            new_perm = []
            for i in range(6):
                new_perm.append(perm[i] and uperms[i])
            new_perm = tuple(new_perm)
            self[new_perm] |= other[perm]


@dataclass
class PAS:
    space: Dict = field(default_factory=lambda: defaultdict(dict))
    space_size: Dict = field(default_factory=lambda: defaultdict(int))

    def hierarchical_extend(self, other, uperms):
        for perm in other.space:
            new_perm = []
            for i in range(6):
                new_perm.append(perm[i] and uperms[i])
            new_perm = tuple(new_perm)
            self.space[new_perm].update(other.space[perm])
            self.space_size[new_perm] += other.space_size[perm]

    def __contains__(self, key):
        for addresses in self.space.values():
            for address in addresses:
                if address <= key < address + addresses[address]:
                    return True
        return False

    def is_in_kernel_space(self, key):
        for perms, addresses in self.space.items():
            if perms[0] or perms[1] or perms[2]:
                for address in addresses:
                    if address <= key < address + addresses[address]:
                        return True
        return False

    def is_in_kernel_x_space(self, key):
        for perms, addresses in self.space.items():
            if perms[0] and perms[2]:
                for address in addresses:
                    if address <= key < address + addresses[address]:
                        return True
        return False

    def is_in_user_space(self, key):
        for perms, addresses in self.space.items():
            if not (perms[0] or perms[1] or perms[2]):
                for address in addresses:
                    if address <= key < address + addresses[address]:
                        return True
        return False

    def __repr__(self):
        ret = ""
        for perm in self.space:
            symb = lambda x, s: s if x else "-"
            ret += "{}{}{} {}{}{}: {}\n".format(
                symb(perm[0], "R"),
                symb(perm[1], "W"),
                symb(perm[2], "X"),
                symb(perm[3], "R"),
                symb(perm[4], "W"),
                symb(perm[5], "X"),
                self.space_size[perm],
            )
        return ret

    def get_kernel_size(self):
        size = 0
        for perm in self.space:
            if not (perm[3] or perm[4] or perm[5]):
                size += self.space_size[perm]
        return size

    def get_user_size(self):
        size = 0
        for perm in self.space:
            if perm[3] or perm[4] or perm[5]:
                size += self.space_size[perm]
        return size


class Machine:
    @classmethod
    def from_machine_config(cls, machine_config, **kwargs):
        """Create a machine starting from a YAML file descriptor"""

        # Check no intersection between memory regions
        ram_portion = portion.empty()
        for region_dict in machine_config["memspace"]["ram"]:
            region_portion = portion.closed(region_dict["start"], region_dict["end"])
            if not ram_portion.intersection(region_portion).empty:
                logger.fatal("RAM regions overlapping!")
                exit(1)
            ram_portion = ram_portion.union(region_portion)

        # Module to use
        architecture_module = importlib.import_module(
            "architectures." + machine_config["cpu"]["architecture"]
        )

        # Create CPU
        cpu = architecture_module.CPU.from_cpu_config(machine_config["cpu"])

        # Create MMU
        try:
            mmu_class = getattr(
                architecture_module, machine_config["mmu"]["mode"].upper()
            )
        except AttributeError:
            logger.fatal("Unknown MMU mode!")
            exit(1)
        mmu = mmu_class(machine_config["mmu"])

        # Create RAM
        memory = architecture_module.PhysicalMemory(machine_config["memspace"])

        return architecture_module.Machine(cpu, mmu, memory, **kwargs)

    def __init__(self, cpu, mmu, memory, **kwargs):
        self.cpu = cpu
        self.mmu = mmu
        self.memory = memory
        self.gtruth = {}
        self.data = None
        self.cpu.machine = self
        self.mmu.machine = self
        self.memory.machine = self

    def get_miasm_machine(self):
        return None

    def __del__(self):
        self.memory.close()

    def apply_parallel(
        self, frame_size, parallel_func, iterators=None, max_address=-1, **kwargs
    ):
        """Apply parallel_func using multiple core to frame_size chunks of RAM or iterators arguments"""

        # Prepare the pool
        logger.info("Parsing memory...")
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus, initializer=tqdm.set_lock, initargs=(mp.Lock(),))

        if iterators is None:
            # Create iterators for parallel execution
            _, addresses_iterators = self.memory.get_addresses(
                frame_size, cpus=cpus, max_address=max_address
            )
        else:
            addresses_iterators = iterators

        # GO!
        parsing_results_async = [
            pool.apply_async(
                parallel_func, args=(addresses_iterator, frame_size, pidx), kwds=kwargs
            )
            for pidx, addresses_iterator in enumerate(addresses_iterators)
        ]
        pool.close()
        pool.join()

        print("\n")  # Workaround for tqdm

        return parsing_results_async


class CPU:
    opcode_to_mmu_regs = None
    opcode_to_gregs = None

    @classmethod
    def from_cpu_config(cls, cpu_config, **kwargs):
        return CPU(cpu_config)

    machine = None

    def __init__(self, params):
        self.architecture = params["architecture"]
        self.bits = params["bits"]
        self.endianness = params["endianness"]
        self.processor_features = params.get("processor_features", {})
        self.registers_values = params.get("registers_values", {})

    @staticmethod
    def extract_bits_little(entry, pos, n):
        return (entry >> pos) & ((1 << n) - 1)

    @staticmethod
    def extract_bits_big(entry, pos, n):
        return (entry >> (32 - pos - n)) & ((1 << n) - 1)

    @staticmethod
    def extract_bits_big64(entry, pos, n):
        return (entry >> (64 - pos - n)) & ((1 << n) - 1)

    def parse_opcode(self, buff, page_addr, offset):
        raise NotImplementedError

    def parse_opcodes_parallel(self, addresses, frame_size, pidx, **kwargs):
        sleep(uniform(pidx, pidx + 1) // 1000)

        opcodes = {}
        mm = copy(self.machine.memory)
        mm.reopen()

        # Cicle over every frame
        total_elems, iterator = addresses
        for frame_addr in tqdm(
            iterator, position=-pidx, total=total_elems, leave=False
        ):
            frame_buf = mm.get_data(
                frame_addr, frame_size
            )  # We parse memory in PAGE_SIZE chunks

            for idx, opcode in enumerate(
                iter_unpack(self.processor_features["opcode_unpack_fmt"], frame_buf)
            ):
                opcode = opcode[0]
                opcodes.update(
                    self.parse_opcode(
                        opcode, frame_addr, idx * self.processor_features["instr_len"]
                    )
                )

        return opcodes

    def find_registers_values_dataflow(self, opcodes, zero_registers=[]):
        # Miasm require to define a memory() function to access to the underlaying
        # memory layer during the Python translation
        # WORKAROUND: memory() does not permit more than 2 args...
        endianness = self.endianness

        def memory(addr, size):
            return int.from_bytes(self.machine.memory.get_data(addr, size), endianness)

        machine = self.machine.get_miasm_machine()
        vm = self.machine.memory.get_miasm_vmmngr()
        mdis = machine.dis_engine(
            bin_stream_vm(vm), dont_dis_nulstart_bloc=False, loc_db=LocationDB()
        )

        ir_arch = machine.ira(mdis.loc_db)
        py_transl = TranslatorPython()

        # Disable MIASM logging
        logging.getLogger("asmblock").disabled = True

        registers_values = defaultdict(set)
        # We use a stack data structure (deque) in order to manage also parent functions (EXPERIMENTAL not implemented here)
        instr_deque = deque([(addr, opcodes[addr]) for addr in opcodes])
        while len(instr_deque):
            instr_addr, instr_data = instr_deque.pop()

            # Ignore instruction with associated function not found
            if instr_data["f_addr"] == -1:
                continue

            # Do not dataflow instructions with source registers in zero_registers
            # because contain zero
            if instr_data["gpr"] in zero_registers:
                registers_values[instr_data["register"]].add(0x0)
                continue

            # Disable disassemble logging for unimplemented opcodes
            with DisableLogs():
                try:
                    # Initialize dependency graph machinery
                    mdis.dont_dis = [instr_addr + self.processor_features["instr_len"]]
                    asmcfg = mdis.dis_multiblock(instr_data["f_addr"])
                    ircfg = ir_arch.new_ircfg_from_asmcfg(asmcfg)
                    dg = DependencyGraph(ircfg)
                except Exception as e:
                    # Disassembler can raises exception if a JMP address is register dependent,
                    # in that case the analysis fails..
                    # print(e)
                    # traceback.print_exc()
                    continue

            # Collect function informations
            try:
                current_loc_key = next(iter(ircfg.getby_offset(instr_addr)))
                assignblk_index = 0
                current_block = ircfg.get_block(current_loc_key)
                for assignblk_index, assignblk in enumerate(current_block):
                    if assignblk.instr.offset == instr_addr:
                        break
            except Exception as e:
                # If the function graph does not contain the instruction or the disassembler cannot complete the
                # task due to unimplemented instructions next(iter(...)) raises a fatal exception
                # print(e)
                # traceback.print_exc()
                continue

            # Recreate default CPU config registers state and general registers to look for
            bits = self.bits
            gp_registers = [ExprId(reg, bits) for reg in instr_data["gpr"]]
            init_ctx = {
                ExprId(name.upper(), bits): ExprInt(value, bits)
                for name, value in self.registers_values.items()
            }

            # Generate solutions
            loops = 0
            for sol_nb, sol in enumerate(
                dg.get(current_block.loc_key, gp_registers, assignblk_index, set())
            ):
                # The solution contains a loop, we permit only a maximum of 10 solutions with loops...
                if sol.has_loop:
                    loops += 1
                    if loops > 10:
                        break

                # Emulate the blocks chain
                results = sol.emul(ir_arch, ctx=init_ctx)

                # Retrieve the solutions and translate it in a python expression
                for reg_name, reg_value_expr in results.items():
                    try:
                        translated_expr = py_transl.from_expr(reg_value_expr)
                        evaluated_expr = eval(translated_expr)
                        registers_values[instr_data["register"]].add(evaluated_expr)

                    except NameError:
                        # The expression depends on another register which can imply that the function is called
                        # by another one which pass it the value

                        # using parents function drastically slow down (block?) the analysis (EXPERIMENTAL not used here)
                        # for f_parent in instr_data["f_parents"]:
                        #     instr_deque.append((instr_addr, {"register": instr_data["register"],
                        #                                      "gpr": instr_data["gpr"],
                        #                                      "f_addr": f_parent,
                        #                                      "f_parents": []
                        #                                     }))

                        continue

                    except Exception as e:
                        # Possible other exceptions due to MIASM incomplete internal implementation
                        # print(e)
                        # traceback.print_exc()
                        continue

        del vm
        self.machine.memory.free_miasm_memory()
        return registers_values


class MMU:
    machine = None

    def __init__(self, mmu_config):
        self.mmu_config = mmu_config

    @staticmethod
    def extract_bits_little(entry, pos, n):
        return (entry >> pos) & ((1 << n) - 1)

    @staticmethod
    def extract_bits_big(entry, pos, n):
        return (entry >> (32 - pos - n)) & ((1 << n) - 1)

    @staticmethod
    def extract_bits_big64(entry, pos, n):
        return (entry >> (64 - pos - n)) & ((1 << n) - 1)


class MMURadix(MMU):
    PAGE_SIZE = 0
    extract_bits = None
    paging_unpack_format = ""
    page_table_class = PageTable
    radix_levels = {}
    top_prefix = 0
    entries_size = 0
    map_ptr_entries_to_levels = {}
    map_datapages_entries_to_levels = {}
    map_level_to_table_size = {}
    map_entries_to_shifts = {}
    map_reserved_entries_to_levels = {}

    def classify_entry(self, page_addr, entry):
        raise NotImplementedError

    def derive_page_address(self, addr, mode="global"):
        # Derive the addresses of pages containing the address
        addrs = []
        for lvl in range(self.radix_levels[mode] - 1, -1, -1):
            for entry_class in self.map_datapages_entries_to_levels[mode][lvl]:
                if entry_class is not None:
                    shift = self.map_entries_to_shifts[mode][entry_class]
                    addrs.append((lvl, (addr >> shift) << shift))
        return addrs

    def parse_parallel_frame(
        self, addresses, frame_size, pidx, mode="global", **kwargs
    ):
        # Every process sleep a random delay in order to desincronize access to disk and maximixe the throuput
        sleep(uniform(pidx, pidx + 1) // 1000)

        data_pages = []
        empty_tables = []
        page_tables = [{} for i in range(self.radix_levels[mode])]
        mm = copy(self.machine.memory)
        mm.reopen()

        # Cicle over every frame
        total_elems, iterator = addresses
        for frame_addr in tqdm(
            iterator, position=-pidx, total=total_elems, leave=False
        ):
            frame_buf = mm.get_data(frame_addr, frame_size)
            invalids, pt_classes, table_obj = self.parse_frame(
                frame_buf, frame_addr, frame_size
            )

            # It is a data page
            if invalids or -2 in pt_classes:
                data_pages.append(frame_addr)
            elif -1 in pt_classes:
                empty_tables.append(frame_addr)
            else:
                for pt_class in pt_classes:
                    page_tables[pt_class][frame_addr] = table_obj

        return page_tables, data_pages, empty_tables

    def parse_frame(
        self, frame_buf, frame_addr, frame_size, frame_level=-1, mode="global"
    ):
        frame_d = defaultdict(dict)
        if frame_level >= 0:
            reseved_classes = self.machine.mmu.map_reserved_entries_to_levels[mode][
                frame_level
            ]
            data_classes = self.machine.mmu.map_datapages_entries_to_levels[mode][
                frame_level
            ]
            ptr_class = self.machine.mmu.map_ptr_entries_to_levels[mode][frame_level]
            # frame_size = self.machine.mmu.map_level_to_table_size[mode][frame_level]

        invalids = 0
        empty_entries = 0
        # Unpack records inside the frame
        for entry_idx, entry in enumerate(
            iter_unpack(self.paging_unpack_format, frame_buf)
        ):
            if (
                frame_level >= 0
                and entry_idx * self.machine.mmu.entries_size >= frame_size
            ):
                break

            entry = entry[0]
            # Classify the entry
            entry_classes = self.classify_entry(frame_addr, entry)
            # It's a data page
            if entry_classes[0] is None:
                if frame_level >= 0:
                    invalids += 1
                    continue
                else:
                    break

            # It's an empty page (data page or page table)
            if entry_classes[0] is False:
                empty_entries += 1
                continue

            # Add all records if scanning the memory or only the record of a particular level if it is creating a table to be print
            if frame_level < 0:
                for entry_obj in entry_classes:
                    entry_type = type(entry_obj)
                    frame_d[entry_type][entry_idx] = entry_obj
            else:
                for entry_obj in entry_classes:
                    entry_type = type(entry_obj)
                    if (
                        type(entry_obj) in data_classes
                        or type(entry_obj) is ptr_class
                        or type(entry_obj) in reseved_classes
                    ):
                        frame_d[entry_type][entry_idx] = entry_obj
                        break
                else:
                    invalids += 1

        # Classify the frame
        pt_classes = self.classify_frame(
            frame_d,
            empty_entries,
            int(frame_size // self.page_table_class.entry_size),
            mode=mode,
        )

        if -1 in pt_classes or -2 in pt_classes:  # EMPTY or DATA
            table_obj = None
        else:
            table_obj = self.page_table_class(
                frame_addr, frame_size, frame_d, pt_classes
            )
        return invalids, pt_classes, table_obj

    def classify_frame(
        self,
        frame_d,
        empty_c,
        entries_per_frame,
        mode="global",
    ):
        if empty_c == entries_per_frame:
            return [-1]  # EMPTY

        # For each level check if a table is a valid candidate
        frame_classes = []
        for level in range(self.radix_levels[mode]):
            entries = empty_c
            if self.map_ptr_entries_to_levels[mode][level] is not None:
                entries += len(frame_d[self.map_ptr_entries_to_levels[mode][level]])
            for data_class in self.map_datapages_entries_to_levels[mode][level]:
                if data_class is not None:
                    entries += len(frame_d[data_class])
            for reserved_class in self.map_reserved_entries_to_levels[mode][level]:
                entries += len(frame_d[reserved_class])
            if entries == entries_per_frame:
                frame_classes.append(level)

        if not frame_classes:
            return [-2]  # DATA
        else:
            return frame_classes


class PhysicalMemory:
    machine = None

    def __deepcopy__(self, memo):
        return PhysicalMemory(self.raw_configuration)

    def __copy__(self):
        return PhysicalMemory(self.raw_configuration)

    def __getstate__(self):
        self.close()
        if self._miasm_vm:
            del self._miasm_vm
            self._miasm_vm = None
        state = deepcopy(self.__dict__)
        self.reopen()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reopen()

    def __init__(self, regions_defs):
        self._is_opened = False
        self._miasm_vm = None
        self._memregions = []
        self._memsize = 0
        self.physpace = {"ram": portion.empty(), "not_ram": portion.empty()}
        self.raw_configuration = regions_defs

        # Load dump RAM files
        try:
            for region_def in regions_defs["ram"]:
                # Load the dump file for a memory region
                fd = open(region_def["dumpfile"], "rb")
                mm = mmap(fd.fileno(), 0, MAP_SHARED, PROT_READ)
                mm.madvise(MADV_HUGEPAGE)
                self._memsize += len(mm)
                region_size = len(mm)

                if region_size != region_def["end"] - region_def["start"] + 1:
                    raise IOError(
                        "Declared size {} is different from real size {} for: {}".format(
                            region_def["end"] - region_def["start"] + 1,
                            region_size,
                            region_def["dumpfile"],
                        )
                    )

                self._memregions.append(
                    {
                        "filename": region_def["dumpfile"],
                        "fd": fd,
                        "mmap": mm,
                        "size": region_size,
                        "start": region_def["start"],
                        "end": region_def["end"],
                    }
                )
                self.physpace["ram"] |= portion.closed(
                    region_def["start"], region_def["end"]
                )

            self._memregions.sort(key=lambda x: x["start"])
            self._is_opened = True

        except Exception as e:
            self.close()
            raise IOError("Error in opening dump files! Error: {}".format(e))

        # Load not RAM regions
        for region_def in regions_defs.get("not_ram", []):
            self.physpace["not_ram"] |= portion.closed(
                region_def["start"], region_def["end"]
            )
        self.physpace["not_valid_regions"] = self.physpace["not_ram"].difference(
            self.physpace["ram"]
        )
        self.physpace["defined_regions"] = (
            self.physpace["not_ram"] | self.physpace["ram"]
        )

    def __del__(self):
        self.close()

    def __len__(self):
        return self._memsize

    def __contains__(self, key):
        if not isinstance(key, int):
            raise TypeError
        return key in self.physpace["ram"]

    def close(self):
        for region in self._memregions:
            try:
                if region["mmap"] is not None:
                    region["mmap"].close()
                    del region["mmap"]
                    region["mmap"] = None
                if region["fd"] is not None:
                    region["fd"].close()
                    del region["fd"]
                    region["fd"] = None
            except Exception:
                continue
        self._is_opened = False

    def reopen(self):
        for region in self._memregions:
            region["fd"] = open(region["filename"], "rb")
            region["mmap"] = mmap(region["fd"].fileno(), 0, MAP_SHARED, PROT_READ)
            region["mmap"].madvise(MADV_HUGEPAGE)
        self._is_opened = True

    def get_data(self, start, size):
        for region in self._memregions:
            if region["start"] <= start <= region["end"]:
                return region["mmap"][
                    start - region["start"] : start - region["start"] + size
                ]
        return bytearray()

    def get_addresses(self, size, align_offset=0, cpus=1, max_address=-1):
        """Return a list contains tuples (for a total of cpus tuples). Each tuple contains the len of the iterator, and an iterator over part of all the addresses aligned to align_offset and distanced by size present in RAM"""
        if size == 0:
            return 0, []

        ranges = []
        multi_ranges = [[] for cid in range(cpus)]
        for region in self._memregions:
            region_start = region["start"]
            region_end = region["end"]

            if region_start >= max_address > 0:
                continue

            if align_offset > region["size"]:
                continue

            if 0 < max_address < region_end:
                chunk_end = max_address
            else:
                chunk_end = region_end

            original_r = range(region_start + align_offset, chunk_end + 1, size)

            if cpus == 1:
                ranges.append(original_r)
            else:
                # Split the iterator over the region in cpus iterators
                range_size = len(original_r) // cpus * size
                if range_size <= size:
                    range_size = size
                    vcpus = len(original_r) // size
                else:
                    vcpus = cpus

                first_elem = region_start + align_offset
                multi_ranges[0].append(
                    range(first_elem, region_start + range_size, size)
                )

                prev_last = region_start + range_size
                for i in range(1, vcpus - 1):
                    r = (prev_last - align_offset - region_start) % size
                    if r == 0:
                        first_elem = prev_last
                    else:
                        first_elem = prev_last + (size - r)

                    last_elem = (i + 1) * range_size + region_start
                    multi_ranges[i].append(range(first_elem, last_elem, size))

                    prev_last = last_elem

                r = (prev_last - align_offset - region_start) % size
                if r == 0:
                    first_elem = prev_last
                else:
                    first_elem = prev_last + (size - r)

                if first_elem > chunk_end:
                    continue
                multi_ranges[vcpus - 1].append(range(first_elem, chunk_end + 1, size))

        if cpus == 1:
            total_elems = sum([len(x) for x in ranges])
            return total_elems, [total_elems, chain(*ranges)]
        else:
            total_elems = 0
            for cid in range(cpus):
                elems_in_iter = sum([len(x) for x in multi_ranges[cid]])
                total_elems += elems_in_iter
                multi_ranges[cid] = (elems_in_iter, chain(*multi_ranges[cid]))
            return total_elems, multi_ranges

    def get_miasm_vmmngr(self):
        """Load each RAM file in a MIASM virtual memory region"""
        if self._miasm_vm is not None:
            return self._miasm_vm

        vm = Vm()
        for region_def in self._memregions:
            vm.add_memory_page(
                region_def["start"],
                PAGE_READ | PAGE_WRITE | PAGE_EXEC,
                region_def["fd"].read(),
                region_def["filename"],
            )
            region_def["fd"].seek(0)
        self._miasm_vm = vm
        return self._miasm_vm

    def get_memregions(self):
        return self._memregions

    def free_miasm_memory(self):
        if self._miasm_vm:
            self._miasm_vm = None
            gc.collect()


class MMUShell(Cmd):
    intro = "MMUShell.   Type help or ? to list commands.\n"

    def __init__(self, completekey="tab", stdin=None, stdout=None, machine={}):
        super(MMUShell, self).__init__(completekey, stdin, stdout)
        self.machine = machine
        self.prompt = "[MMUShell " + self.machine.cpu.architecture + "]# "
        self.data = {}
        self.gtruth = {}

    def reload_data_from_file(self, data_filename):
        # Load a previous session
        logger.info("Loading previous session data...")
        try:
            self.data = load(data_filename, compression="lzma")
        except Exception as e:
            logger.fatal("Fatal error loading session data! Error:{}".format(e))
            import traceback

            print(traceback.print_exc())
            exit(1)

    def load_gtruth(self, gtruth_fd):
        try:
            self.gtruth = Load(gtruth_fd)
            gtruth_fd.close()
        except Exception as e:
            logger.fatal("Fatal error loading ground truth file! Error:{}".format(e))
            exit(1)

    def do_exit(self, arg):
        """Exit :)"""
        logger.info("Bye! :)")
        return True

    def do_save_data(self, arg):
        """Save data in a compressed pickle file"""

        if not len(arg):
            logger.info("Use: save_data FILENAME")
            return

        try:
            logger.info("Saving data...")
            dump(self.data, arg, compression="lzma")
        except Exception as e:
            logger.error("Error in session data. Error: {}".format(e))
            return

    # def do_show_machine_config(self, arg):
    #     """Show the machine configuration"""
    #     pprint(self.machine_config)

    def do_ipython(self, arg):
        """Open an IPython shell"""
        embed()

    def emptyline(self):
        pass

    def parse_int(self, value):
        if any([c not in "0123456789abcdefxABCDEFX" for c in value]):
            raise ValueError
        if value[:2].lower() == "0x":
            return int(value, 16)
        else:
            return int(value, 10)

    def radix_roots_from_data_page(
        self, pg_lvl, pg_addr, rev_map_pages, rev_map_tables
    ):
        # For a page address pointed by tables of level 'level' find all the radix root of trees containing it

        level_tables = set()
        prev_level_tables = set()

        # Collect all table at level 'pg_lvl' which point to that page
        level_tables.update(rev_map_pages[pg_lvl][pg_addr])
        logger.debug(
            "radix_roots_from_data_pages: level_tables found {} for pg_addr {}".format(
                len(level_tables), hex(pg_addr)
            )
        )

        # Raise the tree in order to find the top table
        for tree_lvl in range(pg_lvl - 1, -1, -1):
            for table_addr in level_tables:
                prev_level_tables.update(rev_map_tables[tree_lvl][table_addr])

            level_tables = prev_level_tables
            prev_level_tables = set()
            logger.debug(
                "radix_roots_from_data_pages: level_tables found {} for pg_addr {}".format(
                    len(level_tables), hex(pg_addr)
                )
            )

        return set(level_tables)

    def physpace(
        self,
        addr,
        page_tables,
        empty_tables,
        lvl=0,
        uperms=(True,) * 6,
        hierarchical=False,
        mode="global",
        cache=defaultdict(dict),
    ):
        """Recursively evaluate the consistency and return the kernel/user physical space addressed"""
        pas = PAS()
        data_classes = self.machine.mmu.map_datapages_entries_to_levels[mode][lvl]
        logging.debug(f"physpace() radix: {hex(addr)} Lvl: {lvl} Table: {hex(addr)}")

        # Leaf level
        if lvl == self.machine.mmu.radix_levels[mode] - 1:
            for data_class in data_classes:
                for entry in page_tables[lvl][addr].entries[data_class].values():
                    perms = entry.get_permissions()
                    pas.space[perms][entry.address] = entry.size
                    pas.space_size[perms] += entry.size

            cache[lvl][addr] = (True, pas)
            return True, pas

        else:  # Superior levels
            ptr_class = self.machine.mmu.map_ptr_entries_to_levels[mode][lvl]
            if ptr_class in page_tables[lvl][addr].entries:
                for entry in page_tables[lvl][addr].entries[ptr_class].values():
                    if entry.address not in page_tables[lvl + 1]:
                        if (
                            entry.address not in empty_tables
                        ):  # It is not an empty table!
                            logging.debug(
                                f"physpace() radix: {hex(addr)} parent level: {lvl} table: {hex(entry.address)} invalid"
                            )
                            cache[lvl][addr] = (False, None)
                            return False, None
                    else:
                        if entry.address not in cache[lvl + 1]:
                            low_cons, low_pas = self.physpace(
                                entry.address,
                                page_tables,
                                empty_tables,
                                lvl + 1,
                                uperms=uperms,
                                hierarchical=hierarchical,
                                mode=mode,
                                cache=cache,
                            )
                        else:
                            low_cons, low_pas = cache[lvl + 1][entry.address]

                        if not low_cons:
                            logging.debug(
                                f"physpace() radix: {hex(addr)} parent level: {lvl} table: {hex(entry.address)} invalid"
                            )
                            cache[lvl][addr] = (False, None)
                            return False, None

                        if hierarchical:
                            pas.hierarchical_extend(low_pas, uperms)
                        else:
                            pas.hierarchical_extend(low_pas, (True,) * 6)

            for data_class in data_classes:
                if (
                    data_class in page_tables[lvl][addr].entries
                    and data_class is not None
                ):
                    for entry in page_tables[lvl][addr].entries[data_class].values():
                        perms = entry.get_permissions()
                        pas.space[perms][entry.address] = entry.size
                        pas.space_size[perms] += entry.size

            cache[lvl][addr] = (True, pas)
            return True, pas

    def resolve_vaddr(self, cr3, vaddr, mode="global"):
        # Return the paddr or -1

        # Split in possible table resolution paths
        for splitted_addr in self.machine.mmu.split_vaddr(vaddr):
            current_table_addr = cr3
            requested_steps = len(splitted_addr) - 1
            resolution_steps = 0

            for level_idx, idx_t in enumerate(splitted_addr[:-1]):
                level_class, entry_idx = idx_t
                # Missing valid table, no valid resolution path
                if current_table_addr not in self.data.page_tables["global"][level_idx]:
                    logging.debug(
                        f"resolve_vaddr() Missing table {hex(current_table_addr)}"
                    )
                    logging.debug("resolve_vaddr() RESOLUTION PATH FAILED! ########")
                    break

                logging.debug(
                    f"resolve_vaddr(): Resolution path Lvl: {level_class} Table: {hex(current_table_addr)} Entry addr: {hex( current_table_addr + self.machine.mmu.page_table_class.entry_size * entry_idx)}"
                )
                # Find valid entry in table
                if (
                    entry_idx
                    in self.data.page_tables["global"][level_idx][
                        current_table_addr
                    ].entries[level_class]
                ):
                    current_table_addr = (
                        self.data.page_tables["global"][level_idx][current_table_addr]
                        .entries[level_class][entry_idx]
                        .address
                    )

                    resolution_steps += 1

                    # Each resolution path involves different number of steps (table to walk)
                    if resolution_steps == requested_steps:
                        return current_table_addr + splitted_addr[-1][1]
                    else:
                        continue
                else:
                    logging.debug("resolve_vaddr() RESOLUTION PATH FAILED! ########")
                    break
        else:
            return -1

    def virtspace(
        self,
        addr,
        lvl=0,
        prefix=0,
        uperms=(True,) * 6,
        hierarchical=False,
        mode="global",
        cache=defaultdict(dict),
    ):
        """Recursively reconstruct virtual address space"""

        virtspace = VAS()
        data_classes = self.machine.mmu.map_datapages_entries_to_levels[mode][lvl]
        ptr_class = self.machine.mmu.map_ptr_entries_to_levels[mode][lvl]
        cache[lvl][addr] = defaultdict(portion.empty)

        if lvl == self.machine.mmu.radix_levels[mode] - 1:
            for data_class in data_classes:
                shift = self.machine.mmu.map_entries_to_shifts[mode][data_class]
                for entry_idx, entry in (
                    self.data.page_tables[mode][lvl][addr].entries[data_class].items()
                ):
                    permissions = entry.get_permissions()
                    virt_addr = prefix | (entry_idx << shift)
                    virtspace[permissions] |= portion.closedopen(
                        virt_addr, virt_addr + entry.size
                    )

            cache[lvl][addr] = virtspace
            return virtspace

        else:
            if ptr_class in self.data.page_tables[mode][lvl][addr].entries:
                shift = self.machine.mmu.map_entries_to_shifts[mode][ptr_class]
                for entry_idx, entry in (
                    self.data.page_tables[mode][lvl][addr].entries[ptr_class].items()
                ):
                    if entry.address not in self.data.page_tables[mode][lvl + 1]:
                        continue
                    else:
                        permissions = entry.get_permissions()

                        if entry.address not in cache[lvl + 1]:
                            virt_addr = prefix | (entry_idx << shift)
                            low_virts = self.virtspace(
                                entry.address,
                                lvl + 1,
                                virt_addr,
                                permissions,
                                hierarchical=hierarchical,
                                mode=mode,
                                cache=cache,
                            )
                        else:
                            low_virts = cache[lvl + 1][entry.address]

                        if hierarchical:
                            virtspace.hierarchical_extend(low_virts, uperms)
                        else:
                            virtspace.hierarchical_extend(low_virts, (True,) * 6)

            for data_class in data_classes:
                if (
                    data_class in self.data.page_tables[mode][lvl][addr].entries
                    and data_class is not None
                ):
                    shift = self.machine.mmu.map_entries_to_shifts[mode][data_class]
                    for entry_idx, entry in (
                        self.data.page_tables[mode][lvl][addr]
                        .entries[data_class]
                        .items()
                    ):
                        permissions = entry.get_permissions()
                        virt_addr = prefix | (entry_idx << shift)
                        virtspace[permissions] |= portion.closedopen(
                            virt_addr, virt_addr + entry.size
                        )

            cache[lvl][addr] = virtspace
            return virtspace
