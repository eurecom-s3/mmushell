from architectures.generic import Machine as MachineDefault
from architectures.generic import CPU as CPUDefault
from architectures.generic import PhysicalMemory as PhysicalMemoryDefault
from architectures.generic import MMUShell as MMUShellDefault
from architectures.generic import TableEntry, PageTable, MMURadix, PAS, RadixTree
import logging
from collections import defaultdict, deque
from prettytable import PrettyTable
from time import sleep
from tqdm import tqdm
from copy import deepcopy, copy
from random import uniform
from struct import iter_unpack, unpack
from dataclasses import dataclass
import multiprocessing as mp

# import cProfile
import portion
from more_itertools import divide
from IPython import embed

logger = logging.getLogger(__name__)


@dataclass
class Data:
    is_mem_parsed: bool
    is_radix_found: bool
    page_tables: dict
    data_pages: list
    empty_tables: list
    reverse_map_tables: list
    reverse_map_pages: list
    satps: dict


class SATP:
    def __init__(self, satp):
        self.satp = satp
        self.mode = MMU.extract_bits(satp, 31, 1)
        self.asid = MMU.extract_bits(satp, 22, 9)
        self.address = MMU.extract_bits(satp, 0, 22) << 12

    def __repr__(self):
        print(f"Mode:{self.mode}, ASID:{self.asid}, Address: {hex(self.address)}")


class Machine(MachineDefault):
    def __init__(self, cpu, mmu, memory, **kwargs):
        super(Machine, self).__init__(cpu, mmu, memory, **kwargs)


class PhysicalMemory(PhysicalMemoryDefault):
    pass


class CPU(CPUDefault):
    @classmethod
    def from_cpu_config(cls, cpu_config, **kwargs):
        if cpu_config["bits"] == 32:
            return CPU32(cpu_config)
        else:
            return CPU64(cpu_config)

    def __init__(self, features):
        super(CPU, self).__init__(features)
        CPU.endianness = self.endianness
        CPU.extract_bits = CPU.extract_bits_little


class CPU32(CPU):
    pass


class CPU64(CPU):
    pass


#####################################################################
# 32 bit entries and page table
#####################################################################


class TEntry32(TableEntry):
    entry_size = 4
    entry_name = "TEntry32"
    size = 0
    labels = [
        "Address:",
        "Dirty:",
        "Accessed:",
        "Global:",
        "User:",
        "Readable:",
        "Writable:",
        "Exectuable:",
    ]
    addr_fmt = "0x{:08x}"

    def __hash__(self):
        return hash(self.entry_name)

    def __repr__(self):
        e_resume = self.entry_resume_stringified()
        return str(
            [self.labels[i] + " " + str(e_resume[i]) for i in range(len(self.labels))]
        )

    def entry_resume(self):
        return [
            self.address,
            self.is_dirty_entry(),
            self.is_accessed_entry(),
            self.is_global_entry(),
            not self.is_supervisor_entry(),
            self.is_readable_entry(),
            self.is_writeble_entry(),
            self.is_executable_entry(),
        ]

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[0] = self.addr_fmt.format(res[0])
        for idx, r in enumerate(res[1:], start=1):
            res[idx] = str(r)
        return res

    def is_dirty_entry(self):
        return bool(MMU.extract_bits(self.flags, 7, 1))

    def is_accessed_entry(self):
        return bool(MMU.extract_bits(self.flags, 6, 1))

    def is_global_entry(self):
        return bool(MMU.extract_bits(self.flags, 5, 1))

    def is_supervisor_entry(self):
        return not MMU.extract_bits(self.flags, 4, 1)

    def is_readable_entry(self):
        return bool(MMU.extract_bits(self.flags, 1, 1))

    def is_writeble_entry(self):
        return bool(MMU.extract_bits(self.flags, 2, 1))

    def is_executable_entry(self):
        return bool(MMU.extract_bits(self.flags, 3, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 10, 21) << 12

    def get_permissions(self):
        perms = (
            self.is_readable_entry(),
            self.is_writeble_entry(),
            self.is_executable_entry(),
        )
        if self.is_supervisor_entry():
            return perms + (False, False, False)
        else:
            return (False, False, False) + perms


class PTE4KB32(TEntry32):
    entry_name = "PTE4KB32"
    size = 1024 * 4


class PTE4MB(TEntry32):
    entry_name = "PTE4MB"
    size = 1024 * 1024 * 4


class PTP32(TEntry32):
    entry_name = "PTP32"
    size = 0


class PTP32L0(PTP32):
    entry_name = "PTP32L0"


class PageTableSV32(PageTable):
    entry_size = 4
    table_fields = [
        "Entry address",
        "Pointed address",
        "Dirty",
        "Accessed",
        "Global",
        "User",
        "Readable",
        "Writable",
        "Exectuable",
        "Class",
    ]
    addr_fmt = "0x{:08x}"

    def __repr__(self):
        table = PrettyTable()
        table.field_names = self.table_fields

        for entry_class in self.entries:
            for entry_idx, entry_obj in self.entries[entry_class].items():
                entry_addr = self.address + (entry_idx * self.entry_size)
                table.add_row(
                    [self.addr_fmt.format(entry_addr)]
                    + entry_obj.entry_resume_stringified()
                    + [entry_class.entry_name]
                )

        table.sortby = "Entry address"
        return str(table)


#####################################################################
# 64 bit entries and page table
#####################################################################


class TEntry64(TEntry32):
    entry_size = 8
    entry_name = "TEntry64"
    size = 0
    addr_fmt = "0x{:016x}"

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 10, 44) << 12


class PTE4KB64(TEntry64):
    entry_name = "PTE4KB64"
    size = 1024 * 4


class PTE2MB(TEntry64):
    entry_name = "PTE2MB"
    size = 1024 * 1024 * 2


class PTE1GB(TEntry64):
    entry_name = "PTE1GB"
    size = 1024 * 1024 * 1024


class PTE512GB(TEntry64):
    entry_name = "PTE512GB"
    size = 1024 * 1024 * 1024 * 512


class PTP64(TEntry64):
    entry_name = "PTP64"
    size = 0


class PTP64L0(PTP64):
    entry_name = "PTP64L0"


class PTP64L1(PTP64):
    entry_name = "PTP64L1"


class PTP64L2(PTP64):
    entry_name = "PTP64L2"


class PageTableSV39(PageTableSV32):
    entry_size = 8
    addr_fmt = "0x{:016x}"


class PageTableSV48(PageTableSV32):
    entry_size = 8
    addr_fmt = "0x{:016x}"


#################################################################
# MMU Modes
#################################################################


class MMU(MMURadix):
    PAGE_SIZE = 4096
    extract_bits = MMURadix.extract_bits_little
    paging_unpack_format = "<I"
    page_table_class = PageTableSV32
    radix_levels = {"global": 2}
    top_prefix = 0
    entries_size = 4


class SV32(MMU):
    paging_unpack_format = "<I"
    page_table_class = PageTableSV32
    radix_levels = {"global": 2}
    top_prefix = 0x0
    map_ptr_entries_to_levels = {"global": [PTP32L0, None]}
    map_datapages_entries_to_levels = {"global": [[PTE4MB], [PTE4KB32]]}
    map_level_to_table_size = {"global": [4096, 4096]}
    map_entries_to_shifts = {"global": {PTP32L0: 22, PTE4MB: 22, PTE4KB32: 12}}
    map_reserved_entries_to_levels = {"global": [[], []]}

    def return_not_leaf_entry(self, entry):
        addr = PTP32L0.extract_addr(entry)
        if addr in self.machine.memory.physpace["not_valid_regions"]:
            return [None]
        else:
            return [PTP32L0(addr, MMU.extract_bits(entry, 0, 10))]

    def return_leaf_entry(self, entry):
        addr = PTE4KB32.extract_addr(entry)
        flags = MMU.extract_bits(entry, 0, 10)
        ret = [PTE4KB32(addr, flags)]

        # Check if it could be a HUGE page
        if not MMU.extract_bits(entry, 10, 10):
            ret.append(PTE4MB(addr, flags))
        return ret

    def classify_entry(self, page_addr, entry):
        # If BIT V=0 is EMPTY
        if not MMU.extract_bits(entry, 0, 1):
            return [False]

        # If R,W,X are 0 it can be a non-leaf entry
        if not MMU.extract_bits(entry, 1, 3):
            # Pointer entry
            # D,A,U, must be 0
            if MMU.extract_bits(entry, 4, 1) or MMU.extract_bits(entry, 6, 2):
                return [None]
            else:
                return self.return_not_leaf_entry(entry)

        else:
            # #######################
            # Rules to much restrictive
            # W=1 and R=0 combinations are reserved
            # D=1 => A=1
            # #######################
            return self.return_leaf_entry(entry)

    def extend_prefix(self, prefix, entry_idx, entry_class):
        if entry_class in [PTP32L0, PTE4MB]:
            prefix = entry_idx << 22
            return prefix
        else:
            return prefix | (entry_idx << 12)

    def split_vaddr(self, vaddr):
        return (
            (
                (PTP32L0, MMU.extract_bits(vaddr, 22, 10)),
                (PTE4KB32, MMU.extract_bits(vaddr, 12, 10)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 12)),
            ),
            (
                (PTE4MB, MMU.extract_bits(vaddr, 22, 10)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 22)),
            ),
        )


class SV39(SV32):
    paging_unpack_format = "<Q"
    page_table_class = PageTableSV39
    radix_levels = {"global": 3}
    top_prefix = 0x0
    map_ptr_entries_to_levels = {"global": [PTP64L0, PTP64L1, None]}
    map_datapages_entries_to_levels = {"global": [[PTE1GB], [PTE2MB], [PTE4KB64]]}
    map_level_to_table_size = {"global": [4096, 4096, 4096]}
    map_entries_to_shifts = {
        "global": {PTP64L0: 30, PTE1GB: 30, PTP64L1: 21, PTE2MB: 21, PTE4KB64: 12}
    }
    map_reserved_entries_to_levels = {"global": [[], [], []]}

    def return_not_leaf_entry(self, entry):
        addr = PTP64.extract_addr(entry)
        if addr in self.machine.memory.physpace["not_valid_regions"]:
            return [None]
        else:
            flags = MMU.extract_bits(entry, 0, 10)
            return [PTP64L0(addr, flags), PTP64L1(addr, flags)]

    def return_leaf_entry(self, entry):
        addr = PTE4KB64.extract_addr(entry)
        flags = MMU.extract_bits(entry, 0, 10)
        ret = [PTE4KB64(addr, flags)]

        # Check if it could be a HUGE page
        if not MMU.extract_bits(entry, 10, 9):
            ret.append(PTE2MB(addr, flags))
        if not MMU.extract_bits(entry, 10, 18):
            ret.append(PTE1GB(addr, flags))
        return ret

    def classify_entry(self, page_addr, entry):
        # On RISCV64 some top bits of PTE are reserved for future flags
        if MMU.extract_bits(entry, 0, 1) and MMU.extract_bits(entry, 54, 10):
            return [None]
        else:
            return super(SV39, self).classify_entry(page_addr, entry)

    def extend_prefix(self, prefix, entry_idx, entry_class):
        if entry_class in [PTP64L0, PTE1GB]:
            return entry_idx << 30

        elif entry_class in [PTP64L1, PTE2MB]:
            prefix = entry_idx << 21
            return prefix

        else:
            return prefix | (entry_idx << 12)

    def split_vaddr(self, vaddr):
        return (
            (
                (PTP64L0, MMU.extract_bits(vaddr, 30, 26)),
                (PTP64L1, MMU.extract_bits(vaddr, 21, 9)),
                (PTE4KB64, MMU.extract_bits(vaddr, 12, 9)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 12)),
            ),
            (
                (PTP64L0, MMU.extract_bits(vaddr, 30, 26)),
                (PTE2MB, MMU.extract_bits(vaddr, 21, 9)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 21)),
            ),
            (
                (PTE1GB, MMU.extract_bits(vaddr, 30, 26)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 30)),
            ),
        )


class SV48(SV39):
    paging_unpack_format = "<Q"
    page_table_class = PageTableSV39
    radix_levels = {"global": 4}
    top_prefix = 0x0
    map_ptr_entries_to_levels = {"global": [PTP64L0, PTP64L1, PTP64L2, None]}
    map_datapages_entries_to_levels = {
        "global": [[PTE512GB], [PTE1GB], [PTE2MB], [PTE4KB64]]
    }
    map_level_to_table_size = {"global": [4096, 4096, 4096, 4096]}
    map_entries_to_shifts = {
        "global": {
            PTP64L0: 39,
            PTE512GB: 39,
            PTP64L1: 30,
            PTE1GB: 30,
            PTP64L2: 21,
            PTE2MB: 21,
            PTE4KB64: 12,
        }
    }
    map_reserved_entries_to_levels = {"global": [[], [], []]}

    def return_not_leaf_entry(self, entry):
        addr = PTP64.extract_addr(entry)
        if addr in self.machine.memory.physpace["not_valid_regions"]:
            return [None]
        else:
            flags = MMU.extract_bits(entry, 0, 10)
            return [PTP64L0(addr, flags), PTP64L1(addr, flags)]

    def return_leaf_entry(self, entry):
        addr = PTE4KB64.extract_addr(entry)
        flags = MMU.extract_bits(entry, 0, 10)
        ret = [PTE4KB64(addr, flags)]

        # Check if it could be a HUGE page
        if not MMU.extract_bits(entry, 10, 9):
            ret.append(PTE2MB(addr, flags))
        if not MMU.extract_bits(entry, 10, 18):
            ret.append(PTE1GB(addr, flags))
        if not MMU.extract_bits(entry, 10, 27):
            ret.append(PTE512GB(addr, flags))
        return ret

    def classify_entry(self, page_addr, entry):
        # On RISCV64 some top bits of PTE are reserved for future flags
        if MMU.extract_bits(entry, 0, 1) and MMU.extract_bits(entry, 54, 10):
            return [None]
        else:
            return super(SV48, self).classify_entry(page_addr, entry)

    def extend_prefix(self, prefix, entry_idx, entry_class):
        if entry_class in [PTP64L0, PTE512GB]:
            return entry_idx << 39

        elif entry_class in [PTP64L1, PTE1GB]:
            return entry_idx << 30

        elif entry_class in [PTP64L2, PTE2MB]:
            prefix = entry_idx << 21
            return prefix

        else:
            return prefix | (entry_idx << 12)

    def split_vaddr(self, vaddr):
        return (
            (
                (PTP64L0, MMU.extract_bits(vaddr, 39, 17)),
                (PTP64L1, MMU.extract_bits(vaddr, 30, 9)),
                (PTP64L2, MMU.extract_bits(vaddr, 21, 9)),
                (PTE4KB64, MMU.extract_bits(vaddr, 12, 9)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 12)),
            ),
            (
                (
                    PTP64L0,
                    MMU.extract_bits(vaddr, 39, 17),
                    (PTP64L1, MMU.extract_bits(vaddr, 30, 26)),
                    (PTE2MB, MMU.extract_bits(vaddr, 21, 9)),
                    ("OFFSET", MMU.extract_bits(vaddr, 0, 21)),
                )
            ),
            (
                (PTP64L0, MMU.extract_bits(vaddr, 39, 17)),
                (PTE1GB, MMU.extract_bits(vaddr, 30, 26)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 30)),
            ),
            (
                (PTE512GB, MMU.extract_bits(vaddr, 39, 17)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 39)),
            ),
        )


class MMUShell(MMUShellDefault):
    def __init__(self, completekey="tab", stdin=None, stdout=None, machine={}):
        super(MMUShell, self).__init__(completekey, stdin, stdout, machine)

        if not self.data:
            self.data = Data(
                is_mem_parsed=False,
                is_radix_found=False,
                page_tables={
                    "global": [
                        {} for i in range(self.machine.mmu.radix_levels["global"])
                    ]
                },
                data_pages=[],
                empty_tables=[],
                reverse_map_tables=[
                    defaultdict(set)
                    for i in range(self.machine.mmu.radix_levels["global"])
                ],
                reverse_map_pages=[
                    defaultdict(set)
                    for i in range(self.machine.mmu.radix_levels["global"])
                ],
                satps={},
            )

    def do_parse_memory(self, args):
        """Parse memory to find tables"""
        if self.data.is_mem_parsed:
            logger.warning("Memory already parsed")
            return
        self.parse_memory()
        self.data.is_mem_parsed = True

    def do_show_table(self, args):
        """Show MMU table. Usage: show_table ADDRESS [level]"""
        args = args.split()
        if len(args) < 1:
            logger.warning("Missing table address")
            return

        try:
            addr = self.parse_int(args[0])
        except ValueError:
            logger.warning("Invalid table address")
            return

        if addr not in self.machine.memory:
            logger.warning("Table not in RAM range")
            return

        lvl = -1
        if len(args) > 1:
            try:
                lvl = self.parse_int(args[1])
                if lvl > self.machine.mmu.radix_levels["global"] - 1:
                    raise ValueError
            except ValueError:
                logger.warning(
                    "Level must be an integer between 0 and {}".format(
                        str(self.machine.mmu.radix_levels["global"] - 1)
                    )
                )
                return

        if lvl == -1:
            table_size = self.machine.mmu.PAGE_SIZE
        else:
            table_size = self.machine.mmu.map_level_to_table_size["global"][lvl]
        table_buff = self.machine.memory.get_data(addr, self.machine.mmu.PAGE_SIZE)
        invalids, pt_classes, table_obj = self.machine.mmu.parse_frame(
            table_buff, addr, table_size, lvl
        )
        print(table_obj)
        print(f"Invalid entries: {invalids} Table levels: {pt_classes}")

    def parse_memory(self):
        logger.info("Look for paging tables...")
        parallel_results = self.machine.apply_parallel(
            self.machine.mmu.PAGE_SIZE, self.machine.mmu.parse_parallel_frame
        )
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            page_tables, data_pages, empty_tables = result.get()

            for level in range(self.machine.mmu.radix_levels["global"]):
                self.data.page_tables["global"][level].update(page_tables[level])

            self.data.data_pages.extend(data_pages)
            self.data.empty_tables.extend(empty_tables)

        self.data.data_pages = set(self.data.data_pages)
        self.data.empty_tables = set(self.data.empty_tables)

        logger.info("Reduce false positives...")
        # Remove all tables which point to inexistent table of lower level
        for lvl in range(self.machine.mmu.radix_levels["global"] - 1):
            ptr_class = self.machine.mmu.map_ptr_entries_to_levels["global"][lvl]

            referenced_nxt = []
            for table_addr in list(self.data.page_tables["global"][lvl].keys()):
                for entry_obj in (
                    self.data.page_tables["global"][lvl][table_addr]
                    .entries[ptr_class]
                    .values()
                ):
                    if (
                        entry_obj.address
                        not in self.data.page_tables["global"][lvl + 1]
                        and entry_obj.address not in self.data.empty_tables
                    ):
                        # Remove the table
                        self.data.page_tables["global"][lvl].pop(table_addr)
                        break

                    else:
                        referenced_nxt.append(entry_obj.address)

            # Remove table not referenced by upper levels
            referenced_nxt = set(referenced_nxt)
            for table_addr in set(
                self.data.page_tables["global"][lvl + 1].keys()
            ).difference(referenced_nxt):
                self.data.page_tables["global"][lvl + 1].pop(table_addr)

        logger.info("Fill reverse maps...")
        for lvl in range(0, self.machine.mmu.radix_levels["global"]):
            ptr_class = self.machine.mmu.map_ptr_entries_to_levels["global"][lvl]
            page_class = self.machine.mmu.map_datapages_entries_to_levels["global"][
                lvl
            ][
                0
            ]  # Trick: it has only one dataclass per level
            for table_addr, table_obj in self.data.page_tables["global"][lvl].items():
                for entry_obj in table_obj.entries[ptr_class].values():
                    self.data.reverse_map_tables[lvl][entry_obj.address].add(
                        table_obj.address
                    )
                for entry_obj in table_obj.entries[page_class].values():
                    self.data.reverse_map_pages[lvl][entry_obj.address].add(
                        table_obj.address
                    )

    def do_find_radix_trees(self, args):
        """Reconstruct radix trees"""
        if not self.data.is_mem_parsed:
            logging.info("Please, parse the memory first!")
            return

        # Some table level was not found...
        if not len(self.data.page_tables["global"][0]):
            logger.warning("OOPS... no tables in first level... Wrong MMU mode?")
            return

        # Go back from PTLn up to top level, the particular form of PTLn permits to find PTL0
        logging.info("Go up the paging trees starting from data pages...")
        candidates = []
        already_explored = set()
        for page_addr in tqdm(self.data.data_pages):
            derived_addresses = self.machine.mmu.derive_page_address(page_addr)
            for derived_address in derived_addresses:
                if derived_address in already_explored:
                    continue
                lvl, addr = derived_address
                candidates.extend(
                    self.radix_roots_from_data_page(
                        lvl,
                        addr,
                        self.data.reverse_map_pages,
                        self.data.reverse_map_tables,
                    )
                )
                already_explored.add(derived_address)
        candidates = list(
            set(candidates).intersection(self.data.page_tables["global"][0].keys())
        )
        candidates.sort()

        logger.info("Filter candidates...")
        satps = {}
        for candidate in tqdm(candidates):
            # Obtain radix tree infos
            consistency, pas = self.physpace(
                candidate, self.data.page_tables["global"], self.data.empty_tables
            )

            # Only consistent trees are valid
            if not consistency:
                continue

            # Esclude empty trees
            if pas.get_kernel_size() == pas.get_user_size() == 0:
                continue

            vas = self.virtspace(candidate, 0, self.machine.mmu.top_prefix)
            satps[candidate] = RadixTree(candidate, 0, pas, vas)

        self.data.satps = satps
        self.data.is_radix_found = True

    def do_show_radix_trees(self, args):
        """Show found radix trees"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        labels = [
            "Radix address",
            "First level",
            "Kernel size (Bytes)",
            "User size (Bytes)",
        ]
        table = PrettyTable()
        table.field_names = labels
        for satp in self.data.satps.values():
            table.add_row(satp.entry_resume_stringified())
        table.sortby = "Radix address"
        print(table)


class MMUShellGTruth(MMUShell):
    def do_show_radix_trees_gtruth(self, args):
        """Compare found radix trees with the gound truth"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        # Parse TP SATPs
        satp_tp = {}
        for satp in self.gtruth["SATP"]:
            new_satp = SATP(satp)
            if new_satp.address in satp_tp:
                continue

            # Validate SATP
            if new_satp.address not in self.data.page_tables["global"][0]:
                continue

            consistency, pas = self.physpace(
                new_satp.address,
                self.data.page_tables["global"],
                self.data.empty_tables,
            )
            if not consistency or (
                not pas.get_kernel_size() and not pas.get_user_size()
            ):
                continue
            satp_tp[new_satp.address] = new_satp

        # True positives, false negatives, false positives
        tps = set(satp_tp.keys()).intersection(set(self.data.satps.keys()))
        fns = set(satp_tp.keys()).difference(set(self.data.satps.keys()))
        fps = set(self.data.satps.keys()).difference(set(satp_tp.keys()))

        # Show results
        table = PrettyTable()
        table.field_names = ["Address", "Found", "First seen", "Last seen"]
        for tp in sorted(tps):
            table.add_row(
                [
                    hex(tp),
                    "X",
                    self.gtruth["SATP"][satp_tp[tp].satp][0],
                    self.gtruth["SATP"][satp_tp[tp].satp][1],
                ]
            )

        for fn in sorted(fns):
            table.add_row(
                [
                    hex(fn),
                    "",
                    self.gtruth["SATP"][satp_tp[fn].satp][0],
                    self.gtruth["SATP"][satp_tp[fn].satp][1],
                ]
            )

        for fp in sorted(fps):
            table.add_row([hex(fp), "False positive", "", ""])

        print(table)
        print(f"TP:{len(tps)} FN:{len(fns)} FP:{len(fps)}")

        # Export results for next analysis
        if len(args):
            from pickle import dump as dump_p

            with open("dump.mmu", "wb") as f:
                results = [{"satp": tp} for tp in sorted(tps)]
                dump_p(results, f)
