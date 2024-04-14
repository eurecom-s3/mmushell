from architectures.generic import Machine as MachineDefault
from architectures.generic import CPU as CPUDefault
from architectures.generic import PhysicalMemory as PhysicalMemoryDefault
from architectures.generic import MMUShell as MMUShellDefault
from architectures.generic import TableEntry, PageTable, MMURadix, PAS, RadixTree
from architectures.generic import CPUReg, VAS
from architectures.generic import MMU as MMUDefault
import logging
from collections import defaultdict, deque
from miasm.analysis.machine import Machine as MIASMMachine
from miasm.core.bin_stream import bin_stream_vm
from miasm.core.locationdb import LocationDB
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

########################################################
# In ARM short mode PTL1 has always a fixed size (1KiB), PTL0 kernel tables has always fixed size (16KiB) while PTL0 user tables has a varible size depending on TTBCR.N (SIZE(PTL0_user) = 2**(14-N) 0 <= TTBCR.N <= 7). The minimum page size is 4KiB.

# Short mode has two permissions models AP[2:0] (the oldest) and AP[2:1] (selected by SCTRL.AFE=1) both non hierarchicals.
# AP[1] in both schemes manage the user accessibility to the page
#   AP[1] = 0 => No access in user mode
#   AP[1] = 1 => User have access, the permissions depends on AP[2] or AP[2,0]

# AP[2] is a "not writable" bit in both the modes
#   AP[2] = 0 => The page is writable, but depends on AP[1] or AP[0] from which mode
#   AP[2] = 1 => Writable permission disabled

# PXN is present also on PTP short entries and its hierarchical!
########################################################


class VASShort(VAS):
    def __repr__(self):
        s = ""
        for k in self:
            k_str = str(k)
            s += k_str + "\n"
            for interval in self[k]:
                s += f"\t[{hex(interval.lower)}, {hex(interval.upper)}]\n"
        return s


@dataclass
class Data:
    is_tables_found: bool
    is_registers_found: bool
    is_radix_found: bool
    opcodes: dict
    regs_values: dict
    page_tables: dict
    data_pages: list
    empty_tables: list
    reverse_map_tables: list
    reverse_map_pages: list
    used_ttbcr: None
    ttbrs: dict


class CPURegARM32(CPUReg):
    @classmethod
    def get_register_obj(cls, reg_name, value):
        return globals()[reg_name](value)


class SCTLR(CPURegARM32):
    def is_valid(self, value):
        if (
            CPU.extract_bits(value, 31, 1)
            or CPU.extract_bits(value, 26, 1)
            or CPU.extract_bits(value, 15, 1)
            or CPU.extract_bits(value, 8, 2)
            or not CPU.extract_bits(value, 23, 1)
            or not CPU.extract_bits(value, 18, 1)
            or not CPU.extract_bits(value, 16, 1)
            or not CPU.extract_bits(value, 6, 1)
            or CPU.extract_bits(value, 3, 2) != 3
        ):
            return False
        else:
            return True

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.m = CPU.extract_bits(value, 0, 1)
            self.ha = CPU.extract_bits(value, 17, 1)
            self.afe = CPU.extract_bits(value, 29, 1)
            self.tre = CPU.extract_bits(value, 28, 1)
            self.ee = CPU.extract_bits(value, 25, 1)

        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.m == other.m
            and self.afe == other.afe
            and self.tre == other.tre
            and self.ee == other.ee
        )

    def __repr__(self):
        return f"SCTLR {hex(self.value)} => HA:{hex(self.ha)}, AFE:{hex(self.afe)}, TRE:{hex(self.tre)}, EE:{hex(self.ee)} M:{hex(self.m)}"


class TTBCR(CPURegARM32):
    def is_valid(self, value):
        if CPU.extract_bits(value, 6, 25) or CPU.extract_bits(value, 3, 1):
            return False
        else:
            return True

    def count_fields_equals(self, other):
        tot = 0
        tot += 1 if self.n == other.n else 0
        tot += 1 if self.eae == other.eae else 0

        return tot

    def get_ptl0_user_table_size(self):
        return 1 << (14 - self.n)

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.n = CPU.extract_bits(value, 0, 3)
            self.pd0 = CPU.extract_bits(value, 4, 1)
            self.pd1 = CPU.extract_bits(value, 5, 1)
            self.eae = CPU.extract_bits(value, 31, 1)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.eae == other.eae and self.n == other.n

    def __repr__(self):
        mmu_mode = "Long" if self.eae else "Short"
        return f"TTBCR {hex(self.value)} => MMU mode: {mmu_mode}, EAE:{hex(self.eae)}, PD1:{hex(self.pd1)}, PD0:{hex(self.pd0)}, N:{hex(self.n)}"


class TTBR0(CPURegARM32):
    def is_valid(self, value):
        x = 14 - SHORT.ttbcr_n
        if x < 7:
            return not CPU.extract_bits(value, 7, x - 7)
        else:
            return True

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.irgn = (CPU.extract_bits(value, 1, 1) << 1) | CPU.extract_bits(
                value, 6, 1
            )
            self.s = CPU.extract_bits(value, 1, 1)
            self.imp = CPU.extract_bits(value, 2, 1)
            self.rgn = CPU.extract_bits(value, 3, 2)
            self.nos = CPU.extract_bits(value, 5, 1)

            x = 14 - SHORT.ttbcr_n
            self.address = CPU.extract_bits(value, x, 31 - x + 1) << x
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.address == other.address

    def __repr__(self):
        return f"TTBR0 {hex(self.value)} => Address:{hex(self.address)}, IRGN:{hex(self.irgn)}, S:{hex(self.s)}, IMP:{hex(self.imp)}, RGN:{hex(self.rgn)}, NOS:{hex(self.nos)}"


class TTBR1(CPURegARM32):
    def is_valid(self, value):
        return not CPU.extract_bits(value, 7, 7)

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.irgn = (CPU.extract_bits(value, 1, 1) << 1) | CPU.extract_bits(
                value, 6, 1
            )
            self.s = CPU.extract_bits(value, 1, 1)
            self.imp = CPU.extract_bits(value, 2, 1)
            self.rgn = CPU.extract_bits(value, 3, 2)
            self.nos = CPU.extract_bits(value, 5, 1)
            self.address = CPU.extract_bits(value, 14, 18) << 14
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.address == other.address

    def __repr__(self):
        return f"TTBR1 {hex(self.value)} => Address:{hex(self.address)}, IRGN:{hex(self.irgn)}, S:{hex(self.s)}, IMP:{hex(self.imp)}, RGN:{hex(self.rgn)}, NOS:{hex(self.nos)}"


#####################################################################
# 32 bit entries and page table
#####################################################################


class TEntry32(TableEntry):
    entry_size = 4
    entry_name = "TEntry32"
    size = 0
    labels = [
        "Address:",
        "TEX:",
        "Cacheble:",
        "Bufferable:",
        "Permsissions:" "Exec:",
        "Kernel exec:",
        "Secure:",
        "Domain:",
        "Shared:",
        "Global:",
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
            self.extract_tex(),
            self.is_cachable_entry(),
            self.is_bufferable_entry(),
            self.extract_permissions(),
            self.is_executable_entry(),
            self.is_kernel_executable_entry(),
            self.is_secure_entry(),
            self.extract_domain(),
            self.is_shared_entry(),
            self.is_global_entry(),
        ]

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[0] = self.addr_fmt.format(res[0])
        res[1] = self.addr_fmt.format(res[1])
        res[4] = self.addr_fmt.format(res[4])
        res[8] = self.addr_fmt.format(res[8])
        for idx, r in enumerate(res):
            res[idx] = str(r)
        return res

    def is_supervisor_entry(self):
        # We use AP[1] as indicator of kernel/user belonging
        return not bool(self.extract_permissions() & 0b10)

    def permissions_mode_21(self):
        perm = self.extract_permissions()
        if perm == 0:
            return True, True, False, False
        elif perm == 1:
            return True, True, True, True
        elif perm == 2:
            return True, False, False, False
        else:
            return True, False, True, False

    def get_permissions(self):
        kr, kw, r, w = self.permissions_mode_21()
        return (
            kr,
            kw,
            self.is_executable_entry(),
            r,
            w,
            self.is_kernel_executable_entry(),
        )


class PTP(TEntry32):
    entry_name = "PTP32"
    size = 0

    def extract_tex(self):
        return 0

    def is_cachable_entry(self):
        return "Ign."

    def is_bufferable_entry(self):
        return "Ign."

    def is_executable_entry(self):
        return True

    def extract_permissions(self):
        return 0

    def is_kernel_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 2, 1))

    def is_secure_entry(self):
        return not bool(MMU.extract_bits(self.flags, 3, 1))

    def extract_domain(self):
        return 0

    def is_shared_entry(self):
        return "Ign."

    def is_global_entry(self):
        return "Ign."

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 10, 22) << 10

    @staticmethod
    def extract_flags(entry):
        return MMU.extract_bits(entry, 0, 10)


class PTSECTION(TEntry32):
    entry_name = "PTSECTION"
    size = 1024 * 1024

    def extract_tex(self):
        return MMU.extract_bits(self.flags, 12, 2)

    def is_cachable_entry(self):
        return bool(MMU.extract_bits(self.flags, 3, 1))

    def is_bufferable_entry(self):
        return bool(MMU.extract_bits(self.flags, 2, 1))

    def extract_permissions(self):
        return (MMU.extract_bits(self.flags, 15, 1) << 2) | MMU.extract_bits(
            self.flags, 10, 2
        )

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 4, 1))

    def is_kernel_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 0, 1))

    def is_secure_entry(self):
        return not bool(MMU.extract_bits(self.flags, 19, 1))

    def extract_domain(self):
        return MMU.extract_bits(self.flags, 5, 4)

    def is_shared_entry(self):
        return bool(MMU.extract_bits(self.flags, 16, 1))

    def is_global_entry(self):
        return not bool(MMU.extract_bits(self.flags, 17, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 20, 12) << 20

    @staticmethod
    def extract_flags(entry):
        return MMU.extract_bits(entry, 0, 20)


class PTSUPERSECTION(TEntry32):
    entry_name = "PTSUPERSECTION"
    size = 1024 * 1024 * 16

    def extract_tex(self):
        return MMU.extract_bits(self.flags, 12, 2)

    def is_cachable_entry(self):
        return bool(MMU.extract_bits(self.flags, 3, 1))

    def is_bufferable_entry(self):
        return bool(MMU.extract_bits(self.flags, 2, 1))

    def extract_permissions(self):
        return (MMU.extract_bits(self.flags, 15, 1) << 2) | MMU.extract_bits(
            self.flags, 10, 2
        )

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 4, 1))

    def is_kernel_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 0, 1))

    def is_secure_entry(self):
        return not bool(MMU.extract_bits(self.flags, 19, 1))

    def extract_domain(self):
        return 0

    def is_shared_entry(self):
        return bool(MMU.extract_bits(self.flags, 16, 1))

    def is_global_entry(self):
        return not bool(MMU.extract_bits(self.flags, 17, 1))

    @staticmethod
    def extract_addr(entry):
        addr = MMU.extract_bits(entry, 24, 8) << 24
        addr = addr | (MMU.extract_bits(entry, 20, 4) << 32)
        addr = addr | (MMU.extract_bits(entry, 5, 4) << 36)
        return addr

    @staticmethod
    def extract_flags(entry):
        return MMU.extract_bits(entry, 0, 20)


class PTLARGE(TEntry32):
    entry_name = "PTLARGE"
    size = 1024 * 64

    def extract_tex(self):
        return MMU.extract_bits(self.flags, 12, 2)

    def is_cachable_entry(self):
        return bool(MMU.extract_bits(self.flags, 3, 1))

    def is_bufferable_entry(self):
        return bool(MMU.extract_bits(self.flags, 2, 1))

    def extract_permissions(self):
        return (MMU.extract_bits(self.flags, 9, 1) << 2) | MMU.extract_bits(
            self.flags, 4, 2
        )

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 15, 1))

    def is_kernel_executable_entry(self):
        return True

    def is_secure_entry(self):
        return False

    def extract_domain(self):
        return 0

    def is_shared_entry(self):
        return bool(MMU.extract_bits(self.flags, 10, 1))

    def is_global_entry(self):
        return not bool(MMU.extract_bits(self.flags, 11, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 16, 16) << 16

    @staticmethod
    def extract_flags(entry):
        return MMU.extract_bits(entry, 0, 16)


class PTSMALL(TEntry32):
    entry_name = "PTSMALL"
    size = 1024 * 4

    def extract_tex(self):
        return MMU.extract_bits(self.flags, 6, 2)

    def is_cachable_entry(self):
        return bool(MMU.extract_bits(self.flags, 3, 1))

    def is_bufferable_entry(self):
        return bool(MMU.extract_bits(self.flags, 2, 1))

    def extract_permissions(self):
        return (MMU.extract_bits(self.flags, 9, 1) << 2) | MMU.extract_bits(
            self.flags, 4, 2
        )

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.flags, 0, 1))

    def is_kernel_executable_entry(self):
        return True

    def is_secure_entry(self):
        return False

    def extract_domain(self):
        return 0

    def is_shared_entry(self):
        return bool(MMU.extract_bits(self.flags, 10, 1))

    def is_global_entry(self):
        return not bool(MMU.extract_bits(self.flags, 11, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 12, 20) << 12

    @staticmethod
    def extract_flags(entry):
        return MMU.extract_bits(entry, 0, 12)


class PageTableARM32(PageTable):
    entry_size = 4
    table_fields = [
        "Entry address",
        "Pointed address",
        "TEX",
        "Cacheble",
        "Bufferable",
        "Permissions",
        "Exec",
        "Kernel exec",
        "Secure",
        "Domain",
        "Shared",
        "Global",
        "Classes",
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


class PhysicalMemory(PhysicalMemoryDefault):
    pass


class CPU(CPUDefault):
    @classmethod
    def from_cpu_config(cls, cpu_config, **kwargs):
        return CPUARM32(cpu_config)

    def __init__(self, features):
        super(CPU, self).__init__(features)
        if self.endianness == "big":
            self.processor_features["opcode_unpack_fmt"] = ">I"
            CPU.extract_bits = CPU.extract_bits_big
        else:
            self.processor_features["opcode_unpack_fmt"] = "<I"
            CPU.extract_bits = CPU.extract_bits_little
        self.processor_features["instr_len"] = 4
        CPU.endianness = self.endianness
        CPU.processor_features = self.processor_features
        CPU.registers_values = self.registers_values


class CPUARM32(CPU):
    def __init__(self, features):
        super(CPUARM32, self).__init__(features)
        self.processor_features["opcode_to_mmu_regs"] = {
            (1, 0, 0, 0): "SCTLR",
            (2, 0, 0, 0): "TTBR0",
            (2, 0, 0, 1): "TTBR1",
            (2, 0, 0, 2): "TTBCR",
            (5, 0, 0, 0): "DFSR",
            (5, 0, 0, 1): "IFSR",
        }
        self.processor_features["opcode_to_gregs"] = [
            "R{}".format(i) for i in range(16)
        ]

        CPU.processor_features = self.processor_features
        CPU.registers_values = self.registers_values

    def parse_opcode(self, instr, page_addr, offset):
        # Collect locations of opcodes
        if (
            CPUARM32.extract_bits(instr, 24, 4) == 0b1110
            and CPUARM32.extract_bits(instr, 4, 1) == 1
            and CPUARM32.extract_bits(instr, 8, 4) == 0b1111
        ):
            opc1 = CPUARM32.extract_bits(instr, 21, 3)
            crn = CPUARM32.extract_bits(instr, 16, 4)
            rt = self.processor_features["opcode_to_gregs"][
                CPUARM32.extract_bits(instr, 12, 4)
            ]
            opc2 = CPUARM32.extract_bits(instr, 5, 3)
            crm = CPUARM32.extract_bits(instr, 0, 4)
            mmu_regs = self.processor_features["opcode_to_mmu_regs"]

            # MRC XXX, YYY (Read from Coprocessor register)
            if CPUARM32.extract_bits(instr, 20, 1) == 1:
                if (crn, opc1, crm, opc2) in mmu_regs:
                    if mmu_regs[(crn, opc1, crm, opc2)] not in [
                        "IFSR",
                        "TTBR0",
                        "TTBR1",
                        "DFSR",
                    ]:
                        return {}

                    return {
                        page_addr
                        + offset: {
                            "register": mmu_regs[(crn, opc1, crm, opc2)],
                            "gpr": [rt],
                            "f_addr": -1,
                            "f_parents": set(),
                            "instruction": "MRC",
                        }
                    }

            # MCR XXX, YYY (Write to Coprocessor register)
            else:
                if (crn, opc1, crm, opc2) in mmu_regs:
                    if mmu_regs[(crn, opc1, crm, opc2)] not in [
                        "TTBR0",
                        "TTBR1",
                        "TTBCR",
                        "SCTLR",
                    ]:
                        return {}

                    return {
                        page_addr
                        + offset: {
                            "register": mmu_regs[(crn, opc1, crm, opc2)],
                            "gpr": [rt],
                            "f_addr": -1,
                            "f_parents": set(),
                            "instruction": "MCR",
                        }
                    }
        return {}

    def identify_functions_start(self, addreses):
        machine = self.machine.get_miasm_machine()
        vm = self.machine.memory.get_miasm_vmmngr()
        mdis = machine.dis_engine(bin_stream_vm(vm), loc_db=LocationDB())
        mdis.follow_call = False
        mdis.dontdis_retcall = False
        instr_len = self.processor_features["instr_len"]

        logger = logging.getLogger("asmblock")
        logger.disabled = True

        for addr in tqdm(addreses):
            # For a passed address disassemble backward as long as we do not
            # find a unconditionally return or an invalid instruction
            cur_addr = addr

            # Maximum 10000 instructions
            instructions = 0
            while True and instructions <= 10000:
                # Stop if found an invalid instruction
                try:
                    asmcode = mdis.dis_instr(cur_addr)

                    # RFE/ERET/UDF
                    if asmcode.name in ["RFE", "ERET", "UDF", "B"]:
                        cur_addr += instr_len
                        break

                    # ARM has a loooot of ways to return from a routine, here the most used
                    # POP  {... PC, ...}
                    elif asmcode.name == "POP":
                        try:
                            for arg in asmcode.args:
                                if asmcode.arg2str(arg) in ["PC", "R15"]:
                                    raise UserWarning

                        except UserWarning:
                            cur_addr += instr_len
                            break

                    # Branch!
                    elif asmcode.name in ["B", "BX", "BXJ", "BL", "BLX"]:
                        # if asmcode.arg2str(asmcode.args[0]) in ["LR", "R14"]:
                        cur_addr += instr_len
                        break

                    # MOV PC, ....
                    elif asmcode.name == "MOV":
                        if asmcode.arg2str(asmcode.args[0]) in ["PC", "R15"]:
                            cur_addr += instr_len
                            break

                    # ORR R15, ....
                    elif asmcode.name == "ORR":
                        if asmcode.arg2str(asmcode.args[0]) in ["PC", "R15"]:
                            cur_addr += instr_len
                            break

                    # LDM XXX  {... PC, ...}
                    elif "LDM" in asmcode.name:
                        if "PC" in asmcode.arg2str(asmcode.args[1]):
                            cur_addr += instr_len
                            break

                    cur_addr -= instr_len
                    instructions += 1

                except Exception:
                    # Stop if found an invalid instruction
                    cur_addr += instr_len
                    break

            if instructions < 10000:
                addreses[addr]["f_addr"] = cur_addr
        del vm


class Machine(MachineDefault):
    def get_miasm_machine(self):
        mn_s = "arm" + ("b" if self.cpu.endianness == "big" else "l")
        return MIASMMachine(mn_s)


#################################################################
# MMU Modes
#################################################################


class MMU(MMURadix):
    PAGE_SIZE = 4096

    paging_unpack_format = "<I"
    page_table_class = PageTableARM32
    radix_levels = {"global": 2}
    top_prefix = 0
    entries_size = 4

    def __init__(self, mmu_config):
        super(MMU, self).__init__(mmu_config)
        self.mmu_endianness = mmu_config.get("endianness", "little")
        if self.mmu_endianness == "little":
            MMU.extract_bits = MMURadix.extract_bits_little
        else:
            MMU.extract_bits = MMURadix.extract_bits_big


class SHORT(MMU):
    map_ptr_entries_to_levels = {"global": [PTP, None]}
    map_datapages_entries_to_levels = {
        "global": [[PTSECTION, PTSUPERSECTION], [PTLARGE, PTSMALL]]
    }
    ttbcr_n = 0
    map_level_to_table_size = {"global": [0, 4096]}
    map_entries_to_shifts = {
        "global": {PTP: 10, PTSECTION: 20, PTSUPERSECTION: 24, PTLARGE: 16, PTSMALL: 12}
    }
    map_reserved_entries_to_levels = {"global": [[], []]}

    def reconstruct_table(
        self,
        frame_addr,
        frame_size,
        table_level,
        table_size,
        table_entries,
        empty_entries,
    ):
        # Reconstruct table_level tables, empty tables and data_pages of a given size
        frame_d = defaultdict(dict)
        page_tables = {}
        empty_tables = []
        data_pages = []
        for table_addr in range(frame_addr, frame_addr + frame_size, table_size):
            frame_d.clear()

            # Count the empty entries
            entry_addresses = set(
                range(table_addr, table_addr + table_size, MMU.entries_size)
            )
            empty_count = len(entry_addresses.intersection(empty_entries))

            # Reconstruct the content of the table candidate
            for entry_addr in entry_addresses.intersection(table_entries.keys()):
                entry_idx = (entry_addr - table_addr) // MMU.entries_size
                for entry_type in table_entries[entry_addr]:
                    frame_d[entry_type][entry_idx] = table_entries[entry_addr][
                        entry_type
                    ]

            # Classify the frame
            pt_classes = self.classify_frame(
                frame_d, empty_count, int(table_size // MMU.entries_size)
            )

            if -1 in pt_classes:  # Empty
                empty_tables.append(table_addr)
            elif -2 in pt_classes:  # Data
                data_pages.append(table_addr)
            elif table_level in pt_classes:
                table_obj = self.page_table_class(
                    table_addr, table_size, deepcopy(frame_d), [table_level]
                )
                page_tables[table_addr] = table_obj
            else:
                continue

        return page_tables, data_pages, empty_tables

    def aggregate_frames(self, frames, frame_size, page_size):
        pages = []
        frame_per_page = int(page_size // frame_size)
        frames = set(frames)

        for frame_addr in frames:
            if frame_addr % page_size != 0:
                continue

            if all(
                [
                    (frame_addr + idx * frame_size) in frames
                    for idx in range(1, frame_per_page)
                ]
            ):
                pages.append(frame_addr)

        return pages

    def parse_parallel_frame(self, addresses, frame_size, pidx, **kwargs):
        sleep(uniform(pidx, pidx + 1) // 1000)
        mm = copy(self.machine.memory)
        mm.reopen()

        # ARM machinery works differentely from INTEL and RISC-V ones:
        # parse all the records in a frame of 16KB and reconstruct all the different tables:
        # -> PTL1 tables of 1KiB
        # -> Empty tables of 1KiB
        # -> PTL0 kernel tables of 16KiB
        # -> PTL0 user tables of size depending by TTBCR
        # -> Data pages of 4KiB

        PTL0_USER_TABLE_SIZE = kwargs["ptl0_u_size"]
        PTL0_KERNEL_TABLE_SIZE = 16 * 1024
        PTL1_TABLE_SIZE = 1024

        # Prepare thread local dictionaries in which collect data
        data_frames = []
        empty_tables = []
        page_tables = {
            "user": [{} for i in range(self.radix_levels["global"])],
            "kernel": [{} for i in range(self.radix_levels["global"])],
        }

        # Cicle over every frame
        table_entries = defaultdict(dict)
        empty_entries = set()
        total_elems, iterator = addresses
        for frame_addr in tqdm(
            iterator, position=-pidx, total=total_elems, leave=False
        ):
            frame_buf = mm.get_data(frame_addr, frame_size)

            table_entries.clear()
            empty_entries.clear()

            # Unpack entries
            for entry_idx, entry in enumerate(
                iter_unpack(self.paging_unpack_format, frame_buf)
            ):
                entry_addr = frame_addr + entry_idx * MMU.entries_size

                entry_classes = self.classify_entry(
                    frame_addr, entry[0]
                )  # In this case frame_addr is not used

                # Data entry
                if None in entry_classes:
                    continue

                # Empty entry
                if False in entry_classes:
                    empty_entries.add(entry_addr)
                    continue

                # Valid entry
                for entry_obj in entry_classes:
                    entry_type = type(entry_obj)
                    table_entries[entry_addr][entry_type] = entry_obj

            empty_entries = set(empty_entries)
            # Look for PTL1 tables and empty tables
            t, d, e = self.reconstruct_table(
                frame_addr, frame_size, 1, PTL1_TABLE_SIZE, table_entries, empty_entries
            )
            page_tables["kernel"][1].update(t)
            data_frames.extend(d)
            empty_tables.extend(e)

            # Look for PTL0 Kernel tables
            t, _, _ = self.reconstruct_table(
                frame_addr,
                frame_size,
                0,
                PTL0_KERNEL_TABLE_SIZE,
                table_entries,
                empty_entries,
            )
            page_tables["kernel"][0].update(t)

            # Look for PTL0 User tables
            if PTL0_USER_TABLE_SIZE != PTL0_KERNEL_TABLE_SIZE:
                t, _, _ = self.reconstruct_table(
                    frame_addr,
                    frame_size,
                    0,
                    PTL0_USER_TABLE_SIZE,
                    table_entries,
                    empty_entries,
                )
                page_tables["user"][0].update(t)

        # Reconstruct data_pages
        data_pages = self.aggregate_frames(data_frames, 1024, 4096)

        return page_tables, data_pages, empty_tables

    def classify_entry(self, page_addr, entry):
        classification = []
        class_bits = MMU.extract_bits(entry, 0, 2)
        # BITS 0,1 determine the class
        if class_bits == 0b00:
            return [False]

        elif class_bits == 0b01:
            # BIT 9 (which is IMPLEMENTATION DEFINED) it has to be zero
            if not MMU.extract_bits(
                entry, 9, 1
            ):  # WARNING! Specs require bit 4 = 0 but OSs do not respect it... :/
                addr = PTP.extract_addr(entry)
                if addr not in self.machine.memory.physpace["not_valid_regions"]:
                    flags = PTP.extract_flags(entry)
                    classification.append(PTP(addr, flags))

            # For Large page entries BIT[6,8] must be 0
            if not MMU.extract_bits(entry, 6, 3):
                addr = PTLARGE.extract_addr(entry)
                flags = PTLARGE.extract_flags(entry)
                ptlarge_obj = PTLARGE(addr, flags)

                if not (
                    ptlarge_obj.extract_tex() == 1
                    and not ptlarge_obj.is_bufferable_entry()
                    and ptlarge_obj.is_cachable_entry()
                ):
                    classification.append(ptlarge_obj)
        else:
            # If bit 9 is 0 it can be a Section/Supersection (bit 9 is IMPLEMENTATION DEFINED)
            if not MMU.extract_bits(entry, 9, 1):
                addr = PTSECTION.extract_addr(entry)
                flags = PTSECTION.extract_flags(entry)
                section_obj = PTSECTION(addr, flags)
                # Some values of TEX, C and B are RESERVED
                if not (
                    section_obj.extract_tex() == 1
                    and not section_obj.is_bufferable_entry()
                    and section_obj.is_cachable_entry()
                ):
                    # BIT 18 discriminate between Section (0) and Supersection (1)
                    if MMU.extract_bits(entry, 18, 1):
                        super_section_addr = PTSUPERSECTION.extract_addr(entry)
                        super_section_flags = PTSUPERSECTION.extract_flags(entry)
                        classification.append(
                            PTSUPERSECTION(super_section_addr, super_section_flags)
                        )
                    else:
                        classification.append(section_obj)

            # Small page
            # Some values of TEX, C and B are RESERVED
            addr = PTSMALL.extract_addr(entry)
            flags = PTSMALL.extract_flags(entry)
            entry_obj = PTSMALL(addr, flags)
            if not (
                entry_obj.extract_tex() == 1
                and not entry_obj.is_bufferable_entry()
                and entry_obj.is_cachable_entry()
            ):
                classification.append(entry_obj)

        if not classification:  # No valid class found
            return [None]
        return classification


class MMUShell(MMUShellDefault):
    def __init__(self, completekey="tab", stdin=None, stdout=None, machine={}):
        super(MMUShell, self).__init__(completekey, stdin, stdout, machine)

        if not self.data:
            self.data = Data(
                is_tables_found=False,
                is_radix_found=False,
                is_registers_found=False,
                opcodes={},
                regs_values={},
                page_tables={
                    "user": [
                        {} for i in range(self.machine.mmu.radix_levels["global"])
                    ],
                    "kernel": [
                        {} for i in range(self.machine.mmu.radix_levels["global"])
                    ],
                },
                data_pages=[],
                empty_tables=[],
                reverse_map_tables={
                    "user": [
                        defaultdict(set)
                        for i in range(self.machine.mmu.radix_levels["global"])
                    ],
                    "kernel": [
                        defaultdict(set)
                        for i in range(self.machine.mmu.radix_levels["global"])
                    ],
                },
                reverse_map_pages={
                    "user": [
                        defaultdict(set)
                        for i in range(self.machine.mmu.radix_levels["global"])
                    ],
                    "kernel": [
                        defaultdict(set)
                        for i in range(self.machine.mmu.radix_levels["global"])
                    ],
                },
                used_ttbcr=None,
                ttbrs=defaultdict(dict),
            )

    def reload_data_from_file(self, data_filename):
        super(MMUShell, self).reload_data_from_file(data_filename)
        SHORT.ttbcr_n = self.data.used_ttbcr.n

    def do_find_registers_values(self, arg):
        """Find MMU load opcodes and execute MMU related functions inside the memory dump in order to extract MMU registers values"""

        if self.data.is_registers_found:
            logging.warning("Registers already searched")
            return

        logger.info("Look for opcodes related to MMU setup...")
        parallel_results = self.machine.apply_parallel(
            self.machine.mmu.PAGE_SIZE, self.machine.cpu.parse_opcodes_parallel
        )

        opcodes = {}
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            opcodes.update(result.get())

        self.data.opcodes = opcodes

        # Filter to look only for opcodes which write on MMU register only and not read from them or from other registers
        filter_f = (
            lambda it: True
            if it[1]["register"] == "TTBCR" and it[1]["instruction"] == "MCR"
            else False
        )
        mmu_wr_opcodes = {k: v for k, v in filter(filter_f, opcodes.items())}

        logging.info("Use heuristics to find function addresses...")
        logging.info("This analysis could be extremely slow!")
        self.machine.cpu.identify_functions_start(mmu_wr_opcodes)

        logging.info("Identify register values using data flow analysis...")

        # We use data flow analysis and merge the results
        dataflow_values = self.machine.cpu.find_registers_values_dataflow(
            mmu_wr_opcodes
        )

        filtered_values = defaultdict(set)
        for register, values in dataflow_values.items():
            for value in values:
                reg_obj = CPURegARM32.get_register_obj(register, value)
                if reg_obj.valid and not any(
                    [
                        val_obj.is_mmu_equivalent_to(reg_obj)
                        for val_obj in filtered_values[register]
                    ]
                ):
                    filtered_values[register].add(reg_obj)

        # Add default values
        reg_obj = CPURegARM32.get_register_obj(
            "TTBCR", self.machine.cpu.registers_values["TTBCR"]
        )
        if reg_obj.valid and all(
            [not reg_obj.is_mmu_equivalent_to(x) for x in filtered_values["TTBCR"]]
        ):
            filtered_values["TTBCR"].add(reg_obj)

        self.data.regs_values = filtered_values
        self.data.is_registers_found = True

        # Show results
        logging.info("TTBCR values recovered:")
        for reg_obj in self.data.regs_values["TTBCR"]:
            logging.info(reg_obj)

    def do_show_registers(self, args):
        """Show registers values found"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        for registers in sorted(self.data.regs_values.keys()):
            for register in self.data.regs_values[registers]:
                print(register)

    def do_set_ttbcr(self, args):
        """Set the value of TTBCR register to be used"""
        args = args.split()
        if len(args) == 0:
            logging.error("Please use find_tables TTBCR_VALUE")
            return

        # The shape of PTL0 tables depends on N value of TTBCR register
        try:
            ttbcr_val = self.parse_int(args[0])
            ttbcr = TTBCR(ttbcr_val)
            if not ttbcr.valid:
                raise ValueError
        except ValueError:
            logger.warning("Invalid TTBCR value")
            return

        self.data.used_ttbcr = ttbcr
        SHORT.ttbcr_n = ttbcr.n

    def do_find_tables(self, args):
        """Find MMU tables in memory"""
        if not self.data.used_ttbcr:
            logging.error("Please set a TTBCR register to use, using set_ttbcr TTBCR")
            return
        ttbcr = self.data.used_ttbcr

        # Delete all the previous table data
        if self.data.is_tables_found:
            for mode in ["user", "kernel"]:
                self.data.reverse_map_pages[mode].clear()
                self.data.reverse_map_tables[mode].clear()
                self.data.page_tables[mode].clear()
            self.data.empty_tables = []
            self.data.data_pages = []

        # Parse memory in chunk of 16KiB
        PTL0_USER_TABLE_SIZE = ttbcr.get_ptl0_user_table_size()
        logger.info("Look for paging tables...")
        parallel_results = self.machine.apply_parallel(
            16384,
            self.machine.mmu.parse_parallel_frame,
            ptl0_u_size=PTL0_USER_TABLE_SIZE,
        )
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            page_tables, data_pages, empty_tables = result.get()

            for level in range(self.machine.mmu.radix_levels["global"]):
                self.data.page_tables["user"][level].update(page_tables["user"][level])
                self.data.page_tables["kernel"][level].update(
                    page_tables["kernel"][level]
                )

            self.data.data_pages.extend(data_pages)
            self.data.empty_tables.extend(empty_tables)

        self.data.data_pages = set(self.data.data_pages)
        self.data.empty_tables = set(self.data.empty_tables)

        # Prepare fo dumplicates removing
        modes = ["kernel"]
        fps = {"kernel": []}
        if PTL0_USER_TABLE_SIZE != 16384:
            modes.append("user")
            fps["user"] = []
            self.data.page_tables["user"][1] = deepcopy(
                self.data.page_tables["kernel"][1]
            )

        # Remove all tables which point to inexistent table of lower level
        logger.info("Reduce false positives...")
        ptr_class = self.machine.mmu.map_ptr_entries_to_levels["global"][0]

        for mode in modes:
            referenced_nxt = []
            for table_addr in list(self.data.page_tables[mode][0].keys()):
                for entry_obj in (
                    self.data.page_tables[mode][0][table_addr]
                    .entries[ptr_class]
                    .values()
                ):
                    if (
                        entry_obj.address not in self.data.page_tables[mode][1]
                        and entry_obj.address not in self.data.empty_tables
                    ):
                        # Remove the table
                        self.data.page_tables[mode][0].pop(table_addr)
                        break

                    else:
                        referenced_nxt.append(entry_obj.address)

            # Remove table not referenced by upper levels
            referenced_nxt = set(referenced_nxt)
            for table_addr in set(self.data.page_tables[mode][1].keys()).difference(
                referenced_nxt
            ):
                self.data.page_tables[mode][1].pop(table_addr)

        logger.info("Fill reverse maps...")
        for mode in modes:
            for lvl in range(0, self.machine.mmu.radix_levels["global"]):
                ptr_class = self.machine.mmu.map_ptr_entries_to_levels["global"][lvl]
                page_classes = self.machine.mmu.map_datapages_entries_to_levels[
                    "global"
                ][lvl]
                for table_addr, table_obj in self.data.page_tables[mode][lvl].items():
                    for entry_obj in table_obj.entries[ptr_class].values():
                        self.data.reverse_map_tables[mode][lvl][entry_obj.address].add(
                            table_obj.address
                        )
                    for page_class in page_classes:
                        for entry_obj in table_obj.entries[page_class].values():
                            self.data.reverse_map_pages[mode][lvl][
                                entry_obj.address
                            ].add(table_obj.address)

        self.data.is_tables_found = True

    def do_show_table(self, args):
        """Show an MMU table at a chosen address. Usage show_table ADDRESS [level table size]"""
        if not self.data.used_ttbcr:
            logging.error("Please set a TTBCR register to use, using set_ttbcr TTBCR")
            return

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

        if len(args) == 3:
            valid_sizes = {0: set([0x4000]), 1: set([0x400])}
            valid_sizes[0].add(self.data.used_ttbcr.get_ptl0_user_table_size())

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

            try:
                table_size = self.parse_int(args[2])
                if table_size not in valid_sizes[lvl]:
                    logging.warning(
                        f"Size not allowed for choosen level! Valid sizes are:{valid_sizes[lvl]}"
                    )
                    return
            except ValueError:
                logger.warning("Invalid size value")
                return
        else:
            table_size = 0x4000
            lvl = -1

        table_buff = self.machine.memory.get_data(addr, table_size)
        invalids, pt_classes, table_obj = self.machine.mmu.parse_frame(
            table_buff, addr, table_size, lvl
        )
        print(table_obj)
        print(f"Invalid entries: {invalids} Table levels: {pt_classes}")

    def virtspace_short(
        self, addr, page_tables, lvl=0, prefix=0, ukx=True, cache=defaultdict(dict)
    ):
        """Recursively reconstruct virtual address space for SHORT mode"""

        virtspace = VASShort()
        data_classes = self.machine.mmu.map_datapages_entries_to_levels["global"][lvl]
        ptr_class = self.machine.mmu.map_ptr_entries_to_levels["global"][lvl]
        cache[lvl][addr] = defaultdict(portion.empty)

        if lvl == self.machine.mmu.radix_levels["global"] - 1:
            for data_class in data_classes:
                shift = self.machine.mmu.map_entries_to_shifts["global"][data_class]
                for entry_idx, entry in (
                    page_tables[lvl][addr].entries[data_class].items()
                ):
                    permissions = entry.extract_permissions()
                    kx = entry.is_kernel_executable_entry() and ukx
                    x = entry.is_executable_entry()

                    virt_addr = prefix | (entry_idx << shift)
                    virtspace[(permissions, x, kx)] |= portion.closedopen(
                        virt_addr, virt_addr + entry.size
                    )
                    cache[lvl][addr][(permissions, x, kx)] |= portion.closedopen(
                        virt_addr, virt_addr + entry.size
                    )

            return virtspace

        else:
            if ptr_class in page_tables[lvl][addr].entries:
                shift = self.machine.mmu.map_entries_to_shifts["global"][ptr_class]
                for entry_idx, entry in (
                    page_tables[lvl][addr].entries[ptr_class].items()
                ):
                    if entry.address not in page_tables[lvl + 1]:
                        continue
                    else:
                        if entry.address not in cache[lvl + 1]:
                            permissions = entry.extract_permissions()
                            kx = entry.is_kernel_executable_entry() and ukx
                            x = entry.is_executable_entry()

                            virt_addr = prefix | (entry_idx << shift)
                            low_virts = self.virtspace_short(
                                entry.address,
                                page_tables,
                                lvl + 1,
                                virt_addr,
                                kx,
                                cache=cache,
                            )
                        else:
                            low_virts = cache[lvl + 1][entry.address]

                        for perm, virts_fragment in low_virts.items():
                            virtspace[perm] |= virts_fragment
                            cache[lvl][addr][perm] |= virts_fragment

            for data_class in data_classes:
                if (
                    data_class in page_tables[lvl][addr].entries
                    and data_class is not None
                ):
                    shift = self.machine.mmu.map_entries_to_shifts["global"][data_class]
                    for entry_idx, entry in (
                        page_tables[lvl][addr].entries[data_class].items()
                    ):
                        permissions = entry.extract_permissions()
                        kx = entry.is_kernel_executable_entry() and ukx
                        x = entry.is_executable_entry()

                        virt_addr = prefix | (entry_idx << shift)
                        virtspace[(permissions, x, kx)] |= portion.closedopen(
                            virt_addr, virt_addr + entry.size
                        )
                        cache[lvl][addr][(permissions, x, kx)] |= portion.closedopen(
                            virt_addr, virt_addr + entry.size
                        )

            return virtspace

    def do_find_radix_trees(self, args):
        """Reconstruct radix trees"""
        if not self.data.is_tables_found:
            logging.info("Please, parse the memory first!")
            return

        if not self.data.is_registers_found:
            logging.info("Please find MMU related opcodes first!")
            return

        if self.data.ttbrs:
            self.data.ttbrs.clear()

        # Some table level was not found...
        if not len(self.data.page_tables["kernel"][0]):
            logger.warning("OOPS... no tables in first level... Wrong MMU mode?")
            return

        # Go back from PTL1 up to PTL0, the particular form of PTLn permits to find PTL0
        logging.info("Go up the paging trees starting from data pages...")
        candidates = {"kernel": []}
        modes = ["kernel"]
        if SHORT.ttbcr_n != 0:
            modes.append("user")
            candidates["user"] = []

        # TTBR0 can be used by a process which manages HW so it can point only to not in RAM pages!
        # This is a special case which it can occur only in ARM
        not_ram_pages = []
        for p in self.machine.memory.physpace["not_ram"]:
            not_ram_pages.extend([x for x in range(p.lower, p.upper, 4096)])
        not_ram_pages = set(not_ram_pages)

        for mode in modes:
            already_explored = set()
            for page_addr in tqdm(
                self.data.data_pages.union(self.data.empty_tables).union(not_ram_pages)
            ):
                derived_addresses = self.machine.mmu.derive_page_address(page_addr)
                for derived_address in derived_addresses:
                    if derived_address in already_explored:
                        continue

                    lvl, addr = derived_address
                    candidates[mode].extend(
                        self.radix_roots_from_data_page(
                            lvl,
                            addr,
                            self.data.reverse_map_pages[mode],
                            self.data.reverse_map_tables[mode],
                        )
                    )
                    already_explored.add(derived_address)

            candidates[mode] = list(
                set(candidates[mode]).intersection(
                    self.data.page_tables[mode][0].keys()
                )
            )
            candidates[mode].sort()

        # Collect interrupt/paging opcodes
        filter_f_read = (
            lambda it: True
            if it[1]["register"] in ["DFSR", "IFSR"] and it[1]["instruction"] == "MRC"
            else False
        )
        kernel_opcodes_read = [
            x[0] for x in filter(filter_f_read, self.data.opcodes.items())
        ]
        filter_f_write = (
            lambda it: True
            if it[1]["register"] in ["TTBR0", "TTBCR"] and it[1]["instruction"] == "MCR"
            else False
        )
        kernel_opcodes_write = [
            x[0] for x in filter(filter_f_write, self.data.opcodes.items())
        ]

        logging.info("Filtering candidates...")
        filtered = {"kernel": {}, "user": {}}
        for mode in modes:
            physpace_cache = defaultdict(
                dict
            )  # We need to use different caches for user and kernel modes
            virtspace_cache = defaultdict(dict)
            for candidate in tqdm(candidates[mode]):
                consistency, pas = self.physpace(
                    candidate,
                    self.data.page_tables[mode],
                    self.data.empty_tables,
                    cache=physpace_cache,
                )

                # Ignore inconsistent radix-tress or which maps zero spaces
                if not consistency or (
                    pas.get_kernel_size() == 0 and pas.get_user_size() == 0
                ):
                    continue

                # Look for kernel trees able to map at least one interrupt/paging related opcodes
                if mode == "kernel":
                    # We check also in user pages (when ttbr1 is not used!) because user pages are always accessible also by the kernel!
                    if not any(
                        [opcode_addr in pas for opcode_addr in kernel_opcodes_read]
                    ) or (
                        SHORT.ttbcr_n != 0
                        and not any(
                            [opcode_addr in pas for opcode_addr in kernel_opcodes_write]
                        )
                    ):
                        continue

                    vas = self.virtspace_short(
                        candidate, self.data.page_tables[mode], cache=virtspace_cache
                    )

                    # At least a kernel executable page must be exist
                    for _, _, kx in vas:
                        if kx:
                            break
                    else:
                        continue

                    radix_tree = RadixTree(
                        candidate, 0, pas, vas, kernel=True, user=False
                    )
                    filtered[mode][candidate] = radix_tree

                else:
                    # No kernel pages are allowed on user radix trees!
                    if pas.get_kernel_size() != 0:
                        continue

                    # At least an executable page must exists
                    vas = self.virtspace_short(
                        candidate, self.data.page_tables[mode], cache=virtspace_cache
                    )
                    for _, x, _ in vas:
                        if x:
                            break
                    else:
                        continue

                    # Filter for at least a writable page for user (AP[1] == 1)
                    for p, _, _ in vas:
                        if p & 0b10 == 2:
                            break
                    else:
                        continue

                    radix_tree = RadixTree(
                        candidate, 0, pas, vas, kernel=False, user=True
                    )
                    filtered[mode][candidate] = radix_tree

        self.data.ttbrs = filtered
        self.data.is_radix_found = True

    def do_show_radix_trees(self, args):
        """Show radix trees found"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        labels = [
            "Radix address",
            "First level",
            "Kernel size (Bytes)",
            "User size (Bytes)",
            "Kernel",
        ]
        table = PrettyTable()
        table.field_names = labels
        for mode in ["kernel", "user"]:
            for ttbr in self.data.ttbrs[mode].values():
                table.add_row(
                    ttbr.entry_resume_stringified() + ["X" if mode == "kernel" else ""]
                )
        table.sortby = "Radix address"
        print(table)


class MMUShellGTruth(MMUShell):
    def do_show_registers_gtruth(self, args):
        """Compare TTBCR register values found with the ground truth"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        # Check if the last value of TTBCR was found
        all_ttbcrs = (
            {}
        )  # QEMU export TTBCR inside various registers as TTBCR, TTBCR, TCR_S etc. due to it's capability to emulate different ARM/ARM64 systems
        for reg_name, value_data in self.gtruth.items():
            if "TCR" in reg_name or "TTBCR" in reg_name:
                for value, value_info in value_data.items():
                    if value not in all_ttbcrs or (
                        value_info[1] > all_ttbcrs[value][1]
                    ):
                        all_ttbcrs[value] = (value_info[0], value_info[1])

        last_ttbcr = TTBCR(
            sorted(all_ttbcrs.keys(), key=lambda x: all_ttbcrs[x][1], reverse=True)[0]
        )

        ttbcr_fields_equals = {}
        for value_found_obj in self.data.regs_values["TTBCR"]:
            ttbcr_fields_equals[value_found_obj] = value_found_obj.count_fields_equals(
                last_ttbcr
            )
        k_sorted = sorted(
            ttbcr_fields_equals.keys(),
            key=lambda x: ttbcr_fields_equals[x],
            reverse=True,
        )
        tcr_found = k_sorted[0]
        correct_fields_found = ttbcr_fields_equals[tcr_found]

        if correct_fields_found:
            print(f"Correct TTBCR value: {last_ttbcr}, Found: {tcr_found}")
            print("TTBCR fields found:... {}/2".format(correct_fields_found))
            print("FP: {}".format(str(len(self.data.regs_values["TTBCR"]) - 1)))
        else:
            print(f"Correct TTBCR value: {last_ttbcr}")
            print("TTBCR fields found:... 0/2")
            print("FP: {}".format(str(len(self.data.regs_values["TTBCR"]))))

    def do_show_radix_trees_gtruth(self, args):
        """Compare found radix trees with the ground truth"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        # Collect TTBR0 and TTBR1 values from the gtruth
        ttbr0s = {}
        ttbr1s = {}
        ttbr0_phy_cache = defaultdict(dict)
        ttbr1_phy_cache = defaultdict(dict)
        virt_cache = defaultdict(dict)
        filter_f_read = (
            lambda it: True
            if it[1]["register"] in ["DFSR", "IFSR"] and it[1]["instruction"] == "MRC"
            else False
        )
        kernel_opcodes_read = [
            x[0] for x in filter(filter_f_read, self.data.opcodes.items())
        ]
        filter_f_write = (
            lambda it: True
            if it[1]["register"] in ["TTBR0", "TTBCR"] and it[1]["instruction"] == "MCR"
            else False
        )
        kernel_opcodes_write = [
            x[0] for x in filter(filter_f_write, self.data.opcodes.items())
        ]

        # User or kernel+user radix trees
        for key in ["TTBR0", "TTBR0_S", "TTBR0_EL1", "TTBR0_EL1_S"]:
            for value, data in self.gtruth.get(key, {}).items():
                ttbr = TTBR0(value)
                if any([ttbr.is_mmu_equivalent_to(x[0]) for x in ttbr0s.values()]):
                    continue

                if SHORT.ttbcr_n != 0:
                    if ttbr.address not in self.data.page_tables["user"][0]:
                        continue

                    consistency, pas = self.physpace(
                        ttbr.address,
                        self.data.page_tables["user"],
                        self.data.empty_tables,
                        cache=ttbr0_phy_cache,
                    )
                    if not consistency:
                        continue

                    if pas.get_kernel_size() != 0:
                        continue

                    virtspace = self.virtspace_short(
                        ttbr.address, self.data.page_tables["user"], cache=virt_cache
                    )
                    for _, x, _ in virtspace:
                        if x:
                            break
                    else:
                        continue

                    # Filter for at least a writable page
                    for p, _, _ in virtspace:
                        if p & 0b10 == 2:
                            break
                    else:
                        continue

                    ttbr0s[ttbr.address] = (ttbr, data)

                else:
                    if ttbr.address not in self.data.page_tables["kernel"][0]:
                        continue

                    consistency, pas = self.physpace(
                        ttbr.address,
                        self.data.page_tables["kernel"],
                        self.data.empty_tables,
                        cache=ttbr0_phy_cache,
                    )
                    if not consistency or (
                        pas.get_kernel_size() == 0 and pas.get_user_size() == 0
                    ):
                        continue

                    if not any(
                        [opcode_addr in pas for opcode_addr in kernel_opcodes_read]
                    ) or not any(
                        [opcode_addr in pas for opcode_addr in kernel_opcodes_write]
                    ):
                        continue

                    # At least a kernel executable page must be exist
                    virtspace = self.virtspace_short(
                        ttbr.address, self.data.page_tables["kernel"], cache=virt_cache
                    )
                    for _, _, kx in virtspace:
                        if kx:
                            break
                    else:
                        continue

                    ttbr0s[ttbr.address] = (ttbr, data)

        # Use only TTBR0 if TTBCR.N = 0
        if SHORT.ttbcr_n != 0:
            virt_cache = defaultdict(dict)
            for key in ["TTBR1", "TTBR1_S", "TTBR1_EL1", "TTBR1_EL1_S"]:
                for value, data in self.gtruth.get(key, {}).items():
                    ttbr = TTBR1(value)
                    if any([ttbr.is_mmu_equivalent_to(x[0]) for x in ttbr1s.values()]):
                        continue

                    if ttbr.address not in self.data.page_tables["kernel"][0]:
                        continue

                    consistency, pas = self.physpace(
                        ttbr.address,
                        self.data.page_tables["kernel"],
                        self.data.empty_tables,
                        cache=ttbr1_phy_cache,
                    )
                    if not consistency or (
                        pas.get_kernel_size() == 0 and pas.get_user_size() == 0
                    ):
                        continue

                    if not any(
                        [opcode_addr in pas for opcode_addr in kernel_opcodes_read]
                    ):
                        continue

                    # At least a kernel executable page must be exist
                    virtspace = self.virtspace_short(
                        ttbr.address, self.data.page_tables["kernel"], cache=virt_cache
                    )
                    for _, _, kx in virtspace:
                        if kx:
                            break
                    else:
                        continue

                    ttbr1s[ttbr.address] = (ttbr, data)

        # True positives, false negatives, false positives
        if SHORT.ttbcr_n == 0:
            tps = sorted(
                set(ttbr0s.keys()).intersection(set(self.data.ttbrs["kernel"].keys()))
            )
            fns = sorted(
                set(ttbr0s.keys()).difference(set(self.data.ttbrs["kernel"].keys()))
            )
            fps = sorted(
                set(self.data.ttbrs["kernel"].keys()).difference(set(ttbr0s.keys()))
            )
        else:
            tps = sorted(
                set(ttbr1s.keys()).intersection(set(self.data.ttbrs["kernel"].keys()))
            )
            fns = sorted(
                set(ttbr1s.keys()).difference(set(self.data.ttbrs["kernel"].keys()))
            )
            fps = sorted(
                set(self.data.ttbrs["kernel"].keys()).difference(set(ttbr1s.keys()))
            )
            tpsu = sorted(
                set(ttbr0s.keys()).intersection(set(self.data.ttbrs["user"].keys()))
            )
            fnsu = sorted(
                set(ttbr0s.keys()).difference(set(self.data.ttbrs["user"].keys()))
            )
            fpsu = sorted(
                set(self.data.ttbrs["user"].keys()).difference(set(ttbr0s.keys()))
            )

        # Show results
        table = PrettyTable()
        table.field_names = ["Address", "Found", "Mode", "First seen", "Last seen"]
        if SHORT.ttbcr_n == 0:
            kernel_regs = ttbr0s
            mode = "K/U"
        else:
            kernel_regs = ttbr1s
            mode = "K"

        for tp in sorted(tps):
            table.add_row(
                [hex(tp), "X", mode, kernel_regs[tp][1][0], kernel_regs[tp][1][1]]
            )

        for fn in sorted(fns):
            table.add_row(
                [hex(fn), "", mode, kernel_regs[fn][1][0], kernel_regs[fn][1][1]]
            )

        for fp in sorted(fps):
            table.add_row([hex(fp), "False positive", mode, "", ""])

        # User
        if SHORT.ttbcr_n != 0:
            for tp in sorted(tpsu):
                table.add_row([hex(tp), "X", "U", ttbr0s[tp][1][0], ttbr0s[tp][1][1]])

            for fn in sorted(fnsu):
                table.add_row([hex(fn), "", "U", ttbr0s[fn][1][0], ttbr0s[fn][1][1]])

            for fp in sorted(fpsu):
                table.add_row([hex(fp), "False positive", "U", "", ""])

        print(table)
        print(f"TP:{len(tps)} FN:{len(fns)} FP:{len(fps)}")
        if SHORT.ttbcr_n != 0:
            print(f"USER TP:{len(tpsu)} FN:{len(fnsu)} FP:{len(fpsu)}")
