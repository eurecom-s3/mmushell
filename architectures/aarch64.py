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


def _dummy_f():  # Workaround pickle defaultdict
    return defaultdict(set)


# For AArch64 TCR.TxSZ and TCR.TGx control the size of the address space in mode x (0 user, 1 kernel) and the size of the granule (the data
# page). The structure of the radix tree is very complicated and depends on both TxSZ and TGx (see get_trees_struct()). The tree can have
# a variable number of levels depending on TGx and TxSZ (the TTBRx_EL1 points to the first level available in the tree)
# and the tables belonging to the first level have a size controlled by TxSZ and depends also by the granule size. The table of inferior
# levels have always a size equal to the granule size used. Records in table have the same shape for tables of upper levels and different
# for table of lower one, permitting loops


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
    used_tcr: None
    ttbrs: dict


class CPURegAArch64(CPUReg):
    @classmethod
    def get_register_obj(cls, reg_name, value):
        return globals()[reg_name](value)


class TCR_EL1(CPURegAArch64):
    def is_valid(self, value):
        if (
            CPU.extract_bits(value, 6, 1) != 0
            or CPU.extract_bits(value, 35, 1) != 0
            or CPU.extract_bits(value, 59, 5) != 0
        ):
            return False
        else:
            return True

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.t0sz = CPU.extract_bits(value, 0, 6)
            self.t1sz = CPU.extract_bits(value, 16, 6)
            self.tg0 = CPU.extract_bits(value, 14, 2)
            self.tg1 = CPU.extract_bits(value, 30, 2)

        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.t0sz == other.t0sz
            and self.t1sz == other.t1sz
            and self.tg0 == other.tg0
            and self.tg1 == other.tg1
        )

    def count_fields_equals(self, other):
        tot = 0
        tot += 1 if self.t0sz == other.t0sz else 0
        tot += 1 if self.t1sz == other.t1sz else 0
        tot += 1 if self.tg0 == other.tg0 else 0
        tot += 1 if self.tg1 == other.tg1 else 0

        return tot

    def get_kernel_granule(self):
        if self.tg1 == 1:
            return 16384
        elif self.tg1 == 2:
            return 4096
        elif self.tg1 == 3:
            return 65536
        else:
            return -1

    def get_user_granule(self):
        if self.tg0 == 0:
            return 4096
        elif self.tg0 == 1:
            return 65536
        elif self.tg0 == 2:
            return 16384
        else:
            return -1

    def get_user_size_offset(self):
        # Some OS are bugged and do not set it to a valid value (accordinging to ARM manual) waiting for a reset one...
        if self.t0sz < 16:
            return 21
        else:
            return self.t0sz

    def get_kernel_size_offset(self):
        # Some OS are bugged and do not set it to a valid value (accordinging to ARM manual) waiting for a reset one...
        if self.t1sz < 16:
            return 21
        else:
            return self.t1sz

    def get_kernel_address_size(self):
        return 1 << (64 - self.get_kernel_size_offset())

    def get_user_address_size(self):
        return 1 << (64 - self.get_user_size_offset())

    # def get_addresses_size(self):
    #     return {"kernel": self.get_kernel_address_size(), "user": self.get_user_address_size() }

    def _get_tree_struct(self, granule, size_offset):
        if granule == 4096:
            if 12 <= size_offset <= 24:
                t = (0, 1 << (28 - size_offset))
            elif 25 <= size_offset <= 33:
                t = (1, 1 << (37 - size_offset))
            elif 34 <= size_offset <= 42:
                t = (2, 1 << (46 - size_offset))
            else:
                t = (3, 1 << (55 - size_offset))
        elif granule == 16384:
            if size_offset == 16:
                t = (0, 16)
            elif 17 <= size_offset <= 27:
                t = (1, 1 << (31 - size_offset))
            elif 28 <= size_offset <= 38:
                t = (2, 1 << (42 - size_offset))
            else:
                t = (3, 1 << (53 - size_offset))
        elif granule == 65536:
            if 12 <= size_offset <= 21:
                t = (1, 1 << (25 - size_offset))
            elif 22 <= size_offset <= 34:
                t = (2, 1 << (38 - size_offset))
            else:
                t = (3, 1 << (51 - size_offset))
        else:
            t = (0, -1)

        return {"granule": granule, "total_levels": 4 - t[0], "top_table_size": t[1]}

    def get_trees_struct(self):
        ret = {"kernel": None, "user": None}
        ret["kernel"] = self._get_tree_struct(
            self.get_kernel_granule(), self.get_kernel_size_offset()
        )
        ret["user"] = self._get_tree_struct(
            self.get_user_granule(), self.get_user_size_offset()
        )

        # WORKAROUND: some OS do not set correctly the register values, using invalid one on real hw...
        if ret["kernel"]["top_table_size"] == -1:
            ret["kernel"] = ret["user"]
        if ret["user"]["top_table_size"] == -1:
            ret["user"] = ret["kernel"]
        return ret

    def __repr__(self):
        return f"TCR_EL1 {hex(self.value)} => T0SZ:{hex(self.t0sz)}, T1SZ:{hex(self.t1sz)}, TG0:{hex(self.tg0)}, TG1:{hex(self.tg1)}"


class TTBR(CPURegAArch64):
    reg_name = "TTBR"
    mode = ""

    def is_valid(self, value):
        return True

    def _calculate_x(self):
        tcr = LONG.tcr
        trees_struct = tcr.get_trees_struct()
        granule = trees_struct[self.mode]["granule"]
        total_levels = trees_struct[self.mode]["total_levels"]
        if self.mode == "kernel":
            txsz = tcr.t1sz
        else:
            txsz = tcr.t0sz

        if granule == 4096:
            step = 9
            max_value = 55
        elif granule == 16384:
            step = 11
            max_value = 53
        else:
            step = 13
            max_value = 51

        return (max_value - total_levels * step) - txsz

    def _get_radix_base(self, value):
        x = self._calculate_x()
        return CPU.extract_bits(value, x, 47 - x + 1) << x

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.cnp = CPU.extract_bits(value, 0, 1)
            self.address = self._get_radix_base(value)
            self.asid = CPU.extract_bits(value, 48, 16)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.address == other.address

    def __repr__(self):
        return f"{self.reg_name} {hex(self.value)} => ASID:{hex(self.asid)}, Address:{hex(self.address)}, CnP: {self.cnp}"


class TTBR0_EL1(TTBR):
    reg_name = "TTBR0_EL1"
    mode = "user"


class TTBR1_EL1(TTBR):
    reg_name = "TTBR1_EL1"
    mode = "kernel"


#####################################################################
# 64 bit entries and page table
#####################################################################


class TEntry64(TableEntry):
    entry_size = 8
    entry_name = "TEntry64"
    size = 0
    labels = [
        "Address:",
        "Attributes:",
        "Secure:",
        "Permissions:",
        "Shareability:",
        "Accessed:",
        "Global:",
        "Block:",
        "Guarded:",
        "Dirty:",
        "Continous:",
        "Kernel exec:",
        "Exec:",
    ]
    addr_fmt = "0x{:016x}"

    def __init__(self, address, lower_flags, upper_flags):
        self.address = address
        self.lower_flags = lower_flags
        self.upper_flags = upper_flags

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
            self.extract_attributes(),
            self.is_secure_entry(),
            self.extract_permissions(),
            self.extract_shareability(),
            self.is_accessed_entry(),
            self.is_global_entry(),
            self.is_block_entry(),
            self.is_guarded_entry(),
            self.is_dirty_entry(),
            self.is_continuous_entry(),
            self.is_kernel_executable_entry(),
            self.is_executable_entry(),
        ]

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[0] = self.addr_fmt.format(res[0])
        res[1] = self.addr_fmt.format(res[1])
        res[3] = self.addr_fmt.format(res[3])
        res[4] = self.addr_fmt.format(res[4])
        for idx, r in enumerate(res):
            res[idx] = str(r)
        return res

    def is_supervisor_entry(self):
        return not bool(MMU.extract_bits(self.lower_flags, 4, 1))

    # Lower attributes (Block and Pages)
    def extract_attributes(self):
        return MMU.extract_bits(self.lower_flags, 0, 3)

    def is_secure_entry(self):
        return not bool(MMU.extract_bits(self.lower_flags, 3, 1))

    def extract_permissions(self):
        return MMU.extract_bits(self.lower_flags, 4, 2)

    def extract_shareability(self):
        return MMU.extract_bits(self.lower_flags, 6, 2)

    def is_accessed_entry(self):
        return bool(MMU.extract_bits(self.lower_flags, 8, 1))

    def is_global_entry(self):
        return not bool(MMU.extract_bits(self.lower_flags, 9, 1))

    def is_block_entry(self):
        return not bool(MMU.extract_bits(self.lower_flags, 14, 1))

    # Upper attributes (Block and Pages)
    def is_guarded_entry(self):
        return bool(MMU.extract_bits(self.upper_flags, 0, 1))

    def is_dirty_entry(self):
        return bool(MMU.extract_bits(self.upper_flags, 1, 1))

    def is_continuous_entry(self):
        return bool(MMU.extract_bits(self.upper_flags, 2, 1))

    def is_kernel_executable_entry(self):
        return not bool(MMU.extract_bits(self.upper_flags, 3, 1))

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.upper_flags, 4, 1))

    @staticmethod
    def extract_lower_flags(entry):
        return MMU.extract_bits(entry, 2, 10)

    @staticmethod
    def extract_upper_flags(entry):
        return MMU.extract_bits(entry, 50, 14)

    def get_permissions(self):
        permissions = self.extract_permissions()
        u = bool(permissions & 0x1)
        w = not bool(permissions & 0x2)
        return (
            True,
            w,
            self.is_kernel_executable_entry(),
            u,
            u and w,
            self.is_executable_entry(),
        )


class PTP(TEntry64):
    entry_name = "PTP"
    size = 0

    def is_supervisor_entry(self):
        return bool(MMU.extract_bits(self.lower_flags, 11, 1))

    # Lower attributes
    def extract_attributes(self):
        return 0

    def is_secure_entry(self):
        return not bool(MMU.extract_bits(self.upper_flags, 13, 1))

    def extract_permissions(self):
        return MMU.extract_bits(self.upper_flags, 11, 2)

    def extract_shareability(self):
        return 0

    def is_accessed_entry(self):
        return False

    def is_global_entry(self):
        return False

    def is_block_entry(self):
        return False

    # Upper attributes
    def is_guarded_entry(self):
        return False

    def is_dirty_entry(self):
        return False

    def is_continuous_entry(self):
        return False

    def is_kernel_executable_entry(self):
        return not bool(MMU.extract_bits(self.upper_flags, 9, 1))

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.upper_flags, 10, 1))

    def get_permissions(self):
        permissions = self.extract_permissions()
        u = not bool(permissions & 0x1)
        w = not bool(permissions & 0x2)

        return (
            True,
            w,
            self.is_kernel_executable_entry(),
            u,
            u and w,
            self.is_executable_entry(),
        )


# Page table pointers
class PTP_4KB(PTP):
    entry_name = "PTP_4KB"

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 12, 36) << 12


class PTP_4KB_L0(PTP_4KB):
    entry_name = "PTP_4KB_L0"


class PTP_4KB_L1(PTP_4KB):
    entry_name = "PTP_4KB_L1"


class PTP_4KB_L2(PTP_4KB):
    entry_name = "PTP_4KB_L2"


class PTP_16KB(PTP):
    entry_name = "PTP_16KB"

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 14, 34) << 14


class PTP_16KB_L0(PTP_16KB):
    entry_name = "PTP_16KB_L0"


class PTP_16KB_L1(PTP_16KB):
    entry_name = "PTP_16KB_L1"


class PTP_16KB_L2(PTP_16KB):
    entry_name = "PTP_16KB_L2"


class PTP_64KB(PTP):
    entry_name = "PTP_64KB"

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 16, 32) << 16


class PTP_64KB_L0(PTP_64KB):
    entry_name = "PTP_64KB_L0"


class PTP_64KB_L1(PTP_64KB):
    entry_name = "PTP_64KB_L1"


# Blocks
class PTBLOCK_L1_4KB(TEntry64):
    entry_name = "PTBLOCK_L1_4KB"
    size = 1024 * 1024 * 1024

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 30, 18) << 30


class PTBLOCK_L2_4KB(TEntry64):
    entry_name = "PTBLOCK_L2_4KB"
    size = 2 * 1024 * 1024

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 21, 27) << 21


class PTBLOCK_L2_16KB(TEntry64):
    entry_name = "PTBLOCK_L2_16KB"
    size = 32 * 1024 * 1024

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 25, 23) << 25


class PTBLOCK_L2_64KB(TEntry64):
    entry_name = "PTBLOCK_L2_64KB"
    size = 512 * 1024 * 1024

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 29, 19) << 29


# Page pointers
class PTPAGE_4KB(TEntry64):
    entry_name = "PTPAGE_4KB"
    size = 4 * 1024

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 12, 36) << 12


class PTPAGE_16KB(TEntry64):
    entry_name = "PTPAGE_16KB"
    size = 16 * 1024

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 14, 34) << 14


class PTPAGE_64KB(TEntry64):
    entry_name = "PTPAGE_64KB"
    size = 64 * 1024

    @staticmethod
    def extract_addr(entry):
        return (
            MMU.extract_bits(entry, 12, 4) << 48 | MMU.extract_bits(entry, 16, 32) << 16
        )


class ReservedEntry(TEntry64):
    entry_name = "Reserved"
    size = 0


class PageTableAArch64(PageTable):
    entry_size = 8
    table_fields = [
        "Entry address",
        "Pointed address",
        "Attributes",
        "Secure",
        "Permsissions",
        "Shareability",
        "Accessed",
        "Global",
        "Block",
        "Guarded",
        "Dirty",
        "Continous",
        "Kernel exec",
        "Exec",
        "Classes",
    ]
    addr_fmt = "0x{:016x}"

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
        return CPUAArch64(cpu_config)

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


class CPUAArch64(CPU):
    def __init__(self, features):
        super(CPUAArch64, self).__init__(features)
        self.processor_features["opcode_to_mmu_regs"] = {
            (1, 0, 2, 0, 2): "TCR_EL1",
            (1, 5, 2, 0, 2): "TCR_EL1",
            (1, 6, 2, 0, 2): "TCR_EL3",
            (1, 0, 2, 0, 0): "TTBR0_EL1",
            (1, 5, 2, 0, 0): "TTBR0_EL1",
            (1, 0, 2, 0, 1): "TTBR1_EL1",
            (1, 5, 2, 0, 1): "TTBR1_EL1",
            (1, 0, 5, 2, 0): "ESR_EL1",  # Read
            (1, 0, 6, 0, 0): "FAR_EL1",  # Read
            (1, 5, 4, 0, 1): "ELR_EL1",  # Read
            (1, 0, 1, 0, 0): "SCTLR_EL1",  # R/W
            (1, 5, 1, 0, 0): "SCTLR_EL1",
            (1, 6, 1, 0, 0): "SCTLR_EL3",
            (1, 4, 4, 1, 0): "SP_EL1",  # Write
            (1, 5, 4, 0, 0): "SPSR_EL1",  # R/W
            (1, 6, 4, 0, 0): "SPSR_EL3",
            (1, 0, 12, 0, 0): "VBAR_EL1",  # Write
            (1, 6, 12, 0, 0): "VBAR_EL3",  # Write
            (1, 0, 13, 0, 1): "CONTEXTIDR_EL1",  # R/W
            (1, 0, 0, 7, 0): "ID_AA64MMFR0_EL1",  # Read
        }
        self.processor_features["opcode_to_gregs"] = [
            "X{}".format(i) for i in range(31)
        ]

        CPU.processor_features = self.processor_features
        CPU.registers_values = self.registers_values

    def parse_opcode(self, instr, page_addr, offset):
        # Collect locations of opcodes

        # RET and ERET
        if (
            CPUAArch64.extract_bits(instr, 0, 5) == 0
            and CPUAArch64.extract_bits(instr, 10, 22) == 0b1101011001011111000000
        ):
            return {page_addr + offset: {"register": "", "instruction": "RET"}}

        elif (
            CPUAArch64.extract_bits(instr, 0, 32) == 0b11010110100111110000001111100000
        ):
            return {page_addr + offset: {"register": "", "instruction": "ERET"}}

        # BLR/BR
        elif (
            CPUAArch64.extract_bits(instr, 0, 5) == 0b00000
            and CPUAArch64.extract_bits(instr, 22, 10) == 0b1101011000
            and CPUAArch64.extract_bits(instr, 10, 11) == 0b11111000000
        ):
            return {page_addr + offset: {"register": "", "instruction": "BLR"}}

        # MSR opcode for MMU registers (write on MMU register)
        elif CPUAArch64.extract_bits(instr, 20, 12) == 0b110101010001:
            rt = CPUAArch64.extract_bits(instr, 0, 5)
            if rt > 30:  # No X31
                return {}

            op0 = CPUAArch64.extract_bits(instr, 19, 1)
            op1 = CPUAArch64.extract_bits(instr, 16, 3)
            crn = CPUAArch64.extract_bits(instr, 12, 4)
            crm = CPUAArch64.extract_bits(instr, 8, 4)
            op2 = CPUAArch64.extract_bits(instr, 5, 3)

            reg_idx = (op0, op1, crn, crm, op2)
            if reg_idx in self.processor_features["opcode_to_mmu_regs"]:
                mmu_regs = self.processor_features["opcode_to_mmu_regs"][reg_idx]
                rt = self.processor_features["opcode_to_gregs"][rt]
                return {
                    page_addr
                    + offset: {
                        "register": mmu_regs,
                        "gpr": [rt],
                        "f_addr": -1,
                        "instruction": "MSR",
                    }
                }

        # MRS opcode for MMU registers (read from MMU register)
        elif CPUAArch64.extract_bits(instr, 20, 12) == 0b110101010011:
            rt = CPUAArch64.extract_bits(instr, 0, 5)
            if rt > 30:  # No X31
                return {}

            op0 = CPUAArch64.extract_bits(instr, 19, 1)
            op1 = CPUAArch64.extract_bits(instr, 16, 3)
            crn = CPUAArch64.extract_bits(instr, 12, 4)
            crm = CPUAArch64.extract_bits(instr, 8, 4)
            op2 = CPUAArch64.extract_bits(instr, 5, 3)

            reg_idx = (op0, op1, crn, crm, op2)
            if reg_idx in self.processor_features["opcode_to_mmu_regs"]:
                mmu_regs = self.processor_features["opcode_to_mmu_regs"][reg_idx]
                rt = self.processor_features["opcode_to_gregs"][rt]

                return {
                    page_addr
                    + offset: {
                        "register": mmu_regs,
                        "gpr": [rt],
                        "f_addr": -1,
                        "instruction": "MRS",
                    }
                }
        else:
            return {}
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

                    # RETAA/RETAB/RET/ERET/B/BR
                    if asmcode.name in ["RETAA", "RETAB", "RET", "ERET", "B", "BR"]:
                        cur_addr += instr_len
                        break

                    # AARCH64 has a different ways to return from a routine, here the most used
                    # MOV PC, ....
                    elif asmcode.name == "MOV":
                        if asmcode.arg2str(asmcode.args[0]) == "PC":
                            cur_addr += instr_len
                            break

                    # JMPs and special opcodes
                    elif asmcode.name in [
                        "BL",
                        "BLR",
                        "BRK",
                        "HLT",
                        "HVC",
                        "SMC",
                        "SVC",
                        "DCPS1",
                        "DCPS2",
                        "DCPS3",
                        "DRPS",
                        "WFE",
                        "WFI",
                    ]:
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
        mn_s = "aarch64" + ("b" if self.cpu.endianness == "big" else "l")
        return MIASMMachine(mn_s)


#################################################################
# MMU Modes
#################################################################


class MMU(MMURadix):
    PAGE_SIZE = 0

    paging_unpack_format = "<Q"
    page_table_class = PageTableAArch64
    radix_levels = {}
    top_prefix = 0
    entries_size = 8

    def __init__(self, mmu_config):
        super(MMU, self).__init__(mmu_config)
        self.mmu_endianness = mmu_config.get("endianness", "little")
        if self.mmu_endianness == "little":
            MMU.extract_bits = MMURadix.extract_bits_little
        else:
            MMU.extract_bits = MMURadix.extract_bits_big


class LONG(MMU):
    tcr = None
    map_ptr_entries_to_levels = {"kernel": [], "user": []}
    map_datapages_entries_to_levels = {"kernel": [], "user": []}
    map_level_to_table_size = {"kernel": [], "user": []}
    map_entries_to_shifts = {"kernel": [], "user": []}
    map_reserved_entries_to_levels = {"kernel": [], "user": []}

    def reconstruct_table(
        self,
        mode,
        frame_addr,
        frame_size,
        table_levels,
        table_size,
        table_entries,
        empty_entries,
    ):
        # Reconstruct table_levels tables, empty tables and data_pages of a given size
        frame_d = defaultdict(dict)
        page_tables = defaultdict(dict)
        empty_tables = []
        data_pages = []
        table_levels = set(table_levels)

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
            pt_classes = set(
                self.classify_frame(
                    frame_d, empty_count, int(table_size // MMU.entries_size), mode=mode
                )
            )

            if -1 in pt_classes:  # Empty
                empty_tables.append(table_addr)
            elif -2 in pt_classes:  # Data
                data_pages.append(table_addr)
            elif table_levels.intersection(pt_classes):
                levels = sorted(table_levels.intersection(pt_classes))
                table_obj = self.page_table_class(
                    table_addr, table_size, deepcopy(frame_d), levels
                )
                for level in levels:
                    page_tables[level][table_addr] = table_obj
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
        # parse all the records in a frame of 64KB and reconstruct all the different tables

        # Prepare thread local dictionaries in which collect data
        data_pages = {"user": [], "kernel": []}
        empty_tables = {"user": [], "kernel": []}
        page_tables = {
            "user": [{} for i in range(self.radix_levels["user"])],
            "kernel": [{} for i in range(self.radix_levels["kernel"])],
        }

        tcr = kwargs["tcr"]
        trees_struct = tcr.get_trees_struct()

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

            # Reconstruct kernel tables
            self.reconstruct_all_tables(
                "kernel",
                trees_struct,
                frame_addr,
                frame_size,
                table_entries,
                empty_entries,
                page_tables,
                data_pages,
                empty_tables,
            )

            # Reconstruct user tables only if the radix tree have a different shape
            if trees_struct["kernel"] != trees_struct["user"]:
                self.reconstruct_all_tables(
                    "user",
                    trees_struct,
                    frame_addr,
                    frame_size,
                    table_entries,
                    empty_entries,
                    page_tables,
                    data_pages,
                    empty_tables,
                )

        return page_tables, data_pages, empty_tables

    def reconstruct_all_tables(
        self,
        mode,
        tree_struct,
        frame_addr,
        frame_size,
        table_entries,
        empty_entries,
        page_tables,
        data_pages,
        empty_tables,
    ):
        granule = tree_struct[mode]["granule"]
        total_levels = tree_struct[mode]["total_levels"]
        top_table_size = tree_struct[mode]["top_table_size"]

        # Top table has a different size, must be parsed separately
        if granule != top_table_size:
            candidate_levels = list(range(1, total_levels))
            t, _, _ = self.reconstruct_table(
                mode,
                frame_addr,
                frame_size,
                [0],
                top_table_size,
                table_entries,
                empty_entries,
            )
            page_tables[mode][0].update(t[0])
        else:
            candidate_levels = list(range(total_levels))

        # Look for other levels
        t, d, e = self.reconstruct_table(
            mode,
            frame_addr,
            frame_size,
            candidate_levels,
            granule,
            table_entries,
            empty_entries,
        )
        for level in t:
            page_tables[mode][level].update(t[level])
        data_pages[mode].extend(d)
        empty_tables[mode].extend(e)

    def classify_entry(self, page_addr, entry):
        classification = []
        class_bits = MMU.extract_bits(entry, 0, 2)

        # BITS 0,1 determine the class
        if class_bits == 0b00:
            return [False]

        # Block or RESERVED for PTL2
        elif class_bits == 0b01:
            # For L2 tables this type of entry is RESERVED and treated as EMPTY
            classification.append(ReservedEntry(0, 0, 0))

            # SH bits has one configuration reserved (0b01)
            # At least 17:20 must be 0
            if (
                MMU.extract_bits(entry, 8, 2) != 0b01
                and not MMU.extract_bits(entry, 12, 4)
                and not MMU.extract_bits(entry, 17, 4)
            ):
                lower_flags = TEntry64.extract_lower_flags(entry)
                upper_flags = TEntry64.extract_upper_flags(entry)

                # Different length of the address could correspond to different granule size and level
                addr = PTBLOCK_L2_4KB.extract_addr(entry)
                classification.append(PTBLOCK_L2_4KB(addr, lower_flags, upper_flags))

                if not MMU.extract_bits(entry, 17, 8):
                    addr = PTBLOCK_L2_16KB.extract_addr(addr)
                    classification.append(
                        PTBLOCK_L2_16KB(addr, lower_flags, upper_flags)
                    )

                if not MMU.extract_bits(entry, 17, 12):
                    addr = PTBLOCK_L2_64KB.extract_addr(addr)
                    classification.append(
                        PTBLOCK_L2_64KB(addr, lower_flags, upper_flags)
                    )

                if not MMU.extract_bits(entry, 17, 13):
                    addr = PTBLOCK_L1_4KB.extract_addr(addr)
                    classification.append(
                        PTBLOCK_L1_4KB(addr, lower_flags, upper_flags)
                    )

        # Page or Pointer
        else:
            # Lower flags for PTP are ignored so we insert 0
            upper_flags = TEntry64.extract_upper_flags(entry)

            # Different length of the address could correspond to different granule size
            addr = PTP_4KB.extract_addr(entry)
            if addr in self.machine.memory.physpace["ram"]:
                classification.append(PTP_4KB_L0(addr, 0, upper_flags))
                classification.append(PTP_4KB_L1(addr, 0, upper_flags))
                classification.append(PTP_4KB_L2(addr, 0, upper_flags))

            if not MMU.extract_bits(entry, 12, 2):
                addr = PTP_16KB.extract_addr(entry)
                if addr in self.machine.memory.physpace["ram"]:
                    classification.append(PTP_16KB_L0(addr, 0, upper_flags))
                    classification.append(PTP_16KB_L1(addr, 0, upper_flags))
                    classification.append(PTP_16KB_L2(addr, 0, upper_flags))

            if not MMU.extract_bits(entry, 12, 4):
                addr = PTP_64KB.extract_addr(entry)
                if addr in self.machine.memory.physpace["ram"]:
                    classification.append(PTP_64KB_L0(addr, 0, upper_flags))
                    classification.append(PTP_64KB_L1(addr, 0, upper_flags))

            # SH bits has one configuration reserved (0b01)
            if MMU.extract_bits(entry, 8, 2) != 0b01:
                lower_flags = TEntry64.extract_lower_flags(entry)

                addr = PTPAGE_4KB.extract_addr(entry)
                classification.append(PTPAGE_4KB(addr, lower_flags, upper_flags))

                addr = PTPAGE_64KB.extract_addr(entry)
                classification.append(PTPAGE_64KB(addr, lower_flags, upper_flags))

                if not MMU.extract_bits(entry, 12, 2):
                    addr = PTPAGE_16KB.extract_addr(entry)
                    classification.append(PTPAGE_16KB(addr, lower_flags, upper_flags))

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
                page_tables={"user": defaultdict(dict), "kernel": defaultdict(dict)},
                data_pages={"user": [], "kernel": []},
                empty_tables={"user": [], "kernel": []},
                reverse_map_tables={"user": None, "kernel": None},
                reverse_map_pages={"user": None, "kernel": None},
                used_tcr=None,
                ttbrs=defaultdict(dict),
            )

    def reload_data_from_file(self, data_filename):
        super(MMUShell, self).reload_data_from_file(data_filename)

        # Reload TCR data and radix tree shape
        LONG.tcr = self.data.used_tcr
        self.do_set_tcr(str(LONG.tcr.value))

    def do_find_registers_values(self, arg):
        """Find MMU load opcodes and execute MMU related functions inside the memory dump in order to extract MMU registers values"""

        if self.data.is_registers_found:
            logging.warning("Registers already searched")
            return

        logger.info("Look for opcodes related to MMU setup...")
        parallel_results = self.machine.apply_parallel(
            65536, self.machine.cpu.parse_opcodes_parallel
        )

        opcodes = {}
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            opcodes.update(result.get())

        self.data.opcodes = opcodes

        # Filter to look only for opcodes which write on MMU register only and not read from them or from other registers
        filter_f = (
            lambda it: True
            if it[1]["register"] == "TCR_EL1" and it[1]["instruction"] == "MSR"
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
                reg_obj = CPURegAArch64.get_register_obj(register, value)
                if reg_obj.valid and not any(
                    [
                        val_obj.is_mmu_equivalent_to(reg_obj)
                        for val_obj in filtered_values[register]
                    ]
                ):
                    filtered_values[register].add(reg_obj)

        self.data.regs_values = filtered_values
        self.data.is_registers_found = True

        # Show results
        logging.info("TCR_EL1 values recovered:")
        for reg_obj in self.data.regs_values["TCR_EL1"]:
            logging.info(reg_obj)

    def do_show_registers(self, args):
        """Show TCR values found"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        for registers in sorted(self.data.regs_values.keys()):
            for register in self.data.regs_values[registers]:
                print(register)

    def do_set_tcr(self, args):
        """Set TCR to be used"""
        args = args.split()
        if len(args) == 0:
            logging.error("Please use find_tables TCR_VALUE")
            return

        try:
            tcr_val = self.parse_int(args[0])
            tcr = TCR_EL1(tcr_val)
            if not tcr.valid:
                raise ValueError
        except ValueError:
            logger.warning("Invalid TCR value")
            return

        self.data.used_tcr = tcr

        # Set all MMU parameters
        LONG.tcr = tcr
        LONG.radix_levels = {}
        trees_struct = tcr.get_trees_struct()

        for mode in ["user", "kernel"]:
            granule = trees_struct[mode]["granule"]
            total_levels = trees_struct[mode]["total_levels"]
            top_table_size = trees_struct[mode]["top_table_size"]

            LONG.radix_levels[mode] = total_levels
            LONG.map_level_to_table_size[mode] = [top_table_size] + (
                [granule] * (total_levels - 1)
            )
            LONG.map_reserved_entries_to_levels[mode] = [
                [] for i in range(total_levels - 1)
            ] + [[ReservedEntry]]

            if granule == 4096:
                if total_levels == 1:
                    LONG.map_datapages_entries_to_levels[mode] = [PTPAGE_4KB]
                    LONG.map_ptr_entries_to_levels[mode] = [None]
                    LONG.map_entries_to_shifts[mode] = {PTPAGE_4KB: 12}
                elif total_levels == 2:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [PTBLOCK_L2_4KB],
                        [PTPAGE_4KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [PTP_4KB_L0, None]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_4KB_L0: 21,
                        PTPAGE_4KB: 12,
                        PTBLOCK_L2_4KB: 21,
                    }
                elif total_levels == 3:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [PTBLOCK_L1_4KB],
                        [PTBLOCK_L2_4KB],
                        [PTPAGE_4KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [
                        PTP_4KB_L0,
                        PTP_4KB_L1,
                        None,
                    ]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_4KB_L0: 30,
                        PTP_4KB_L1: 21,
                        PTPAGE_4KB: 12,
                        PTBLOCK_L2_4KB: 21,
                        PTBLOCK_L1_4KB: 30,
                    }
                else:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [None],
                        [PTBLOCK_L1_4KB],
                        [PTBLOCK_L2_4KB],
                        [PTPAGE_4KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [
                        PTP_4KB_L0,
                        PTP_4KB_L1,
                        PTP_4KB_L2,
                        None,
                    ]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_4KB_L0: 39,
                        PTP_4KB_L1: 30,
                        PTP_4KB_L2: 21,
                        PTPAGE_4KB: 12,
                        PTBLOCK_L2_4KB: 21,
                        PTBLOCK_L1_4KB: 30,
                    }

            elif granule == 16384:
                if total_levels == 1:
                    LONG.map_datapages_entries_to_levels[mode] = [PTPAGE_16KB]
                    LONG.map_ptr_entries_to_levels[mode] = [None]
                    LONG.map_entries_to_shifts[mode] = {PTPAGE_16KB: 14}
                elif total_levels == 2:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [PTBLOCK_L2_16KB],
                        [PTPAGE_16KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [PTP_16KB_L0, None]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_16KB_L0: 25,
                        PTPAGE_16KB: 14,
                        PTBLOCK_L2_16KB: 25,
                    }
                elif total_levels == 3:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [None],
                        [PTBLOCK_L2_16KB],
                        [PTPAGE_16KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [
                        PTP_16KB_L0,
                        PTP_16KB_L1,
                        None,
                    ]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_16KB_L0: 36,
                        PTP_16KB_L1: 25,
                        PTPAGE_16KB: 14,
                        PTBLOCK_L2_16KB: 25,
                    }
                else:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [None],
                        [None],
                        [PTBLOCK_L2_16KB],
                        [PTPAGE_16KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [
                        PTP_16KB_L0,
                        PTP_16KB_L2,
                        None,
                    ]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_16KB_L0: 47,
                        PTP_16KB_L1: 36,
                        PTP_16KB_L2: 25,
                        PTPAGE_16KB: 14,
                        PTBLOCK_L2_16KB: 25,
                    }

            else:
                if total_levels == 1:
                    LONG.map_datapages_entries_to_levels[mode] = [PTPAGE_64KB]
                    LONG.map_ptr_entries_to_levels[mode] = [None]
                    LONG.map_entries_to_shifts[mode] = {
                        PTPAGE_64KB: 16,
                        PTBLOCK_L2_64KB: 29,
                    }
                elif total_levels == 2:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [PTBLOCK_L2_64KB],
                        [PTPAGE_64KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [PTP_64KB_L0, None]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_64KB_L0: 29,
                        PTPAGE_64KB: 16,
                        PTBLOCK_L2_16KB: 29,
                    }
                else:
                    LONG.map_datapages_entries_to_levels[mode] = [
                        [None],
                        [PTBLOCK_L2_64KB],
                        [PTPAGE_64KB],
                    ]
                    LONG.map_ptr_entries_to_levels[mode] = [
                        PTP_64KB_L0,
                        PTP_64KB_L1,
                        None,
                    ]
                    LONG.map_entries_to_shifts[mode] = {
                        PTP_64KB_L0: 42,
                        PTP_64KB_L1: 29,
                        PTPAGE_64KB: 16,
                        PTBLOCK_L2_64KB: 29,
                    }

    def do_find_tables(self, args):
        """Find MMU tables in memory"""
        if not self.data.used_tcr:
            logging.error("Please set a TCR register to use, using set_tcr TCR")
            return
        tcr = self.data.used_tcr

        # Delete all the previous table data
        if self.data.is_tables_found:
            self.data.page_tables = {
                "user": defaultdict(dict),
                "kernel": defaultdict(dict),
            }
            self.data.data_pages = {"user": [], "kernel": []}
            self.data.empty_tables = {"user": [], "kernel": []}
            self.data.reverse_map_tables = {}
            self.data.reverse_map_pages = {}

        # WORKAROUND: initialize here because unpickable!
        self.data.reverse_map_pages = {
            "kernel": defaultdict(_dummy_f),
            "user": defaultdict(_dummy_f),
        }
        self.data.reverse_map_tables = {
            "kernel": defaultdict(_dummy_f),
            "user": defaultdict(_dummy_f),
        }

        # Parse memory in chunk of 64KiB
        logger.info("Look for paging tables...")
        parallel_results = self.machine.apply_parallel(
            65536, self.machine.mmu.parse_parallel_frame, tcr=tcr
        )
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            page_tables, data_pages, empty_tables = result.get()

            for mode in ["user", "kernel"]:
                for level in range(self.machine.mmu.radix_levels[mode]):
                    self.data.page_tables[mode][level].update(page_tables[mode][level])

                self.data.data_pages[mode].extend(data_pages[mode])
                self.data.empty_tables[mode].extend(empty_tables[mode])

        for mode in ["user", "kernel"]:
            self.data.data_pages[mode] = set(self.data.data_pages[mode])
            self.data.empty_tables[mode] = set(self.data.empty_tables[mode])

        # Remove all tables which point to inexistent table of lower level
        logger.info("Reduce false positives...")
        for mode in ["user", "kernel"]:
            for lvl in range(self.machine.mmu.radix_levels[mode] - 1):
                ptr_class = self.machine.mmu.map_ptr_entries_to_levels[mode][lvl]
                referenced_nxt = []
                for table_addr in list(self.data.page_tables[mode][lvl].keys()):
                    for entry_obj in (
                        self.data.page_tables[mode][lvl][table_addr]
                        .entries[ptr_class]
                        .values()
                    ):
                        if (
                            entry_obj.address
                            not in self.data.page_tables[mode][lvl + 1]
                            and entry_obj.address not in self.data.empty_tables[mode]
                        ):
                            # Remove the table
                            self.data.page_tables[mode][lvl].pop(table_addr)
                            break

                        else:
                            referenced_nxt.append(entry_obj.address)

                # Remove table not referenced by upper levels
                referenced_nxt = set(referenced_nxt)
                for table_addr in set(
                    self.data.page_tables[mode][lvl + 1].keys()
                ).difference(referenced_nxt):
                    self.data.page_tables[mode][lvl + 1].pop(table_addr)

        logger.info("Fill reverse maps...")
        for mode in ["user", "kernel"]:
            for lvl in range(0, self.machine.mmu.radix_levels[mode]):
                ptr_class = self.machine.mmu.map_ptr_entries_to_levels[mode][lvl]
                page_classes = self.machine.mmu.map_datapages_entries_to_levels[mode][
                    lvl
                ]
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

        # If kernel and user space use the same configuration, copy kernel data to user
        trees_struct = tcr.get_trees_struct()
        if trees_struct["kernel"] == trees_struct["user"]:
            self.data.page_tables["user"] = self.data.page_tables["kernel"]
            self.data.reverse_map_pages["user"] = self.data.reverse_map_pages["kernel"]
            self.data.reverse_map_tables["user"] = self.data.reverse_map_tables[
                "kernel"
            ]
            self.data.data_pages["user"] = self.data.data_pages["kernel"]
            self.data.empty_tables["user"] = self.data.empty_tables["kernel"]

        self.data.is_tables_found = True

    def do_show_table(self, args):
        """Show MMU table at chosen address. Usage: show_table ADDRESS (user, kernel) [level size]"""
        if not self.data.used_tcr:
            logging.error("Please set a TCR register to use, using set_tcr TCR")
            return

        args = args.split()
        if len(args) < 2:
            logger.warning("Missing argument")
            return

        try:
            addr = self.parse_int(args[0])
        except ValueError:
            logger.warning("Invalid table address")
            return

        if addr not in self.machine.memory:
            logger.warning("Table not in RAM range")
            return

        args[1] = args[1].lower()
        if args[1] not in ["kernel", "user"]:
            logger.warning("Mode must be kernel or user")
            return
        mode = args[1]

        if len(args) == 4:
            try:
                lvl = self.parse_int(args[2])
                if lvl > (self.machine.mmu.radix_levels[mode] - 1):
                    raise ValueError
            except ValueError:
                logger.warning(
                    f"Level must be an integer between 0 and {self.machine.mmu.radix_levels[mode] - 1}"
                )
                return

            trees_struct = LONG.tcr.get_trees_struct()
            valid_sizes = {"user": defaultdict(set), "kernel": defaultdict(set)}
            valid_sizes["kernel"][0].add(trees_struct["kernel"]["top_table_size"])
            valid_sizes["user"][0].add(trees_struct["user"]["top_table_size"])
            for i in range(1, trees_struct["kernel"]["total_levels"]):
                valid_sizes["kernel"][i].add(trees_struct["kernel"]["granule"])
            for i in range(1, trees_struct["user"]["total_levels"]):
                valid_sizes["user"][i].add(trees_struct["user"]["granule"])

            try:
                table_size = self.parse_int(args[3])
                if table_size not in valid_sizes[mode][lvl]:
                    logging.warning(
                        f"Size not allowed for choosen level! Valid sizes are:{valid_sizes[mode][lvl]}"
                    )
                    return
            except ValueError:
                logger.warning("Invalid size value")
                return
        else:
            table_size = 0x10000
            lvl = -1

        table_buff = self.machine.memory.get_data(addr, table_size)
        invalids, pt_classes, table_obj = self.machine.mmu.parse_frame(
            table_buff, addr, table_size, lvl, mode=mode
        )
        print(table_obj)
        print(f"Invalid entries: {invalids} Table levels: {pt_classes}")

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
        if not len(self.data.page_tables["kernel"][0]) and not len(
            self.data.page_tables["user"][0]
        ):
            logger.warning("OOPS... no tables in first level... Wrong MMU mode?")
            return

        ttbrs_candidates = {"kernel": [], "user": []}
        trees_struct = self.machine.mmu.tcr.get_trees_struct()

        # Collect opcodes
        opcode_classes = defaultdict(list)
        for opcode_addr, opcode_data in self.data.opcodes.items():
            opcode_classes[
                (opcode_data["instruction"], opcode_data["register"])
            ].append(opcode_addr)

        # Find all TTBR1_EL1 which contain interrupt related opcodes
        logging.info("Find TTBR1_EL1 candidates...")
        int_opcode_addrs = (
            opcode_classes[("MRS", "ESR_EL1")]
            + opcode_classes[("MRS", "FAR_EL1")]
            + opcode_classes[("MRS", "ELR_EL1")]
        )
        already_explored = set()
        for opcode_addr in int_opcode_addrs:
            derived_addresses = self.machine.mmu.derive_page_address(
                opcode_addr, mode="kernel"
            )
            for derived_address in derived_addresses:
                if derived_address in already_explored:
                    continue

                lvl, addr = derived_address
                ttbrs_candidates["kernel"].extend(
                    self.radix_roots_from_data_page(
                        lvl,
                        addr,
                        self.data.reverse_map_pages["kernel"],
                        self.data.reverse_map_tables["kernel"],
                    )
                )
                already_explored.add(derived_address)

        ttbrs_candidates["kernel"] = list(
            set(ttbrs_candidates["kernel"]).intersection(
                self.data.page_tables["kernel"][0].keys()
            )
        )

        # Filter kernel candidates for ERET and write on MMU registers
        logger.info("Filtering TTBR1_EL1 candidates...")
        mmu_w_opcode_addrs = (
            opcode_classes[("MSR", "TCR_EL1")] + opcode_classes[("MSR", "TTBR0_EL1")]
        )
        phy_cache = defaultdict(dict)
        ttbrs_filtered = {"kernel": {}, "user": {}}
        virt_cache = defaultdict(dict)
        for candidate in tqdm(ttbrs_candidates["kernel"]):
            # Calculate physpace and discard empty ones
            consistency, pas = self.physpace(
                candidate,
                self.data.page_tables["kernel"],
                self.data.empty_tables["kernel"],
                mode="kernel",
                hierarchical=True,
                cache=phy_cache,
            )

            # Discard inconsistent one
            if not consistency:
                continue

            # WARNING! We cannot filter for user_size = 0 due to TCR_EL1.E0PD1 !
            # Check if at least one MMU opcode in physical address space
            for opcode_addr in mmu_w_opcode_addrs:
                if pas.is_in_kernel_space(opcode_addr):
                    break
            else:
                continue

            # Check if at least one ERET opcode in physical address space
            for opcode_addr in opcode_classes[("ERET", "")]:
                if pas.is_in_kernel_space(opcode_addr):
                    break
            else:
                continue

            # At least a page must be writable by the kernel and not by user
            for perms in pas.space:
                if perms[1] and not perms[4]:
                    break
            else:
                continue

            vas = self.virtspace(
                candidate, mode="kernel", hierarchical=True, cache=virt_cache
            )
            radix_tree = RadixTree(
                candidate,
                trees_struct["kernel"]["total_levels"],
                pas,
                vas,
                kernel=True,
                user=False,
            )
            ttbrs_filtered["kernel"][candidate] = radix_tree

        # Find all TTBR0_EL1 which contain at least one RET instruction
        already_explored = set()
        virt_cache.clear()
        logging.info("Find TTBR0_EL1 candidates...")
        for opcode_addr in opcode_classes[("RET", "")]:
            derived_addresses = self.machine.mmu.derive_page_address(
                opcode_addr, mode="user"
            )
            for derived_address in derived_addresses:
                if derived_address in already_explored:
                    continue

                lvl, addr = derived_address
                ttbrs_candidates["user"].extend(
                    self.radix_roots_from_data_page(
                        lvl,
                        addr,
                        self.data.reverse_map_pages["user"],
                        self.data.reverse_map_tables["user"],
                    )
                )
                already_explored.add(derived_address)

        ttbrs_candidates["user"] = list(
            set(ttbrs_candidates["user"]).intersection(
                self.data.page_tables["user"][0].keys()
            )
        )

        logger.info("Filtering TTBR0_EL1 candidates...")
        phy_cache = defaultdict(dict)
        for candidate in tqdm(ttbrs_candidates["user"]):
            # Calculate physpace and discard empty ones
            consistency, pas = self.physpace(
                candidate,
                self.data.page_tables["user"],
                self.data.empty_tables["user"],
                mode="user",
                hierarchical=True,
                cache=phy_cache,
            )

            # Discard inconsistent one
            if not consistency:
                continue

            # At least a page must be R or W in usermode
            for perms in pas.space:
                if perms[3] or perms[4]:
                    break
            else:
                continue

            # Check if at least one BLR opcode in physical address space
            for opcode_addr in opcode_classes[("BLR", "")]:
                if opcode_addr in pas:
                    break
            else:
                continue

            vas = self.virtspace(
                candidate, mode="user", hierarchical=True, cache=virt_cache
            )
            radix_tree = RadixTree(
                candidate,
                trees_struct["user"]["total_levels"],
                pas,
                vas,
                kernel=False,
                user=True,
            )
            ttbrs_filtered["user"][candidate] = radix_tree

        self.data.ttbrs = ttbrs_filtered
        self.data.is_radix_found = True

    def do_show_radix_trees(self, args):
        """Show radix trees found"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        labels = [
            "Radix address",
            "Total levels",
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
        """Compare TCR values found with the ground truth"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        # Check if the last value of TCR was found
        all_tcrs = {}
        for reg_name, value_data in self.gtruth.items():
            if "TCR_EL1" in reg_name:
                for value, value_info in value_data.items():
                    if value not in all_tcrs or (value_info[1] > all_tcrs[value][1]):
                        all_tcrs[value] = (value_info[0], value_info[1])

        last_tcr = TCR_EL1(
            sorted(all_tcrs.keys(), key=lambda x: all_tcrs[x][1], reverse=True)[0]
        )

        tcr_fields_equals = {}
        for value_found_obj in self.data.regs_values["TCR_EL1"]:
            tcr_fields_equals[value_found_obj] = value_found_obj.count_fields_equals(
                last_tcr
            )
        k_sorted = sorted(
            tcr_fields_equals.keys(), key=lambda x: tcr_fields_equals[x], reverse=True
        )
        if not k_sorted:
            print(f"Correct TCR_EL1 value: {last_tcr}")
            print("TCR_EL1 fields found:... 0/4")
            print("FP: {}".format(str(len(self.data.regs_values["TCR_EL1"]))))
            return
        else:
            tcr_found = k_sorted[0]
            correct_fields_found = tcr_fields_equals[tcr_found]
            print(f"Correct TCR_EL1 value: {last_tcr}, Found: {tcr_found}")
            print("TCR_EL1 fields found:... {}/4".format(correct_fields_found))
            print("FP: {}".format(str(len(self.data.regs_values["TCR_EL1"]) - 1)))

    def do_show_radix_trees_gtruth(self, args):
        """Compare radix trees found with the ground truth"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        # Collect TTBR0 and TTBR1 values from the gtruth
        ttbr0s = {}
        ttbr1s = {}
        ttbr0_phy_cache = defaultdict(dict)
        ttbr1_phy_cache = defaultdict(dict)
        virt_cache = defaultdict(dict)

        # Collect opcodes
        opcode_classes = defaultdict(list)
        for opcode_addr, opcode_data in self.data.opcodes.items():
            opcode_classes[
                (opcode_data["instruction"], opcode_data["register"])
            ].append(opcode_addr)

        # Kernel radix trees
        # Filtering using the same criteria used by the algorithm, however we test only candidates which are possible
        # False Negatives beacuse the interection must always pass the check!
        mmu_w_opcode_addrs = (
            opcode_classes[("MSR", "TCR_EL1")]
            + opcode_classes[("MSR", "TTBR0_EL1")]
            + opcode_classes[("MSR", "TTBR1_EL1")]
        )

        kernel_radix_trees = (
            False  # Some AArch64 machines do not have TTBR1_EL1 but only TTBR0_EL1
        )
        for key in ["TTBR1_EL1", "TTBR1_EL1_S"]:
            for value, data in tqdm(self.gtruth.get(key, {}).items()):
                ttbr = TTBR1_EL1(value)

                try:
                    for addr_r, data_r in ttbr1s.items():
                        ttbr_r, dates_r = data_r
                        if ttbr.is_mmu_equivalent_to(ttbr_r):
                            if data[0] < dates_r[0]:
                                ttbr1s[addr_r][1][0] = data[0]
                            if data[1] > dates_r[1]:
                                ttbr1s[addr_r][1][1] = data[1]
                            raise UserWarning
                except UserWarning:
                    continue

                if ttbr.address not in self.data.page_tables["kernel"][0]:
                    continue

                ttbr1s[ttbr.address] = [ttbr, list(data)]
                kernel_radix_trees = True

        if kernel_radix_trees:
            tps = sorted(
                set(ttbr1s.keys()).intersection(set(self.data.ttbrs["kernel"].keys()))
            )
            fps = sorted(
                set(self.data.ttbrs["kernel"].keys()).difference(set(ttbr1s.keys()))
            )
            fns_candidates = set(ttbr1s.keys()).difference(
                set(self.data.ttbrs["kernel"].keys())
            )

            fns = []
            # Check False negatives
            for candidate in tqdm(fns_candidates):
                # Calculate physpace and discard empty ones
                consistency, pas = self.physpace(
                    candidate,
                    self.data.page_tables["kernel"],
                    self.data.empty_tables["kernel"],
                    mode="kernel",
                    hierarchical=True,
                    cache=ttbr1_phy_cache,
                )

                # Discard inconsistent one
                if not consistency:
                    continue

                # Check if at least one ERET opcode in physical address space
                for opcode_addr in opcode_classes[("ERET", "")]:
                    if pas.is_in_kernel_space(opcode_addr):
                        break
                else:
                    continue

                # WARNING! We cannot filter for user_size = 0 due to TCR_EL1.E0PD1 !
                # Check if at least one MMU opcode in physical address space
                for opcode_addr in mmu_w_opcode_addrs:
                    if pas.is_in_kernel_space(opcode_addr):
                        break
                else:
                    continue

                fns.append(candidate)
            fns.sort()

        # User radix trees
        for key in ["TTBR0_EL1", "TTBR0_EL1_S"]:
            for value, data in tqdm(self.gtruth.get(key, {}).items()):
                ttbr = TTBR0_EL1(value)

                try:
                    for addr_r, data_r in ttbr0s.items():
                        ttbr_r, dates_r = data_r
                        if ttbr.is_mmu_equivalent_to(ttbr_r):
                            if data[0] < dates_r[0]:
                                ttbr0s[addr_r][1][0] = data[0]
                            if data[1] > dates_r[1]:
                                ttbr0s[addr_r][1][1] = data[1]
                            raise UserWarning
                except UserWarning:
                    continue

                if ttbr.address not in self.data.page_tables["user"][0]:
                    continue

                ttbr0s[ttbr.address] = [ttbr, list(data)]

        # If not TTBR1 uses TTBR0 as TTBR0+TTBR1
        user_ttbrs = list(self.data.ttbrs["user"])
        if not kernel_radix_trees:
            user_ttbrs.extend(self.data.ttbrs["kernel"].keys())

        tpsu = sorted(set(ttbr0s.keys()).intersection(set(user_ttbrs)))
        fpsu = sorted(set(user_ttbrs).difference(set(ttbr0s.keys())))
        fnsu_candidates = set(ttbr0s.keys()).difference(set(user_ttbrs))

        # Filter FN
        fnsu = []
        for candidate in tqdm(fnsu_candidates):
            # Calculate physpace and discard empty ones
            consistency, pas = self.physpace(
                candidate,
                self.data.page_tables["user"],
                self.data.empty_tables["user"],
                mode="user",
                hierarchical=True,
                cache=ttbr0_phy_cache,
            )

            # Discard inconsistent one
            if not consistency:
                continue

            # At least a page must be R or W in usermode
            for perms in pas.space:
                if perms[3] or perms[4]:
                    break
            else:
                continue

            # Check if at least one BLR opcode in physical address space
            for opcode_addr in opcode_classes[("BLR", "")]:
                if opcode_addr in pas:
                    break
            else:
                continue

            # Check if at least one RET opcode in physical address space
            for opcode_addr in opcode_classes[("RET", "")]:
                if opcode_addr in pas:
                    break
            else:
                continue

            fnsu.append(candidate)
        fnsu.sort()

        # Show results
        table = PrettyTable()
        table.field_names = ["Address", "Found", "Mode", "First seen", "Last seen"]
        kernel_regs = ttbr1s

        if kernel_radix_trees:
            umode = "U"
            for tp in sorted(tps):
                table.add_row(
                    [hex(tp), "X", "K", kernel_regs[tp][1][0], kernel_regs[tp][1][1]]
                )

            for fn in sorted(fns):
                table.add_row(
                    [hex(fn), "", "K", kernel_regs[fn][1][0], kernel_regs[fn][1][1]]
                )

            for fp in sorted(fps):
                table.add_row([hex(fp), "False positive", "K", "", ""])
        else:
            umode = "K"

        # User
        for tp in sorted(tpsu):
            table.add_row([hex(tp), "X", umode, ttbr0s[tp][1][0], ttbr0s[tp][1][1]])

        for fn in sorted(fnsu):
            table.add_row([hex(fn), "", umode, ttbr0s[fn][1][0], ttbr0s[fn][1][1]])

        for fp in sorted(fpsu):
            table.add_row([hex(fp), "False positive", umode, "", ""])

        print(table)
        if kernel_radix_trees:
            print(f"TP:{len(tps)} FN:{len(fns)} FP:{len(fps)}")
            print(f"USER TP:{len(tpsu)} FN:{len(fnsu)} FP:{len(fpsu)}")
        else:
            print(f"TP:{len(tpsu)} FN:{len(fnsu)} FP:{len(fpsu)}")
