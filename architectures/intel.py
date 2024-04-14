from json import dump
from architectures.generic import Machine as MachineDefault
from architectures.generic import CPU as CPUDefault
from architectures.generic import PhysicalMemory as PhysicalMemoryDefault
from architectures.generic import MMUShell as MMUShellDefault
from architectures.generic import TableEntry, PageTable, MMURadix, PAS, RadixTree
from architectures.generic import CPUReg
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
from miasm.analysis.machine import Machine as MIASMMachine

# from IPython import embed

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
    idts: list
    cr3s: dict


class IDTR(CPUReg):
    def __init__(self, idtr):
        self.value = idtr
        self.address = MMU.extract_bits(idtr, 16, 64)
        self.size = MMU.extract_bits(idtr, 0, 16)
        self.valid = True

    def __repr__(self):
        return (
            f"IDTR: {hex(self.value)} => Address: {hex(self.address)}, Size:{self.size}"
        )


class CR3_32(CPUReg):
    def __init__(self, cr3):
        self.value = cr3
        self.pwt = bool(MMU.extract_bits(cr3, 3, 1))
        self.pcd = bool(MMU.extract_bits(cr3, 4, 1))
        self.address = MMU.extract_bits(cr3, 12, 20) << 12
        self.valid = True

    def __repr__(self):
        return f"CR3_32: {hex(self.value)} => PWT:{self.pwt}, PCD:{self.pcd}, Address: {hex(self.address)}"


class CR3_PAE(CPUReg):
    def __init__(self, cr3):
        self.value = cr3
        self.address = MMU.extract_bits(cr3, 5, 27) << 5
        self.valid = True

    def __repr__(self):
        return f"CR3_PAE: {hex(self.value)} => Address: {hex(self.address)}"


class CR3_64(CPUReg):
    def __init__(self, cr3):
        self.value = cr3
        self.pwt = bool(MMU.extract_bits(cr3, 3, 1))
        self.pcd = bool(MMU.extract_bits(cr3, 4, 1))
        self.address = MMU.extract_bits(cr3, 12, CPU.m_phy - 12) << 12
        self.valid = True

    def __repr__(self):
        return f"CR3_64: {hex(self.value)} => PWT:{self.pwt}, PCD:{self.pcd}, Address: {hex(self.address)}"


class Machine(MachineDefault):
    def get_miasm_machine(self):
        if self.cpu.bits == 32:
            return MIASMMachine("x86_32")
        else:
            return MIASMMachine("x86_64")


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

    def parse_idt(self, addr):
        entries = []
        IDTable = self.processor_features["idt_table_class"]
        for idx in range(256):
            entry_buff = self.machine.memory.get_data(
                addr + idx * self.processor_features["idt_entry_size"],
                self.processor_features["idt_entry_size"],
            )
            try:
                raw_entry = unpack(
                    self.processor_features["idt_unpack_format"], entry_buff
                )
            except Exception:
                break
            entry_idt = self.classify_idt_entry(raw_entry)

            if entry_idt is None:
                break
            else:
                entries.append(entry_idt)

        return IDTable(
            addr, len(entries) * self.processor_features["idt_entry_size"], entries
        )

    def find_idt_tables(self):
        # Look for IDT only in pointed pages
        idt_entry_size = self.processor_features["idt_entry_size"]

        # Workaround to reduce memory fingerprint
        iterators = [
            self.machine.memory.get_addresses(idt_entry_size, align_offset=i)
            for pidx, i in enumerate(range(0, idt_entry_size, 4))
        ]

        pool = mp.Pool(
            idt_entry_size // 4, initializer=tqdm.set_lock, initargs=(mp.Lock(),)
        )
        idt_candidates_async = [
            pool.apply_async(
                self.find_idt_tables_parallel, args=(iterators[pidx], pidx)
            )
            for pidx, i in enumerate(range(0, idt_entry_size, 4))
        ]

        pool.close()
        pool.join()

        idts = []
        for res in idt_candidates_async:
            idts.extend(res.get())

        print("\n")  # Workaround TQDM
        return idts

    def find_idt_tables_parallel(self, addresses_it, pidx):
        # Random sleep to desyncronize accesses to disk
        sleep(uniform(pidx, pidx + 1) // 1000)

        idt_candidates = []
        idt_under_analysis = deque(maxlen=256)
        mm = copy(self.machine.memory)
        mm.reopen()
        IDTable = self.processor_features["idt_table_class"]
        idt_entry_size = self.processor_features["idt_entry_size"]
        idt_unpack_format = self.processor_features["idt_unpack_format"]

        naddresses, addresses = addresses_it[1]
        for addr in tqdm(
            addresses, total=naddresses, unit="tables", position=-pidx, leave=False
        ):
            # parsing table machinery, we risk to lose some tables however...
            table_buff = mm.get_data(addr, idt_entry_size)

            if len(table_buff) < idt_entry_size:
                idt_under_analysis.clear()
                continue

            # Parse the entry
            try:
                raw_entry = unpack(idt_unpack_format, table_buff)
            except Exception:
                idt_under_analysis.clear()
                continue
            entry_idt = self.classify_idt_entry(raw_entry)

            # If the entry is invalid finalize the current IDT under analysis
            if entry_idt is None:
                if len(idt_under_analysis) > 0:
                    if self.validate_idt(idt_under_analysis):
                        idt_candidates.append(
                            IDTable(
                                addr - len(idt_under_analysis) * idt_entry_size,
                                len(idt_under_analysis) * idt_entry_size,
                                tuple(deepcopy(idt_under_analysis)),
                            )
                        )

                    idt_under_analysis.clear()
            else:
                # Check if the candidate has reach the maximum size
                if len(idt_under_analysis) == 256:
                    if self.validate_idt(idt_under_analysis):
                        idt_candidates.append(
                            IDTable(
                                addr - len(idt_under_analysis) * idt_entry_size,
                                len(idt_under_analysis) * idt_entry_size,
                                tuple(deepcopy(idt_under_analysis)),
                            )
                        )

                idt_under_analysis.append(entry_idt)

        return idt_candidates

    def classify_idt_entry(self, entry):
        raise NotImplementedError

    def validate_idt(self, candidate):
        raise NotImplementedError


class CPU32(CPU):
    def __init__(self, features):
        super(CPU32, self).__init__(features)
        self.processor_features["idt_entry_size"] = 8
        self.processor_features["idt_unpack_format"] = "<2I"
        self.processor_features["idt_table_class"] = IDTable32
        CPU.processor_features = self.processor_features

        # Check validity of m_phy
        m_phy = self.processor_features.get("m_phy", -1)
        if m_phy <= 0 or m_phy >= 52:
            logging.fatal(
                "m_phy must be positive and less then 40 in IA32 mode MMU modes"
            )
            exit(1)
        CPU.m_phy = m_phy

    def classify_idt_entry(self, entry):
        # 32bit IDT entries
        p = CPU.extract_bits(entry[1], 15, 1)
        offset = (CPU.extract_bits(entry[1], 16, 16) << 16) + CPU.extract_bits(
            entry[0], 0, 16
        )
        s_selector = CPU.extract_bits(entry[0], 16, 16)
        e_type = CPU.extract_bits(entry[1], 8, 5)
        dpl = CPU.extract_bits(entry[1], 13, 2)

        # If P = 0 the entry is empty
        if not p:
            return IDTEntry32(offset, s_selector, e_type, dpl, False)

        # BIT 0:7 and BIT 12 of Block 1 must be 0
        if CPU.extract_bits(entry[1], 0, 8) or CPU.extract_bits(entry[1], 12, 1):
            return None

        # BIT 10 Block 1 must be 1
        if not CPU.extract_bits(entry[1], 10, 1):
            return None

        # EURISTIC (almost) all operating systems using only RING 3 and 0
        if 0 < dpl < 3:
            return None

        # BIT 9 Block 1 if 0 is a TASK GATE and has other constrains
        if not CPU.extract_bits(entry[1], 9, 1):
            # I have to esclude a check on BITS 0:16 and 16:31 (which should be 0)
            # because some OSs write on them (Windows...)
            return IDTTaskEntry32(offset, s_selector, e_type, dpl, True)

        if e_type in [6, 14]:
            return IDTInterruptEntry32(offset, s_selector, e_type, dpl, True)
        else:
            return IDTTrapEntry32(offset, s_selector, e_type, dpl, True)

    def validate_idt(self, candidate):
        # Check if minimum interrupt handlers are defined
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]:
            try:
                if not candidate[i].p:
                    return False

                # INT0 (Breakpoint) can be used in RING 3
                if i == 3:
                    if candidate[i].dpl != 3:
                        return False
                else:
                    if i not in [1, 3, 4, 5]:
                        if candidate[i].dpl != 0:
                            return False
            except IndexError:
                return False
        return True


class CPU64(CPU):
    def __init__(self, features):
        super(CPU64, self).__init__(features)
        self.processor_features["idt_entry_size"] = 16
        self.processor_features["idt_unpack_format"] = "<4I"
        self.processor_features["idt_table_class"] = IDTable64
        CPU.processor_features = self.processor_features

        # Check validity of m_phy
        m_phy = self.processor_features.get("m_phy", -1)
        if m_phy <= 0 or m_phy >= 52:
            logging.fatal(
                "m_phy must be positive and less then 52 in PAE/IA64 MMU modes"
            )
            exit(1)
        CPU.m_phy = m_phy

    def classify_idt_entry(self, entry):
        s_selector = CPU.extract_bits(entry[0], 16, 16)
        offset = (
            (entry[2] << 32)
            + (CPU.extract_bits(entry[1], 16, 16) << 16)
            + CPU.extract_bits(entry[0], 0, 16)
        )
        ist = CPU.extract_bits(entry[1], 0, 3)
        dpl = CPU.extract_bits(entry[1], 13, 2)
        e_type = CPU.extract_bits(entry[1], 8, 4)

        # If BIT 15 of block 2 is 0 the entry is empty
        if not CPU.extract_bits(entry[1], 15, 1):
            return IDTEntry64(offset, s_selector, e_type, dpl, False, ist)

        # All the bits of blocks 6 and 7 must be 0
        if entry[3]:
            return None

        # BITS 3,4,5,6,7,12 of Block 2 must be 0
        if any([CPU.extract_bits(entry[1], b, 1) != 0 for b in [3, 4, 5, 6, 7, 12]]):
            return None

        # TYPE field can be only 14 or 15
        if e_type < 14:
            return None

        # Use only RING 3 and 0
        if 0 < dpl < 3:
            return None

        if e_type == 14:
            return IDTInterruptEntry64(offset, s_selector, e_type, dpl, True, ist)
        else:
            return IDTTrapEntry64(offset, s_selector, e_type, dpl, True, ist)

    def validate_idt(self, candidate):
        # Check if minimum interrupt handlers are defined
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19]:
            try:
                if not candidate[i].p:
                    return False

                # Check the canonical form of the interrupt handler address
                if CPU.extract_bits(candidate[i].offset, 47, 18) != 0x1FFFF:
                    return False

                # INT0 (Breakpoint) could be used in RING 3
                if i == 3:
                    if candidate[i].dpl != 3:
                        return False
                else:
                    # The other exceptions are reserved to RING 0
                    if i not in [1, 3, 4, 5]:
                        if candidate[i].dpl != 0:
                            return False

            except IndexError:
                return False
        return True


#####################################################################
# 32 bit entries and page table
#####################################################################


class IDTEntry32:
    entry_name = "Empty"
    labels = [
        "Present:",
        "Type:",
        "Interrupt address:",
        "Segment:",
        "DPL:",
        "Gate size:",
    ]
    addr_fmt = "0x{:08x}"

    def __init__(self, offset, segment, typ, dpl, p):
        self.offset = offset
        self.segment = segment
        self.type = typ
        self.dpl = dpl
        self.p = p

    def __hash__(self):
        return hash(self.entry_name)

    def __repr__(self):
        e_resume = self.entry_resume_stringified()
        return str(
            [self.labels[i] + " " + str(e_resume[i]) for i in range(len(self.labels))]
        )

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[2] = self.addr_fmt.format(res[2])
        for idx, r in enumerate(res):
            res[idx] = str(r)
        return res

    def entry_resume(self):
        return [self.p, self.entry_name, "", "", "", ""]


class IDTInterruptEntry32(IDTEntry32):
    entry_name = "Interrupt"
    labels = [
        "Present:",
        "Type:",
        "Interrupt address:",
        "Segment:",
        "DPL:",
        "Gate size:",
    ]

    def entry_resume(self):
        return [
            self.p,
            self.entry_name,
            self.offset,
            self.segment,
            self.dpl,
            self.gate_size(),
        ]

    def gate_size(self):
        return 32 if CPU.extract_bits(self.type, 3, 1) else 16


class IDTTrapEntry32(IDTInterruptEntry32):
    entry_name = "Trap"


class IDTTaskEntry32(IDTEntry32):
    entry_name = "Task"
    labels = ["Present:", "Type:", "TSS Segment:", "DPL:"]

    def entry_resume(self):
        return [self.p, self.entry_name, "Res.", self.segment, self.dpl, "Ign."]


class IDTable32:
    table_name = "Interrupt table"
    table_fields = [
        "Entry ID",
        "Present",
        "Type",
        "Interrupt address",
        "Segment",
        "DPL",
        "Gate size",
    ]
    addr_fmt = "0x{:08x}"

    def __hash__(self):
        return hash(self.table_name)

    def __init__(self, address, psize, entries):
        self.address = address
        self.size = psize
        self.entries = entries

    def __repr__(self):
        table = PrettyTable()
        table.field_names = self.table_fields

        for entry_idx, entry_obj in enumerate(self.entries):
            # entry_addr = self.address + (entry_idx * CPU.processor_features["idt_entry_size"])
            table.add_row([str(entry_idx)] + entry_obj.entry_resume_stringified())

        return str(table)

    def __len__(self):
        return self.size


class TEntry32(TableEntry):
    entry_size = 4
    entry_name = "TEntry32"
    size = 0
    labels = [
        "Address:",
        "Global:",
        "PAT:",
        "Dirty:",
        "Accessed:",
        "PCD:",
        "PWT:",
        "Kernel:",
        "Writable:",
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
            self.is_global_set(),
            self.is_pat_set(),
            self.is_dirty_entry(),
            self.is_accessed_entry(),
            self.is_pcd_set(),
            self.is_pwt_set(),
            self.is_supervisor_entry(),
            self.is_writeble_entry(),
        ]

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[0] = self.addr_fmt.format(res[0])
        for idx, r in enumerate(res[1:], start=1):
            res[idx] = str(r)
        return res

    def is_writeble_entry(self):
        return bool(MMU.extract_bits(self.flags, 1, 1))

    def is_readable_entry(self):
        return True

    def is_executable_entry(self):  # X bit does not exists in IA32
        return True

    def is_supervisor_entry(self):
        return not MMU.extract_bits(self.flags, 2, 1)

    def is_pwt_set(self):
        return bool(MMU.extract_bits(self.flags, 3, 1))

    def is_pcd_set(self):
        return bool(MMU.extract_bits(self.flags, 4, 1))

    def is_accessed_entry(self):
        return bool(MMU.extract_bits(self.flags, 5, 1))

    def is_dirty_entry(self):
        return bool(MMU.extract_bits(self.flags, 6, 1))

    def is_global_set(self):
        return bool(MMU.extract_bits(self.flags, 8, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 12, 20) << 12

    def get_permissions(self):
        perms = (True, self.is_writeble_entry(), True)
        if self.is_supervisor_entry():
            return perms + (False, False, False)
        else:
            return (False, False, False) + perms


class PTE4KB32(TEntry32):
    entry_name = "PTE4KB32"
    size = 1024 * 4

    def is_pat_set(self):
        return bool(MMU.extract_bits(self.flags, 7, 1))


class PDE4MB(TEntry32):
    entry_name = "PDE4MB"
    size = 1024 * 1024 * 4

    def is_pat_set(self):
        return bool(MMU.extract_bits(self.flags, 12, 1))

    @staticmethod
    def extract_addr(entry):
        low = MMU.extract_bits(entry, 22, 10)
        high = MMU.extract_bits(entry, 13, CPU.m_phy - 32)
        return (high << 32) + (low << 22)


class PDE32(TEntry32):
    entry_name = "PDE32"
    size = 0

    def is_dirty_entry(self):
        return "Ign."

    def is_pat_set(self):
        return "Ign."

    def is_global_set(self):
        return "Ign."


class PageTableIntel32(PageTable):
    entry_size = 4
    table_fields = [
        "Entry address",
        "Pointed address",
        "Global",
        "PAT",
        "Dirty",
        "Accessed",
        "PCD",
        "PWT",
        "Supervisor",
        "Writable",
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


class IDTEntry64(IDTEntry32):
    entry_name = "Empty"
    labels = ["Present:", "Type:", "Interrupt address:", "Segment:", "DPL:", "IST:"]

    def __init__(self, offset, segment, typ, dpl, p, ist):
        super(IDTEntry64, self).__init__(offset, segment, typ, dpl, p)
        self.ist = ist

    def entry_resume(self):
        return [self.p, self.entry_name, 0, 0, 0, 0]


class IDTInterruptEntry64(IDTEntry64):
    entry_name = "Interupt"

    def entry_resume(self):
        return [self.p, self.entry_name, self.offset, self.segment, self.dpl, self.ist]


class IDTTrapEntry64(IDTInterruptEntry64):
    entry_name = "Trap"


class IDTable64(IDTable32):
    table_name = "Interrupt table"
    table_fields = [
        "Entry ID",
        "Present",
        "Type",
        "Interrupt address",
        "Segment",
        "DPL",
        "IST",
    ]
    addr_fmt = "0x{:016x}"

    def __repr__(self):
        table = PrettyTable()
        table.field_names = self.table_fields

        for entry_idx, entry_obj in enumerate(self.entries):
            # entry_addr = self.address + (entry_idx * CPU.processor_features["idt_entry_size"])
            table.add_row([str(entry_idx)] + entry_obj.entry_resume_stringified())

        return str(table)


class TEntry64(TEntry32):
    entry_size = 8
    entry_name = "TEntry64"
    size = 0
    labels = [
        "Address:",
        "NX:",
        "Prot. key:",
        "Global:",
        "PAT:",
        "Dirty:",
        "Accessed:",
        "PCD:",
        "PWT:",
        "Kernel:",
        "Writable:",
    ]
    addr_fmt = "0x{:016x}"

    def __init__(self, address, flags, *args):
        super(TEntry64, self).__init__(address, flags, args)
        self.upper_flags = args[0]

    def __repr__(self):
        e_resume = self.entry_resume_stringified()
        return str(
            [self.labels[i] + " " + str(e_resume[i]) for i in range(len(self.labels))]
        )

    def entry_resume(self):
        return [
            self.address,
            self.is_executable_entry(),
            self.prot_key(),
            self.is_global_set(),
            self.is_pat_set(),
            self.is_dirty_entry(),
            self.is_accessed_entry(),
            self.is_pcd_set(),
            self.is_pwt_set(),
            self.is_supervisor_entry(),
            self.is_writeble_entry(),
        ]

    def is_executable_entry(self):
        return not bool(MMU.extract_bits(self.upper_flags, 5, 1))

    def prot_key(self):
        return "Ign."

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 12, CPU.m_phy - 12) << 12

    def get_permissions(self):
        perms = (True, self.is_writeble_entry(), self.is_executable_entry())
        if self.is_supervisor_entry():
            return perms + (False, False, False)
        else:
            return (False, False, False) + perms


class PTE4KB64(TEntry64):
    entry_name = "PTE4KB64"
    size = 1024 * 4

    def is_pat_set(self):
        return bool(MMU.extract_bits(self.flags, 7, 1))


class TPE64(TEntry64):
    entry_name = "TPE"
    size = 0

    def is_pat_set(self):
        return "Ign."

    def is_global_set(self):
        return "Ign."

    def is_dirty_entry(self):
        return "Ign."

    def prot_key(self):
        return MMU.extract_bits(self.upper_flags, 0, 4)


class PDE64(TPE64):
    entry_name = "PDE64"
    size = 0


class PDE2MB(TEntry64):
    entry_name = "PDE2MB"
    size = 1024 * 1024 * 2

    def is_pat_set(self):
        return bool(MMU.extract_bits(self.flags, 12, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 21, CPU.m_phy - 21) << 21


class PTE4KBPAE(PTE4KB64):
    entry_name = "PTE4KBPAE"

    def prot_key(self):
        return "Res."


class PDE2MBPAE(PDE2MB):
    entry_name = "PDE2MBPAE"

    def prot_key(self):
        return "Res."


class PDEPAE(PDE64):
    entry_name = "PDEPAE"

    def prot_key(self):
        return "Res."


class PDPTEPAE(TEntry64):
    entry_name = "PDPTEPAE"
    size = 0

    def is_pat_set(self):
        return "Ign."

    def is_global_set(self):
        return "Res."

    def is_dirty_entry(self):
        return "Res."

    def is_writeble_entry(self):
        return True

    def is_supervisor_entry(self):
        return False

    def is_accessed_entry(self):
        return "Res."

    def get_permissions(self):
        perms = (True, True, self.is_executable_entry())
        return perms * 2


class PDPTE(TPE64):
    entry_name = "PDPTE"
    size = 0


class PDPTE1GB(TEntry64):
    entry_name = "PDPTE1GB"
    size = 1024 * 1024 * 1024

    def is_pat_set(self):
        return bool(MMU.extract_bits(self.flags, 12, 1))

    @staticmethod
    def extract_addr(entry):
        return MMU.extract_bits(entry, 30, CPU.m_phy - 30) << 30


class PML4E(TPE64):
    entry_name = "PML4E"


class PageTableIntel64(PageTableIntel32):
    entry_size = 8
    table_fields = [
        "Entry address",
        "Pointed address",
        "NX",
        "Prot. key",
        "Global",
        "PAT",
        "Dirty",
        "Accessed",
        "PCD",
        "PWT",
        "Supervisor",
        "Writable",
        "Classes",
    ]


#################################################################
# MMU Modes
#################################################################


class MMU(MMURadix):
    PAGE_SIZE = 4096
    extract_bits = MMURadix.extract_bits_little
    paging_unpack_format = "<I"
    page_table_class = PageTableIntel32
    radix_levels = {"global": 2}
    top_prefix = 0
    entries_size = 4


################################################################
# MMU Modes
################################################################
class IA32(MMU):
    paging_unpack_format = "<I"
    page_table_class = PageTableIntel32
    radix_levels = {"global": 2}
    top_prefix = 0x0
    map_ptr_entries_to_levels = {"global": [PDE32, None]}
    map_datapages_entries_to_levels = {"global": [[PDE4MB], [PTE4KB32]]}
    map_level_to_table_size = {"global": [4096, 4096]}
    map_entries_to_shifts = {"global": {PDE32: 22, PDE4MB: 22, PTE4KB32: 12}}
    cr3_class = CR3_32
    map_reserved_entries_to_levels = {"global": [[], []]}

    def __init__(self, mmu_config):
        super(IA32, self).__init__(mmu_config)
        self.classify_entry = self.classify_entry_full

    def classify_entry_pd_only(self, page_addr, entry):
        # If BIT P=0 is EMPTY
        if not MMU.extract_bits(entry, 0, 1):
            return [False]

        # Check BIT 7
        if not MMU.extract_bits(entry, 7, 1):
            addr = PDE32.extract_addr(entry)
            if addr not in self.machine.memory.physpace["not_valid_regions"]:
                return [PDE32(addr, MMU.extract_bits(entry, 0, 13))]
            else:
                return [None]

        # Check BIT 21:M_PHYS-19: if are not 0 it's a PTE4K (PSE-40 AMD)
        if MMU.extract_bits(entry, CPU.m_phy - 19, 21 - (CPU.m_phy - 19) + 1):
            return [None]

        # It is a PDE4MB
        return [PDE4MB(PDE4MB.extract_addr(entry), MMU.extract_bits(entry, 0, 13))]

    def classify_entry_pt_only(self, page_addr, entry):
        # If BIT P=0 is EMPTY
        if not MMU.extract_bits(entry, 0, 1):
            return [False]
        else:
            return [
                PTE4KB32(PTE4KB32.extract_addr(entry), MMU.extract_bits(entry, 0, 13))
            ]

    def classify_entry_full(self, page_addr, entry):
        # If BIT P=0 is EMPTY
        if not MMU.extract_bits(entry, 0, 1):
            return [False]

        # -----------------------------------
        # Heuristic filter might be not valid
        # For 32-bit this heuristic
        # is the only filter to discard entries
        # but we prefer to not use it (more, more general!)
        # -----------------------------------
        # if is_dirty_entry and not is_accessed_entry:
        #    return [None]
        # -----------------------------------

        # Extract flags and address
        addr = PTE4KB32.extract_addr(entry)  # This is also the PDE32
        flags = MMU.extract_bits(entry, 0, 13)

        # Check BIT 7
        if not MMU.extract_bits(entry, 7, 1):
            ret = [PTE4KB32(addr, flags)]
            if addr not in self.machine.memory.physpace["not_valid_regions"]:
                ret.append(PDE32(addr, flags))
            return ret

        # Check BIT 21:M_PHYS-19: if are not 0 it's a PTE4K (PSE-40 AMD)
        if MMU.extract_bits(entry, CPU.m_phy - 19, 21 - (CPU.m_phy - 19) + 1):
            return [PTE4KB32(addr, flags)]

        # It is a PDE4M or a PTE4KB
        addr_4mb = PDE4MB.extract_addr(entry)
        return [PTE4KB32(addr, flags), PDE4MB(addr_4mb, flags)]

    def extend_prefix(self, prefix, entry_idx, entry_class):
        if entry_class in [PDE32, PDE4MB]:
            prefix = entry_idx << 22
            return prefix
        else:
            return prefix | (entry_idx << 12)

    def split_vaddr(self, vaddr):
        return (
            (
                (PDE32, MMU.extract_bits(vaddr, 22, 10)),
                (PTE4KB32, MMU.extract_bits(vaddr, 12, 10)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 12)),
            ),
            (
                (PDE4MB, MMU.extract_bits(vaddr, 22, 10)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 22)),
            ),
        )


class PAE(MMU):
    paging_unpack_format = "<Q"
    page_table_class = PageTableIntel64
    radix_levels = {"global": 3}
    top_prefix = 0x0
    map_ptr_entries_to_levels = {"global": [PDPTEPAE, PDEPAE, None]}
    map_datapages_entries_to_levels = {"global": [[None], [PDE2MBPAE], [PTE4KBPAE]]}
    map_level_to_table_size = {"global": [32, 4096, 4096]}
    map_entries_to_shifts = {
        "global": {PDPTEPAE: 30, PDEPAE: 21, PDE2MBPAE: 21, PTE4KBPAE: 12}
    }
    cr3_class = CR3_PAE
    map_reserved_entries_to_levels = {"global": [[], [], []]}

    def classify_entry(self, page_addr, entry):
        # Check BIT 0: must be 1 for a valid entry
        if not MMU.extract_bits(entry, 0, 1):
            return [False]

        # Check BITS 62:M_PHYS: they should be 0
        if MMU.extract_bits(entry, CPU.m_phy, 62 - CPU.m_phy + 1):
            return [None]

        addr_4k = PTE4KBPAE.extract_addr(entry)
        flags = MMU.extract_bits(entry, 0, 13)
        flags2 = MMU.extract_bits(entry, 59, 5)

        # Check BIT 7
        if MMU.extract_bits(entry, 7, 1):
            # BIT 7 = 1
            # Check BITS 20:13 if not all 0 it's a PTE4KB
            if MMU.extract_bits(entry, 13, 8):
                return [PTE4KBPAE(addr_4k, flags, flags2)]

            # It can be a PDE2MB
            addr_2mb = PDE2MBPAE.extract_addr(entry)
            return [
                PTE4KBPAE(addr_4k, flags, flags2),
                PDE2MBPAE(addr_2mb, flags, flags2),
            ]

        # BIT 7 = 0
        ret = [PTE4KBPAE(addr_4k, flags, flags2)]
        if addr_4k not in self.machine.memory.physpace["not_valid_regions"]:
            ret.append(PDEPAE(PDEPAE.extract_addr(entry), flags, flags2))

            # Check BITS 1,2,5,6,8,63: if they are not all 0 it cannot be a PDPTE
            if not (
                MMU.extract_bits(entry, 1, 2)
                or MMU.extract_bits(entry, 5, 2)
                or MMU.extract_bits(entry, 8, 1)
                or MMU.extract_bits(entry, 63, 1)
            ):
                ret.append(PDPTEPAE(PDPTEPAE.extract_addr(entry), flags, flags2))
        return ret

    def parse_parallel_frame(self, addresses, frame_size, pidx, **kwargs):
        # Custom version for PAE mode
        # The top table is composed by only 4 entries so we parse them directly here in a special way

        # Every process sleep a random delay in order to desincronize access to disk and maximixe the throuput
        sleep(uniform(pidx, pidx + 1) // 1000)

        data_pages = []
        empty_tables = []
        page_tables = [{} for i in range(self.radix_levels["global"])]
        pdpt_table = defaultdict(dict)
        frame_d = defaultdict(dict)
        mm = copy(self.machine.memory)
        mm.reopen()

        # Cicle over every frame
        total_elems, iterator = addresses
        for frame_addr in tqdm(
            iterator, position=-pidx, total=total_elems, leave=False
        ):
            frame_buf = mm.get_data(frame_addr, self.PAGE_SIZE)

            empty_entries = 0
            pdpt_empty_entries = 0
            is_invalid = False
            frame_d.clear()

            # Unpack records inside the frame
            for entry_idx, entry in enumerate(
                iter_unpack(self.paging_unpack_format, frame_buf)
            ):
                entry = entry[0]

                # Every four entry we can have a new PDPT table
                if entry_idx % 4 == 0:
                    pdpt_table.clear()
                    pdpt_empty_entries = 0

                # Classify entry
                entry_classes = self.classify_entry(frame_addr, entry)

                # It's a data page
                if entry_classes[0] is None:
                    is_invalid = True
                    continue

                # It's an empty page
                if entry_classes[0] is False:
                    empty_entries += 1
                    pdpt_empty_entries += 1
                else:
                    # Add entry to PDPT tables
                    for entry_obj in entry_classes:
                        if type(entry_obj) is PDPTEPAE:
                            pdpt_table[PDPTEPAE][entry_idx % 4] = entry_obj
                            break

                # Validate the PDPT if 4 aligned entries are parsed and add it to the page_tables
                if (entry_idx + 1) % 4 == 0:
                    if (
                        pdpt_empty_entries != 4
                        and pdpt_empty_entries + len(pdpt_table[PDPTEPAE]) == 4
                    ):
                        pdpt_table_obj = PageTableIntel64(
                            frame_addr + (entry_idx - 3) * 8,
                            4,
                            deepcopy(pdpt_table),
                            [0],
                        )
                        page_tables[0][pdpt_table_obj.address] = deepcopy(
                            pdpt_table_obj
                        )

                # Add the entries only if the table is not already marked as invalid
                if not is_invalid:
                    if entry_classes[0] is not False:
                        for entry_obj in entry_classes:
                            entry_type = type(entry_obj)
                            if entry_type is PDPTEPAE:
                                continue
                            else:
                                frame_d[entry_type][entry_idx] = entry_obj

            # If it is invalid add it to data pages directly
            if is_invalid:
                data_pages.append(frame_addr)
                continue

            # Classify the frame
            pt_classes = self.classify_frame(
                frame_d,
                empty_entries,
                int(frame_size // self.page_table_class.entry_size),
            )

            if -1 in pt_classes:  # EMPTY or DATA
                empty_tables.append(frame_addr)
            elif -2 in pt_classes:
                data_pages.append(frame_addr)
            else:
                for pt_class in pt_classes:
                    table_obj = self.page_table_class(
                        frame_addr, self.PAGE_SIZE, frame_d, pt_classes
                    )
                    page_tables[pt_class][frame_addr] = deepcopy(table_obj)

        return page_tables, data_pages, empty_tables

    def extend_prefix(self, prefix, entry_idx, entry_class):
        if entry_class == PDPTEPAE:
            prefix = entry_idx << 30
            return prefix

        elif entry_class in [PDE2MBPAE, PDEPAE]:
            return prefix | (entry_idx << 21)

        else:
            return prefix | (entry_idx << 12)

    def split_vaddr(self, vaddr):
        return (
            (PDPTEPAE, MMU.extract_bits(vaddr, 30, 9)),
            (PDEPAE, MMU.extract_bits(vaddr, 21, 9)),
            (PTE4KBPAE, MMU.extract_bits(vaddr, 12, 9)),
            ("OFFSET", MMU.extract_bits(vaddr, 0, 12)),
        ), (
            (PDPTEPAE, MMU.extract_bits(vaddr, 30, 9)),
            (PDE2MBPAE, MMU.extract_bits(vaddr, 21, 9)),
            ("OFFSET", MMU.extract_bits(vaddr, 0, 21)),
        )


class IA64(MMU):
    paging_unpack_format = "<Q"
    page_table_class = PageTableIntel64
    radix_levels = {"global": 4}
    top_prefix = 0xFFFF800000000000
    map_ptr_entries_to_levels = {"global": [PML4E, PDPTE, PDE64, None]}
    map_datapages_entries_to_levels = {
        "global": [[None], [PDPTE1GB], [PDE2MB], [PTE4KB64]]
    }
    map_level_to_table_size = {"global": [4096, 4096, 4096, 4096]}
    map_entries_to_shifts = {
        "global": {
            PML4E: 39,
            PDPTE: 30,
            PDPTE1GB: 30,
            PDE64: 21,
            PDE2MB: 21,
            PTE4KB64: 12,
        }
    }
    cr3_class = CR3_64
    map_reserved_entries_to_levels = {"global": [[], [], [], []]}

    def classify_entry(self, page_addr, entry):
        # Check BIT 0: must be 1 for a valid entry
        if not MMU.extract_bits(entry, 0, 1):
            return [False]

        # Check BITS 51:M: should be all 0 for a valid entry
        if MMU.extract_bits(entry, CPU.m_phy, 51 - CPU.m_phy + 1):
            return [None]

        flags = MMU.extract_bits(entry, 0, 13)
        flags2 = MMU.extract_bits(entry, 59, 5)

        # Check BIT 7
        if not MMU.extract_bits(entry, 7, 1):
            # BIT 7 = 0
            addr = PTE4KB64.extract_addr(entry)
            ret = [PTE4KB64(addr, flags, flags2)]

            if addr not in self.machine.memory.physpace["not_valid_regions"]:
                ret.extend(
                    [
                        PDE64(PDE64.extract_addr(entry), flags, flags2),
                        PDPTE(PDPTE.extract_addr(entry), flags, flags2),
                        PML4E(PML4E.extract_addr(entry), flags, flags2),
                    ]
                )
            return tuple(ret)

        # BIT 7 = 1
        # Check BITS 20:13: if one of them is not 0, it's a PTE4KB
        pte4k = PTE4KB64(PTE4KB64.extract_addr(entry), flags, flags2)
        if MMU.extract_bits(entry, 13, 8):
            return [pte4k]

        ret = [pte4k, PDE2MB(PDE2MB.extract_addr(entry), flags, flags2)]

        # Check BITS 29:21 if one of them is not 0 it cannot be a PDPTE 1GB
        if not MMU.extract_bits(entry, 21, 9):
            ret.append(PDPTE1GB(PDPTE1GB.extract_addr(entry), flags, flags2))

        return tuple(ret)

    def extend_prefix(self, prefix, entry_idx, entry_class):
        if entry_class == PML4E:
            prefix = entry_idx << 39
            if self.extract_bits(prefix, 47, 1):
                prefix = 0xFFFF800000000000 | prefix  # Canonical form
            return prefix

        elif entry_class in [PDPTE1GB, PDPTE]:
            return prefix | (entry_idx << 30)

        elif entry_class in [PDE2MB, PDE64]:
            return prefix | (entry_idx << 21)

        else:
            return prefix | (entry_idx << 12)

    def split_vaddr(self, vaddr):
        return (
            (
                (PML4E, MMU.extract_bits(vaddr, 39, 9)),
                (PDPTE, MMU.extract_bits(vaddr, 30, 9)),
                (PDE64, MMU.extract_bits(vaddr, 21, 9)),
                (PTE4KB64, MMU.extract_bits(vaddr, 12, 9)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 12)),
            ),
            (
                (PML4E, MMU.extract_bits(vaddr, 39, 9)),
                (PDPTE, MMU.extract_bits(vaddr, 30, 9)),
                (PDE2MB, MMU.extract_bits(vaddr, 21, 9)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 21)),
            ),
            (
                (PML4E, MMU.extract_bits(vaddr, 39, 9)),
                (PDPTE1GB, MMU.extract_bits(vaddr, 30, 9)),
                ("OFFSET", MMU.extract_bits(vaddr, 0, 30)),
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
                idts=[],
                cr3s={},
            )

    def do_parse_memory(self, args):
        """Find MMU tables and IDTs"""
        if self.data.is_mem_parsed:
            logger.warning("Memory already parsed")
            return

        if type(self.machine.mmu) is IA32:
            self.parse_memory_ia32()
        elif type(self.machine.mmu) is PAE:
            self.parse_memory_pae()
        elif type(self.machine.mmu) is IA64:
            self.parse_memory_ia64()
        else:
            logging.fatal("OOPS... MMU class unkown!")
            exit(-1)
        self.data.is_mem_parsed = True

    def do_show_idt(self, args):
        """Show the IDT at a chosen address. Usage: show_idt ADDRESS"""
        args = args.split()
        if len(args) < 1:
            logger.warning("Missing table address")
            return

        try:
            addr = self.parse_int(args[0])
        except ValueError:
            logger.warning("Invalid table address")
            return

        print(self.machine.cpu.parse_idt(addr))

    def do_show_table(self, args):
        """Show an MMU table at a chosen address. Usage: show_table ADDRESS [level]"""
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
                if lvl > (self.machine.mmu.radix_levels["global"] - 1):
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
        table_buff = self.machine.memory.get_data(addr, table_size)
        invalids, pt_classes, table_obj = self.machine.mmu.parse_frame(
            table_buff, addr, table_size, lvl
        )
        print(table_obj)
        print(f"Invalid entries: {invalids} Table levels: {pt_classes}")

    def parse_memory_ia32(self):
        # Due to the impossibility to differentiate among data and page table (PT) in a simple way,
        # we first look for PDs and then only to PTs pointed by PDs, otherwise memory consumption is unmanageable...

        # Look for only PD, then use that to find only PT
        logger.info("Look for page directories..")
        self.machine.mmu.classify_entry = self.machine.mmu.classify_entry_pd_only
        parallel_results = self.machine.apply_parallel(
            self.machine.mmu.PAGE_SIZE, self.machine.mmu.parse_parallel_frame
        )

        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            page_tables, data_pages, empty_tables = result.get()

            self.data.page_tables["global"][0].update(page_tables[0])
            self.data.data_pages.extend(data_pages)
            self.data.empty_tables.extend(empty_tables)

        logger.info("Collect page tables...")
        # Collect all PT addresses pointed by PD
        pt_candidates = set()
        for pd_obj in self.data.page_tables["global"][0].values():
            for entry_obj in pd_obj.entries[PDE32].values():
                pt_candidates.add(entry_obj.address)
        pt_candidates = list(pt_candidates)
        pt_candidates.sort()
        self.machine.mmu.classify_entry = self.machine.mmu.classify_entry_pt_only

        # Workaround to reduce thread memory consumption
        data = self.data
        self.data = None

        iterators = [
            (len(y), y)
            for y in [list(x) for x in divide(mp.cpu_count(), pt_candidates)]
        ]
        parsing_results_async = self.machine.apply_parallel(
            self.machine.mmu.PAGE_SIZE,
            self.machine.mmu.parse_parallel_frame,
            iterators=iterators,
        )

        # Restore previous data and set classify_entry to full version
        self.data = data
        self.machine.mmu.classify_entry = self.machine.mmu.classify_entry_full

        # Reaggregate data
        logger.info("Reaggregate threads data...")
        for result in parsing_results_async:
            page_tables, data_pages, empty_tables = result.get()
            self.data.page_tables["global"][1].update(page_tables[1])
            self.data.data_pages.extend(data_pages)
            self.data.empty_tables.extend(empty_tables)

        # Remove PT from data pages (in the first phase the alogrith has classified PT as data pages, now that
        # we know which PT is a true one, they must be removed from data pages)
        self.data.data_pages = set(self.data.data_pages)
        self.data.data_pages.difference_update(
            self.data.page_tables["global"][1].keys()
        )
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
            ]  # Trick! Only one dataclass per level
            for table_addr, table_obj in self.data.page_tables["global"][lvl].items():
                for entry_obj in table_obj.entries[ptr_class].values():
                    self.data.reverse_map_tables[lvl][entry_obj.address].add(
                        table_obj.address
                    )
                for entry_obj in table_obj.entries[page_class].values():
                    self.data.reverse_map_pages[lvl][entry_obj.address].add(
                        table_obj.address
                    )

        logger.info("Look for interrupt tables...")
        self.data.idts = self.machine.cpu.find_idt_tables()

    def parse_memory_pae(self):
        # It uses the same function of IA64 but with a custom parse_parallel_frame
        self.parse_memory_ia64()

    def parse_memory_ia64(self):
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
            ]  # Trick! Only one dataclass per level
            for table_addr, table_obj in self.data.page_tables["global"][lvl].items():
                for entry_obj in table_obj.entries[ptr_class].values():
                    self.data.reverse_map_tables[lvl][entry_obj.address].add(
                        table_obj.address
                    )
                for entry_obj in table_obj.entries[page_class].values():
                    self.data.reverse_map_pages[lvl][entry_obj.address].add(
                        table_obj.address
                    )

        logger.info("Look for interrupt tables...")
        self.data.idts = self.machine.cpu.find_idt_tables()

    def do_show_idtrs(self, args):
        """Show IDT tables founds"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        table = PrettyTable()
        table.field_names = ["Address", "Size"]
        for idt in self.data.idts:
            table.add_row([hex(idt.address), idt.size])
        print(table)

    def do_find_radix_trees(self, args):
        """Reconstruct radix trees"""

        # Some table level was not found...
        if not len(self.data.page_tables["global"][0]):
            logger.warning("OOPS... no tables in first level... Wrong MMU mode?")
            return

        cr3s = {}
        # No valid IDT found
        if not self.data.idts:
            logger.warning("No valid IDTs found, collect all valid CR3s...")

            # Start from page tables of lower level and derive upper level tables (aka CR3)
            # Filter for self-referencing CR3 (euristic) does not work with microkernels or SMAP/SMEP
            cr3_candidates = []
            already_explored = set()
            for page_addr in tqdm(self.data.data_pages.union(self.data.empty_tables)):
                derived_addresses = self.machine.mmu.derive_page_address(page_addr)
                for derived_address in derived_addresses:
                    if derived_address in already_explored:
                        continue
                    lvl, addr = derived_address
                    cr3_candidates.extend(
                        self.radix_roots_from_data_page(
                            lvl,
                            addr,
                            self.data.reverse_map_pages,
                            self.data.reverse_map_tables,
                        )
                    )
                    already_explored.add(derived_address)
            cr3_candidates = list(
                set(cr3_candidates).intersection(
                    self.data.page_tables["global"][0].keys()
                )
            )

            # Refine dataset and use a fake IDT table
            cr3s = {-1: {}}

            logger.info("Filter candidates...")
            for cr3 in tqdm(cr3_candidates):
                # Obtain radix tree infos
                consistency, pas = self.physpace(
                    cr3,
                    self.data.page_tables["global"],
                    self.data.empty_tables,
                    hierarchical=True,
                )

                # Only consistent trees are valid
                if not consistency:
                    continue

                # Esclude empty trees
                if pas.get_kernel_size() == pas.get_user_size() == 0:
                    continue

                vas = self.virtspace(
                    cr3, 0, self.machine.mmu.top_prefix, hierarchical=True
                )
                cr3s[-1][cr3] = RadixTree(cr3, 0, pas, vas)

            self.data.cr3s = cr3s
            self.data.is_radix_found = True
            return

        for idt_obj in self.data.idts:  # Cycle on all IDT found
            cr3_candidates = set()
            cr3s[idt_obj.address] = {}

            # We cannot filter radix trees root on the basis that the pointed tree is able to address its top table
            # this assumption is not valid for microkernels!

            # Collect all possible CR3: a valid CR3 must be able to address the IDT
            logger.info("Collect all valids CR3s...")
            idt_pg_addresses = self.machine.mmu.derive_page_address(
                idt_obj.address >> 12 << 12
            )

            for level, addr in idt_pg_addresses:
                cr3_candidates.update(
                    self.radix_roots_from_data_page(
                        level,
                        addr,
                        self.data.reverse_map_pages,
                        self.data.reverse_map_tables,
                    )
                )
            cr3_candidates = list(
                cr3_candidates.intersection(self.data.page_tables["global"][0].keys())
            )
            logger.info(
                "Number of possible CR3s for IDT located at {}:{}".format(
                    hex(idt_obj.address), len(cr3_candidates)
                )
            )

            # Collect the page containig each virtual addresses defined inside interrupt handlers
            handlers_pages = set()
            for handler in idt_obj.entries:
                # Task Entry does not point to interrupt hanlder
                if isinstance(handler, IDTTaskEntry32):
                    continue

                # Ignore handler unused
                if not handler.p:
                    continue

                handlers_pages.add(handler.offset >> 12 << 12)

            # Try to resolve interrupt virtual addresses and count the number of unresolved interrupt handlers
            cr3s_for_idt = []
            for cr3_candidate in cr3_candidates:
                errors = 0
                for vaddr in handlers_pages:
                    paddr = self.resolve_vaddr(cr3_candidate, vaddr)
                    if paddr == -1:
                        logging.debug(
                            f"find_radix_trees(): {hex(cr3_candidate)} failed to solve {hex(vaddr)}"
                        )
                        errors += 1

                cr3s_for_idt.append([cr3_candidate, errors])

            # At least one CR3 must be found...
            if not cr3s_for_idt:
                continue

            # Save only CR3s which resolv the max number of addresses
            cr3s_for_idt.sort(key=lambda x: (x[1], x[0]))
            max_value = cr3s_for_idt[0][1]
            logger.debug(
                "Interrupt pages: {}, Maximum pages resolved: {}".format(
                    len(handlers_pages), len(handlers_pages) - max_value
                )
            )

            # Consider only CR3 which resolve the maximum number of interrupt pages
            for cr3 in cr3s_for_idt:
                if max_value != cr3[1]:
                    break

                # Extract an approximation of the kernel and user physical address space
                consistency, pas = self.physpace(
                    cr3[0],
                    self.data.page_tables["global"],
                    self.data.empty_tables,
                    hierarchical=True,
                )

                # Only consistent trees are valid
                if not consistency:
                    continue

                # Esclude empty trees
                if pas.get_kernel_size() == pas.get_user_size() == 0:
                    continue

                vas = self.virtspace(
                    cr3[0], 0, self.machine.mmu.top_prefix, hierarchical=True
                )
                cr3s[idt_obj.address][cr3[0]] = RadixTree(cr3[0], 0, pas, vas)

        self.data.cr3s = cr3s
        self.data.is_radix_found = True

    def do_show_radix_trees(self, args):
        """Show radix trees found able to address a chosen IDT table. Usage: show_radix_trees PHY_IDT_ADDRESS"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        # Check if the IDT requested is in the list of IDT found
        args = args.split()
        if len(args) < 1:
            logging.warning("Missing IDT")
            return
        idt_addr = self.parse_int(args[0])

        if not self.data.idts:
            logging.info("No IDT found by MMUShell")
            idt_addr = -1
        else:
            for idt in self.data.idts:
                if idt_addr == idt.address:
                    break
            else:
                logging.warning("IDT requested not in IDT found!")
                return

        # Show results
        labels = [
            "Radix address",
            "First level",
            "Kernel size (Bytes)",
            "User size (Bytes)",
        ]
        table = PrettyTable()
        table.field_names = labels
        for cr3 in self.data.cr3s[idt_addr].values():
            table.add_row(cr3.entry_resume_stringified())
        table.sortby = "Radix address"
        print(table)


class MMUShellGTruth(MMUShell):
    def do_show_idtrs_gtruth(self, args):
        """Compare IDTs found with the ground truth"""
        if not self.data.is_mem_parsed:
            logging.info("Please, find them first!")
            return

        # We need to use a CR3 to translate the IDTR in virtual address, we use the last CR3 defined
        cr3_class = self.machine.mmu.cr3_class

        # Filter CR3 for valid one and the find the last used one
        keys = list(self.gtruth["CR3"].keys())
        keys.sort(key=lambda x: self.gtruth["CR3"][x][1], reverse=True)

        for cr3 in keys:
            cr3_obj = cr3_class(cr3)

            # Validate CR3
            if cr3_obj.address not in self.data.page_tables["global"][0]:
                continue
            consistency, pas = self.physpace(
                cr3_obj.address,
                self.data.page_tables["global"],
                self.data.empty_tables,
                hierarchical=True,
            )
            if not consistency or (
                not pas.get_kernel_size() and not pas.get_user_size()
            ):
                continue
            else:
                valid_cr3_obj = cr3_obj
                break
        else:
            logging.warning("OOPS.. no valid CR3 found")
            return

        # Collect all found physical addresses
        idts = [x.address for x in self.data.idts]

        # Resolve the IDT virtual address
        table = PrettyTable()
        table.field_names = [
            "Virtual address",
            "Physical address",
            "Found",
            "First seen",
            "Last seen",
        ]

        tp = 0
        unresolved = 0
        keys = list(self.gtruth["IDTR"].keys())
        keys.sort(key=lambda x: self.gtruth["IDTR"][x][1])
        for idtr in keys:
            idtr_obj = IDTR(idtr)

            paddr = self.resolve_vaddr(valid_cr3_obj.address, idtr_obj.address)
            # Not solved by the CR3...
            if paddr == -1:
                unresolved += 1
                table.add_row(
                    [
                        hex(idtr_obj.address),
                        "?",
                        "?",
                        self.gtruth["IDTR"][idtr_obj.value][0],
                        self.gtruth["IDTR"][idtr_obj.value][1],
                    ]
                )
            else:
                if paddr in idts:
                    tp += 1
                    found = "X"
                else:
                    found = ""
                table.add_row(
                    [
                        hex(idtr_obj.address),
                        hex(paddr),
                        found,
                        self.gtruth["IDTR"][idtr_obj.value][0],
                        self.gtruth["IDTR"][idtr_obj.value][1],
                    ]
                )

        print(f"Use CR3 address: {hex(valid_cr3_obj.address)}")
        print(table)
        print(f"TP:{tp} FP:{len(idts) - tp} Unresolved: {unresolved}")

        # Export results for next analysis
        if len(args) == 2 and args[1] == "export":
            from pickle import dump as dump_p

            with open("dump.mmu", "wb") as f:
                results = [{"cr3": tp} for tp in sorted(tps)]
                dump_p(results, f)

    def do_show_radix_trees_gtruth(self, args):
        """Compare radix trees found able to address a chosen IDT table with the ground truth. Usage: show_radix_trees_gtruth PHY_IDT_ADDRESS"""
        if not self.data.is_radix_found:
            logging.info("Please, find them first!")
            return

        # Check if the IDT requested is in the list of IDT found
        args = args.split()
        if len(args) < 1:
            logging.warning("Missing IDT")
            return
        idt_addr = self.parse_int(args[0])
        if idt_addr not in self.machine.memory:
            logging.warning("IDT address not in RAM")
            return

        if not self.data.idts:
            # If no valid IDT has been found by MMUShell, it parses and uses the IDT
            # address pass by the user as filter to identify TP radix trees
            logging.info("No IDT found by MMUShell")

            # Parse IDT
            idt = self.machine.cpu.parse_idt(idt_addr)
            if not len(idt):
                logging.warning("No IDT at address")
                return
        else:
            for idt in self.data.idts:
                if idt_addr == idt.address:
                    break
            else:
                logging.warning("IDT requested not in IDT found!")
                return

        # Collect all valid CR3
        latest_idt_va_used = IDTR(
            sorted(
                list(self.gtruth["IDTR"].keys()),
                key=lambda x: self.gtruth["IDTR"][x][1],
            )[-1]
        )
        idts = {}
        cr3_errors = defaultdict(list)

        for cr3 in self.gtruth["CR3"]:
            cr3_obj = self.machine.mmu.cr3_class(cr3)
            if cr3_obj.address not in self.data.page_tables["global"][0]:
                continue
            consistency, pas = self.physpace(
                cr3_obj.address,
                self.data.page_tables["global"],
                self.data.empty_tables,
                hierarchical=True,
            )
            if not consistency or (
                not pas.get_kernel_size() and not pas.get_user_size()
            ):
                continue

            # Check if they are able to address the IDT table
            derived_addresses = self.machine.mmu.derive_page_address(
                idt_addr >> 12 << 12
            )
            if not any([x[1] in pas for x in derived_addresses]):
                continue  # Trick! Only one dataclass per level

            # Check if the CR3 is able to resolve the latest IDTR value used
            # (we check this for simplicity instead of the VA associated with the selected IDT)
            idt_phys = self.resolve_vaddr(cr3_obj.address, latest_idt_va_used.address)
            if idt_phys == -1 or idt_phys not in self.machine.memory:
                continue

            # Collect interrupt VA for resolved IDR
            if idt_phys not in idts:
                vas_interrupts = set()
                idt_obj = self.machine.cpu.parse_idt(idt_phys)
                for handler in idt_obj.entries:
                    if isinstance(handler, IDTTaskEntry32) or not handler.p:
                        continue
                    vas_interrupts.add(handler.offset >> 12 << 12)
                idts[idt_phys] = vas_interrupts

            # Check how much IDT interrupt VA is able to resolve
            errors = 0
            for va in idts[idt_phys]:
                if self.resolve_vaddr(cr3_obj.address, va) == -1:
                    errors += 1
            cr3_errors[errors].append(cr3_obj)

        # Use only CR3 with the minimum number of errors
        valid_cr3s = {}
        for cr3_obj in cr3_errors[sorted(list(cr3_errors.keys()))[0]]:
            valid_cr3s[cr3_obj.address] = cr3_obj

        # Use fake IDT address if no IDT are found
        if not self.data.idts:
            idt_addr = -1

        # True positives, false negatives, false positives
        tps = set(valid_cr3s.keys()).intersection(set(self.data.cr3s[idt_addr].keys()))
        fns = set(valid_cr3s.keys()).difference(set(self.data.cr3s[idt_addr].keys()))
        fps = set(self.data.cr3s[idt_addr].keys()).difference(set(valid_cr3s.keys()))

        # Show results
        table = PrettyTable()
        table.field_names = ["Address", "Found", "First seen", "Last seen"]
        for tp in sorted(tps):
            table.add_row(
                [
                    hex(tp),
                    "X",
                    self.gtruth["CR3"][valid_cr3s[tp].value][0],
                    self.gtruth["CR3"][valid_cr3s[tp].value][1],
                ]
            )

        for fn in sorted(fns):
            table.add_row(
                [
                    hex(fn),
                    "",
                    self.gtruth["CR3"][valid_cr3s[fn].value][0],
                    self.gtruth["CR3"][valid_cr3s[fn].value][1],
                ]
            )

        for fp in sorted(fps):
            table.add_row([hex(fp), "False positive", "", ""])

        print(table)
        print(f"TP:{len(tps)} FN:{len(fns)} FP:{len(fps)}")

        # Export results for next analysis
        if len(args) == 2 and args[1] == "export":
            from pickle import dump as dump_p

            with open("dump.mmu", "wb") as f:
                results = [{"cr3": tp} for tp in sorted(tps)]
                dump_p(results, f)
