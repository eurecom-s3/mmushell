import logging

from architectures.generic import Machine as MachineDefault
from architectures.generic import CPU as CPUDefault
from architectures.generic import PhysicalMemory as PhysicalMemoryDefault
from architectures.generic import MMUShell as MMUShellDefault
from architectures.generic import MMU as MMUDefault
from architectures.generic import CPUReg

from miasm.analysis.machine import Machine as MIASMMachine
from miasm.core.bin_stream import bin_stream_vm
from miasm.core.locationdb import LocationDB
from miasm.jitter.VmMngr import Vm
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, PAGE_EXEC

from prettytable import PrettyTable
from dataclasses import dataclass
from collections import defaultdict
from struct import unpack
from pprint import pprint
from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class Data:
    is_mem_parsed: bool
    is_registers_found: bool
    opcodes: dict
    regs_values: dict


class Machine(MachineDefault):
    def __init__(self, cpu, mmu, memory, **kwargs):
        super(Machine, self).__init__(cpu, mmu, memory, **kwargs)

    def get_miasm_machine(self):
        mn_s = (
            "mips" + str(self.cpu.bits) + ("b" if self.cpu.endianness == "big" else "l")
        )
        return MIASMMachine(mn_s)


class CPURegMIPS(CPUReg):
    @classmethod
    def get_register_obj(cls, reg_name, value):
        return globals()[reg_name](value)


class ContextConfig(CPURegMIPS):
    def is_valid(self, value):
        # Check if it is a mask of contigous 1s
        digits = bin(value)[2:]
        if len(digits) == 1:
            return True

        digit_changes = 0
        for i in range(1, len(digits)):
            if digits[i - 1] != digits[i]:
                digit_changes += 1

        if digits[0] == "1" or digits[-1] == "1":
            return digit_changes <= 1
        else:
            return digit_changes <= 2

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.VirtualIndex = value
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.valid == other.valid and self.VirtualIndex == other.VirtualIndex

    def __repr__(self):
        return (
            f"ContextConfig {hex(self.value)} => VirtualIndex:{hex(self.VirtualIndex)}"
        )


class PageMask(CPURegMIPS):
    def is_valid(self, value):
        if (
            CPU.extract_bits(value, 0, 11) != 0x0
            or CPU.extract_bits(value, 29, 3) != 0x0
        ):
            return False

        MaskX = CPU.extract_bits(value, 11, 2)
        Mask = CPU.extract_bits(value, 13, 16)
        return self._is_a_valid_mask(Mask) and self._is_a_valid_mask(MaskX)

    def _is_a_valid_mask(self, value):
        # Check if it is a mask of 1s
        digits = bin(value)[2:]
        if len(digits) == 1:
            return True

        digit_changes = 0
        for i in range(1, len(digits)):
            if digits[i - 1] != digits[i]:
                digit_changes += 1
        return digit_changes <= 1

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.MaskX = CPU.extract_bits(value, 11, 2)
            self.Mask = CPU.extract_bits(value, 13, 16)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.Mask == other.Mask
            and self.MaskX == other.MaskX
        )

    def __repr__(self):
        return f"PageMask {hex(self.value)} => Mask:{hex(self.Mask)}, MaskX:{hex(self.MaskX)}"


class PageGrain(CPURegMIPS):
    def is_valid(self, value):
        return not (
            CPU.extract_bits(value, 5, 3) != 0x0
            or CPU.extract_bits(value, 13, 13) != 0x0
        )

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.MCCause = CPU.extract_bits(value, 0, 5)
            self.ASE = CPU.extract_bits(value, 8, 13)
            self.S32 = CPU.extract_bits(value, 26, 1)
            self.IEC = CPU.extract_bits(value, 27, 1)
            self.ESP = CPU.extract_bits(value, 28, 1)
            self.ELPA = CPU.extract_bits(value, 29, 1)
            self.XIE = CPU.extract_bits(value, 30, 1)
            self.RIE = CPU.extract_bits(value, 31, 1)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.RIE == other.RIE
            and self.XIE == other.XIE
            and self.ELPA == other.ELPA
            and self.ESP == other.ESP
        )

    def __repr__(self):
        return f"PageGrain {hex(self.value)} => RIE:{hex(self.RIE)}, XIE:{hex(self.XIE)}, ELPA:{hex(self.ELPA)}, ESP:{hex(self.ESP)}"


class SegCtl(CPURegMIPS):
    def is_valid(self, value):
        return not (
            CPU.extract_bits(value, 7, 2) != 0x0
            or CPU.extract_bits(value, 23, 2) != 0x0
        )


class SegCtl0(SegCtl):
    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.CFG0 = CPU.extract_bits(value, 0, 16)
            self.CFG1 = CPU.extract_bits(value, 16, 16)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.CFG0 == other.CFG0
            and self.CFG1 == other.CFG1
        )

    def __repr__(self):
        return (
            f"SegCtl0 {hex(self.value)} => CFG0:{hex(self.CFG0)}, CFG1:{hex(self.CFG1)}"
        )


class SegCtl1(SegCtl):
    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.CFG2 = CPU.extract_bits(value, 0, 16)
            self.CFG3 = CPU.extract_bits(value, 16, 16)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.CFG2 == other.CFG2
            and self.CFG3 == other.CFG3
        )

    def __repr__(self):
        return (
            f"SegCtl1 {hex(self.value)} => CFG2:{hex(self.CFG2)}, CFG3:{hex(self.CFG3)}"
        )


class SegCtl2(SegCtl):
    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.CFG4 = CPU.extract_bits(value, 0, 16)
            self.CFG5 = CPU.extract_bits(value, 16, 16)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.CFG4 == other.CFG4
            and self.CFG5 == other.CFG5
        )

    def __repr__(self):
        return (
            f"SegCtl2 {hex(self.value)} => CFG4:{hex(self.CFG4)}, CFG5:{hex(self.CFG5)}"
        )


class PWBase(CPURegMIPS):
    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.PWBase = value
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.valid == other.valid and self.PWBase == other.PWBase

    def __repr__(self):
        return f"PWBase {hex(self.value)} => PWBase:{hex(self.PWBase)}"


class PWField(CPURegMIPS):
    def is_valid(self, value):
        if CPU.processor_features["R6_CPU"] and (
            CPU.extract_bits(value, 24, 6) < 12
            or CPU.extract_bits(value, 18, 6) < 12
            or CPU.extract_bits(value, 12, 6) < 12
            or CPU.extract_bits(value, 6, 6) < 12
        ):
            return False

        return CPU.extract_bits(value, 30, 2) == 0x0

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.PTEI = CPU.extract_bits(value, 0, 6)
            self.PTI = CPU.extract_bits(value, 6, 6)
            self.MDI = CPU.extract_bits(value, 12, 6)
            self.UDI = CPU.extract_bits(value, 18, 6)
            self.GDI = CPU.extract_bits(value, 24, 6)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.PTEI == other.PTEI
            and self.PTI == other.PTI
            and self.MDI == other.MDI
            and self.UDI == other.UDI
            and self.GDI == other.GDI
        )

    def __repr__(self):
        return f"PWField {hex(self.value)} => PTEI:{hex(self.PTEI)}, PTI:{hex(self.PTI)}, MDI:{hex(self.MDI)}, UDI:{hex(self.UDI)}, GDI:{hex(self.GDI)}"


class PWSize(CPURegMIPS):
    def is_valid(self, value):
        if CPU.processor_features["R6_CPU"] and CPU.extract_bits(value, 6, 6) != 1:
            return False

        return CPU.extract_bits(value, 30, 2) == 0x0

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.PTEW = CPU.extract_bits(value, 0, 6)
            self.PTW = CPU.extract_bits(value, 6, 6)
            self.MDW = CPU.extract_bits(value, 12, 6)
            self.UDW = CPU.extract_bits(value, 18, 6)
            self.GDW = CPU.extract_bits(value, 24, 6)
            self.PS = CPU.extract_bits(value, 30, 1)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.PTEW == other.PTEW
            and self.PTW == other.PTW
            and self.MDW == other.MDW
            and self.UDW == other.UDW
            and self.GDW == other.GDW
            and self.PS == other.PS
        )

    def __repr__(self):
        return f"PWSize {hex(self.value)} => PTEW:{hex(self.PTEW)}, PTW:{hex(self.PTW)}, MDW:{hex(self.MDW)}, UDW:{hex(self.UDW)}, GDW:{hex(self.GDW)}, PS:{hex(self.PS)}"


class Wired(CPURegMIPS):
    def is_valid(self, value):
        return CPU.extract_bits(value, 0, 16) <= CPU.extract_bits(value, 16, 16)

    def __init__(self, value):
        self.value = value
        self.Wired = CPU.extract_bits(value, 0, 16)
        self.Limit = CPU.extract_bits(value, 16, 16)
        if self.is_valid(value):
            self.valid = True
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return self.valid == other.valid and self.value == other.value

    def __repr__(self):
        return f"Wired {hex(self.value)} => Wired:{self.Wired}, Limit:{self.Limit}"


class PWCtl(CPURegMIPS):
    def is_valid(self, value):
        return CPU.extract_bits(value, 8, 23) == 0x0

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.Psn = CPU.extract_bits(value, 0, 6)
            self.HugePg = CPU.extract_bits(value, 6, 1)
            self.DPH = CPU.extract_bits(value, 7, 1)
            self.PWEn = CPU.extract_bits(value, 31, 1)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.Psn == other.Psn
            and self.HugePg == other.HugePg
            and self.DPH == other.DPH
            and self.PWEn == other.PWEn
        )

    def __repr__(self):
        return f"PWCtl {hex(self.value)} => Psn:{hex(self.Psn)}, HugePg:{hex(self.HugePg)}, DPH:{hex(self.DPH)}, PWEn:{hex(self.PWEn)}"


class Config(CPURegMIPS):
    def is_valid(self, value):
        return (
            CPU.extract_bits(value, 4, 3) == 0x0 and CPU.extract_bits(value, 31, 1) == 1
        )

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.K0 = CPU.extract_bits(value, 0, 3)
            self.VI = CPU.extract_bits(value, 3, 1)
            self.MT = CPU.extract_bits(value, 7, 3)
            self.AR = CPU.extract_bits(value, 10, 3)
            self.AT = CPU.extract_bits(value, 13, 2)
            self.BE = CPU.extract_bits(value, 15, 1)
            self.KU = CPU.extract_bits(value, 25, 3)
            self.K23 = CPU.extract_bits(value, 28, 3)
            self.M = CPU.extract_bits(value, 31, 1)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.K0 == other.K0
            and self.MT == other.MT
            and self.KU == other.KU
            and self.K23 == other.K23
        )

    def __repr__(self):
        return f"Config {hex(self.value)} => K0:{hex(self.K0)}, MT:{hex(self.MT)}, KU:{hex(self.KU)}, K23:{hex(self.K23)}"


class Config5(CPURegMIPS):
    def is_valid(self, value):
        return not (
            CPU.extract_bits(value, 1, 1) != 0x0
            or CPU.extract_bits(value, 12, 1) != 0x0
            or CPU.extract_bits(value, 14, 13) != 0x0
        )

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.NFExists = CPU.extract_bits(value, 0, 1)
            self.UFR = CPU.extract_bits(value, 2, 1)
            self.MRP = CPU.extract_bits(value, 3, 1)
            self.LLB = CPU.extract_bits(value, 4, 1)
            self.MVH = CPU.extract_bits(value, 5, 1)
            self.SBRI = CPU.extract_bits(value, 6, 1)
            self.VP = CPU.extract_bits(value, 7, 1)
            self.FRE = CPU.extract_bits(value, 8, 1)
            self.UFE = CPU.extract_bits(value, 9, 1)
            self.L2C = CPU.extract_bits(value, 10, 1)
            self.DEC = CPU.extract_bits(value, 11, 1)
            self.XNP = CPU.extract_bits(value, 13, 1)
            self.MSAEn = CPU.extract_bits(value, 27, 1)
            self.EVA = CPU.extract_bits(value, 28, 1)
            self.CV = CPU.extract_bits(value, 29, 1)
            self.K = CPU.extract_bits(value, 30, 1)
            self.M = CPU.extract_bits(value, 31, 1)
        else:
            self.valid = False

    def is_mmu_equivalent_to(self, other):
        return (
            self.valid == other.valid
            and self.MRP == other.MRP
            and self.MVH == other.MVH
            and self.EVA == other.EVA
        )

    def __repr__(self):
        return f"Config5 {hex(self.value)} => MRP:{hex(self.MRP)}, MVH:{hex(self.MVH)}, EVA:{hex(self.EVA)}"


class CPU(CPUDefault):
    @classmethod
    def from_cpu_config(cls, cpu_config, **kwargs):
        if cpu_config["bits"] == 32:
            return CPUMips32(cpu_config)
        else:
            logging.warning("Sorry :( no support for MIPS64")
            exit(1)

    def __init__(self, features):
        super(CPU, self).__init__(features)
        if self.endianness == "big":
            self.processor_features["opcode_unpack_fmt"] = ">I"
        else:
            self.processor_features["opcode_unpack_fmt"] = "<I"
        self.processor_features["ksegs"] = {}
        self.processor_features["instr_len"] = 4
        self.processor_features["R6_CPU"] = (
            CPU.extract_bits_little(self.registers_values["Config"], 10, 3) >= 2
        )
        CPU.endianness = self.endianness
        CPU.processor_features = self.processor_features
        CPU.registers_values = self.registers_values
        CPU.extract_bits = CPU.extract_bits_little


class CPUMips32(CPU):
    def __init__(self, features):
        super(CPUMips32, self).__init__(features)
        self.processor_features["ksegs"] = {
            "Kseg0": (
                0x80000000,
                0x20000000,
            ),  # Each Kseg segment start address and size
            "Kseg1": (0xA0000000, 0x20000000),
        }
        self.processor_features["kern_code_phys_end"] = 0x20000000
        self.processor_features["opcode_to_mmu_regs"] = {
            (4, 1): "ContextConfig",
            (5, 0): "PageMask",
            (5, 1): "PageGrain",
            (5, 2): "SegCtl0",
            (5, 3): "SegCtl1",
            (5, 4): "SegCtl2",
            (5, 5): "PWBase",
            (5, 6): "PWField",
            (5, 7): "PWSize",
            (6, 0): "Wired",
            (6, 6): "PWCtl",
            (16, 0): "Config",
            (16, 5): "Config5",
            # (4, 0):     "Context", # Registers not used in our analisys
            # (16, 4):    "Config4",
            # (15, 1):    "EBase",
            # (31, 2):    "KScratch0",
            # (31, 3):    "KScratch1",
            # (31, 4):    "KScratch2",
            # (31, 5):    "KScratch3",
            # (31, 6):    "KScratch4",
            # (31, 7):    "KScratch5"
        }

        self.processor_features["opcode_to_gregs"] = [
            "ZERO",
            "AT",
            "V0",
            "V1",
            "A0",
            "A1",
            "A2",
            "A3",
            "T0",
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "S0",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "T8",
            "T9",
            "K0",
            "K1",
            "GP",
            "SP",
            "FP",
            "RA",
        ]
        CPU.processor_features = self.processor_features
        CPU.registers_values = self.registers_values

    def parse_opcode(self, instr, page_addr, offset):
        opcodes = {}
        # Collect MTC0 instructions for MMU registers
        if (
            CPUMips32.extract_bits(instr, 21, 11) == 0b01000000100
            and CPUMips32.extract_bits(instr, 3, 8) == 0x0
        ):
            sel = CPUMips32.extract_bits(instr, 0, 3)
            rd = CPUMips32.extract_bits(instr, 11, 5)
            gr = CPUMips32.processor_features["opcode_to_gregs"][
                CPUMips32.extract_bits(instr, 16, 5)
            ]

            # For each address collect which coprocessor register in involved and which general register it is used to load a value
            if (rd, sel) in self.processor_features["opcode_to_mmu_regs"]:
                phy_addr = page_addr + offset
                mmu_reg = self.processor_features["opcode_to_mmu_regs"][rd, sel]
                for kseg_start, kseg_size in CPU.processor_features["ksegs"].values():
                    opcodes[phy_addr + kseg_start] = {
                        "register": mmu_reg,
                        "gpr": [gr],
                        "f_addr": -1,
                        "f_parents": set(),
                        "instruction": "MTC0",
                    }
        return opcodes

    def identify_functions_start(self, addreses):
        machine = self.machine.get_miasm_machine()
        vm = self.machine.memory.get_miasm_vmmngr()
        mdis = machine.dis_engine(bin_stream_vm(vm), loc_db=LocationDB())
        mdis.follow_call = False
        mdis.dontdis_retcall = False
        instr_len = self.processor_features["instr_len"]

        # Disable MIASM logging
        logger = logging.getLogger("asmblock")
        logger.disabled = True

        for addr in tqdm(addreses):
            # For a passed address disassemble backward as long as we do not
            # find a unconditionally return or an invalid instruction
            cur_addr = addr

            # Maximum 10000 instructions
            instructions = 0
            while True and instructions <= 10000:
                try:
                    asmcode = mdis.dis_instr(cur_addr)

                    # ERET/ERETNC/DRET/BC
                    if asmcode.name in ["BC", "DRET", "ERET", "ERETNC"]:
                        cur_addr += instr_len
                        break

                    # JR RA/JR.HB RA/J/JIC/B
                    elif asmcode.name in [
                        "B",
                        "J",
                        "JIC",
                        "JR",
                        "JR.HB",
                        "BAL",
                        "BALC",
                        "BC",
                        "JALR",
                        "JALR.HB",
                        "JALX",
                        "JIALC",
                        "JIC",
                    ]:
                        cur_addr += instr_len * 2
                        break

                    else:
                        cur_addr -= instr_len
                        instructions += 1

                except IndexError:
                    # Stop if found an invalid instruction
                    cur_addr += instr_len
                    break

            if instructions < 10000:
                addreses[addr]["f_addr"] = cur_addr

        del vm


class PhysicalMemory(PhysicalMemoryDefault):
    def get_miasm_vmmngr(self):
        if self._miasm_vm is not None:
            return self._miasm_vm

        vm = Vm()
        # Due to the existence of two Kernel unmapped segments which map kernel physical memory we need to instruct
        # MIASM to see both of them
        for region_def in tqdm(self._memregions):
            if region_def["start"] == 0:
                for kseg_name, kseg_addr_size in CPU.processor_features[
                    "ksegs"
                ].items():
                    kseg_addr, kseg_size = kseg_addr_size
                    vm.add_memory_page(
                        region_def["start"] + kseg_addr,
                        PAGE_READ | PAGE_WRITE | PAGE_EXEC,
                        region_def["fd"].read(kseg_size),
                        kseg_name,
                    )
                    region_def["fd"].seek(0)
                break

        self._miasm_vm = vm
        return self._miasm_vm


class MMU(MMUDefault):
    PAGE_SIZE = 4096
    extract_bits = MMUDefault.extract_bits_little


class MIPS32(MMU):
    pass


class MMUShell(MMUShellDefault):
    def __init__(self, completekey="tab", stdin=None, stdout=None, machine={}):
        super(MMUShell, self).__init__(completekey, stdin, stdout, machine)

        if not self.data:
            self.data = Data(
                is_mem_parsed=False,
                is_registers_found=False,
                opcodes={},
                regs_values={},
            )

    def do_parse_memory(self, args):
        """Find MMU related opcodes in dump"""
        if self.data.is_mem_parsed:
            logger.warning("Memory already parsed")
            return

        self.parse_memory()
        self.data.is_mem_parsed = True

    def parse_memory(self):
        logger.info("Look for opcodes related to MMU setup...")
        parallel_results = self.machine.apply_parallel(
            self.machine.mmu.PAGE_SIZE,
            self.machine.cpu.parse_opcodes_parallel,
            max_address=self.machine.cpu.processor_features["kern_code_phys_end"],
        )

        opcodes = {}
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            opcodes.update(result.get())

        self.data.opcodes = opcodes

    def do_find_registers_values(self, arg):
        """Find and execute MMU related functions inside the memory dump in order to extract MMU registers values"""

        if not self.data.is_mem_parsed:
            logging.warning("First parse the dump!")
            return

        if self.data.is_registers_found:
            logging.warning("Registers already searched")
            return

        logging.info("This analysis could be extremely slow!")
        logging.info("Use heuristics to find function addresses...")
        self.machine.cpu.identify_functions_start(self.data.opcodes)

        logging.info("Identify register values using data flow analysis...")

        # We use data flow analysis and merge the results
        dataflow_values = self.machine.cpu.find_registers_values_dataflow(
            self.data.opcodes, zero_registers=["ZERO"]
        )

        filtered_values = defaultdict(set)
        for register, values in dataflow_values.items():
            for value in values:
                reg_obj = CPURegMIPS.get_register_obj(register, value)
                if reg_obj.valid:
                    filtered_values[register].add(reg_obj)

        # Add default values
        for register, value in self.machine.cpu.registers_values.items():
            if (
                register
                not in self.machine.cpu.processor_features[
                    "opcode_to_mmu_regs"
                ].values()
            ):
                continue

            reg_obj = CPURegMIPS.get_register_obj(register, value)
            if reg_obj.valid and all(
                [not reg_obj.is_mmu_equivalent_to(x) for x in filtered_values[register]]
            ):
                filtered_values[register].add(reg_obj)

        self.data.regs_values = filtered_values
        self.data.is_registers_found = True

    def do_show_registers(self, args):
        """Show recovered registers values"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        for registers in sorted(self.data.regs_values.keys()):
            for register in self.data.regs_values[registers]:
                print(register)


class MMUShellGTruth(MMUShell):
    def do_show_registers_gtruth(self, args):
        """Show recovered registers values and compare them with the ground truth"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        # Collect ground truth values as last register values loaded or default ones
        mmu_regs = self.machine.cpu.processor_features["opcode_to_mmu_regs"].values()
        gvalues = {}
        for reg_name in mmu_regs:
            if reg_name in self.gtruth:
                last_reg_value = sorted(
                    self.gtruth[reg_name].keys(),
                    key=lambda x: self.gtruth[reg_name][x][1],
                )[-1]
                gvalues[reg_name] = CPURegMIPS.get_register_obj(
                    reg_name, last_reg_value
                )
            elif reg_name in self.machine.cpu.registers_values:
                gvalues[reg_name] = CPURegMIPS.get_register_obj(
                    reg_name, self.machine.cpu.registers_values[reg_name]
                )

        tps = defaultdict(list)
        fps = defaultdict(list)
        fns = {}

        tps_count = 0
        fps_count = 0

        # Check between value recovered with dataflow analisys
        for register, register_obj in gvalues.items():
            tmp_fps = []
            tmp_fps_count = 0
            for found_value in self.data.regs_values[register]:
                if register_obj.is_mmu_equivalent_to(found_value):
                    # Count only one TP per register
                    if register not in tps:
                        tps_count += 1
                    tps[register].append(found_value)
                else:
                    # Count only FP not equivalent among them
                    if all(
                        [not found_value.is_mmu_equivalent_to(x) for x in fps[register]]
                    ):
                        tmp_fps_count += 1
                    tmp_fps.append(found_value)

            # Add false negatives
            if register not in tps:
                fns[register] = register_obj
            else:  # Add false positives only if it is not a false negative
                fps[register] = tmp_fps
                fps_count += tmp_fps_count

        print("\nTrue positives")
        pprint(tps)

        print("\nFalse positives")
        pprint(fps)

        print("\nFalse negatives")
        pprint(fns)
        print(f"\nTP:{tps_count}, FP:{fps_count}, FN:{len(fns)}")
