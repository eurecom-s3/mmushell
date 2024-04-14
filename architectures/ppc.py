from architectures.generic import Machine as MachineDefault
from architectures.generic import CPU as CPUDefault
from architectures.generic import PhysicalMemory as PhysicalMemoryDefault
from architectures.generic import MMUShell as MMUShellDefault
from architectures.generic import MMU as MMUDefault
from architectures.generic import CPUReg
import logging
from prettytable import PrettyTable
from dataclasses import dataclass
from tqdm import tqdm
from struct import unpack, iter_unpack
from collections import defaultdict
from miasm.analysis.machine import Machine as MIASMMachine
from miasm.core.bin_stream import bin_stream_vm
from miasm.core.locationdb import LocationDB
from copy import deepcopy
from pprint import pprint
from time import sleep
from random import uniform
from copy import deepcopy, copy
from math import log2
import portion

logger = logging.getLogger(__name__)


@dataclass
class Data:
    is_mem_parsed: bool
    is_registers_found: bool
    opcodes: dict
    regs_values: dict
    htables: dict


class CPURegPPC(CPUReg):
    @classmethod
    def get_register_obj(cls, reg_name, value):
        # It exists multiple BAT registers
        if "BAT" in reg_name:
            if "U" in reg_name:
                return BATU(value, reg_name)
            else:
                return BATL(value, reg_name)
        else:
            return SDR1(value)


class SDR1(CPURegPPC):
    def is_valid(self, value):
        return CPU.extract_bits(value, 16, 7) == 0x0

    def __init__(self, value):
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.address = CPU.extract_bits(value, 0, 16) << 16
            self.size = 1 << (int(log2(CPU.extract_bits(value, 23, 9) + 1)) + 16)
        else:
            self.valid = False

    def __repr__(self):
        return f"SDR1 {hex(self.value)} => Address:{hex(self.address)}, Size:{hex(self.size)}"


class BATU(CPURegPPC):
    def is_valid(self, value):
        return CPU.extract_bits(value, 15, 4) == 0x0

    def __init__(self, value, name):
        self.bat_name = name
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.bepi = CPU.extract_bits(value, 0, 15)
            self.bl = CPU.extract_bits(value, 19, 11)
            self.vs = CPU.extract_bits(value, 30, 1)
            self.vp = CPU.extract_bits(value, 31, 1)
        else:
            self.valid = False

    def __repr__(self):
        return f"{self.bat_name} {hex(self.value)} => BEPI:{hex(self.bepi)}, BL:{hex(self.bl)}, VS:{hex(self.bl)}, VP:{hex(self.bl)}"

    def __eq__(self, other):
        return self.value == other.value and self.bat_name == other.bat_name

    def __hash__(self):
        return hash((self.value, self.bat_name))


class BATL(CPURegPPC):
    def is_valid(self, value):
        return (
            CPU.extract_bits(value, 15, 10) == 0x0
            and CPU.extract_bits(value, 29, 1) == 0
        )

    def __init__(self, value, name):
        self.bat_name = name
        self.value = value
        if self.is_valid(value):
            self.valid = True
            self.brpn = CPU.extract_bits(value, 0, 15)
            self.pp = CPU.extract_bits(value, 30, 2)
        else:
            self.valid = False

    def __repr__(self):
        return f"{self.bat_name} {hex(self.value)} => BRPN:{hex(self.brpn)}, PP:{hex(self.pp)}"

    def __eq__(self, other):
        return self.value == other.value and self.bat_name == other.bat_name

    def __hash__(self):
        return hash((self.value, self.bat_name))


class PTE32:
    entry_name = "PTE32"
    labels = [
        "Address:",
        "VSID:",
        "RPN:",
        "API:" "Secondary hash:",
        "Referenced:",
        "Changed:",
        "WIMG:",
        "PP:",
    ]
    size = 4
    addr_fmt = "0x{:08x}"

    def __init__(self, address, vsid, h, api, rpn, r, c, wimg, pp):
        self.address = address
        self.vsid = vsid
        self.h = h
        self.api = api
        self.rpn = rpn
        self.r = r
        self.c = c
        self.wimg = wimg
        self.pp = pp

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
            hex(self.vsid),
            hex(self.rpn),
            hex(self.api),
            bool(self.h),
            bool(self.r),
            bool(self.c),
            bin(self.wimg),
            hex(self.pp),
        ]

    def entry_resume_stringified(self):
        res = self.entry_resume()
        res[0] = self.addr_fmt.format(res[0])
        for idx, r in enumerate(res[1:], start=1):
            res[idx] = str(r)
        return res


class HashTable:
    def __init__(self, address, size, ptegs):
        self.address = address
        self.size = size
        self.ptegs = ptegs

    table_fields = [
        "Entry address",
        "VSID",
        "RPN",
        "API",
        "Secondary hash",
        "Referenced",
        "Changed",
        "WIMG",
        "PP",
    ]
    addr_fmt = "0x{:08x}"

    def __repr__(self):
        table = PrettyTable()
        table.field_names = self.table_fields

        for pteg in self.ptegs.values():
            for entry_obj in pteg:
                entry_resume = entry_obj.entry_resume()
                entry_resume[0] = self.addr_fmt.format(entry_resume[0])
                table.add_row(entry_resume)

        table.sortby = "Entry address"
        return str(table)


class PhysicalMemory(PhysicalMemoryDefault):
    pass


class CPU(CPUDefault):
    @classmethod
    def from_cpu_config(cls, cpu_config, **kwargs):
        if cpu_config["bits"] == 32:
            return CPUPPC32(cpu_config)
        else:
            logging.warning("Sorry :( no support for POWER")
            exit(1)

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


class CPUPPC32(CPU):
    def __init__(self, features):
        super(CPUPPC32, self).__init__(features)
        self.processor_features["opcode_to_mmu_regs"] = {
            0: "SR0",
            1: "SR1",
            2: "SR2",
            3: "SR3",
            4: "SR4",
            5: "SR5",
            6: "SR6",
            7: "SR7",
            8: "SR8",
            9: "SR9",
            10: "SR10",
            11: "SR11",
            12: "SR12",
            13: "SR13",
            14: "SR14",
            15: "SR15",
            25: "SDR1",
            528: "IBAT0U",
            529: "IBAT0L",
            530: "IBAT1U",
            531: "IBAT1L",
            532: "IBAT2U",
            533: "IBAT2L",
            534: "IBAT3U",
            535: "IBAT3L",
            536: "DBAT0U",
            537: "DBAT0L",
            538: "DBAT1U",
            539: "DBAT1L",
            540: "DBAT2U",
            541: "DBAT2L",
            542: "DBAT3U",
            543: "DBAT3L",
        }

        self.processor_features["opcode_to_gregs"] = [
            "R{}".format(str(i)) for i in range(32)
        ]
        CPU.processor_features = self.processor_features
        CPU.registers_values = self.registers_values

    def parse_opcode(self, instr, page_addr, offset):
        # Exclude all possible instructions which are not compatible with MTSPR, MTSR, MTSRIN
        if (
            CPUPPC32.extract_bits(instr, 31, 1) != 0
            or CPUPPC32.extract_bits(instr, 0, 6) != 31
        ):
            return {}

        # Look for MTSPR (SDR1 and BATs)
        if CPUPPC32.extract_bits(instr, 21, 10) == 467:
            spr = (CPUPPC32.extract_bits(instr, 16, 5) << 5) + CPUPPC32.extract_bits(
                instr, 11, 5
            )
            if spr == 25 or 528 <= spr <= 543:
                gr = CPUPPC32.extract_bits(instr, 6, 5)
                addr = page_addr + offset
                return {
                    addr: {
                        "register": self.processor_features["opcode_to_mmu_regs"][spr],
                        "gpr": [self.processor_features["opcode_to_gregs"][gr]],
                        "f_addr": -1,
                        "f_parents": set(),
                        "instruction": "MTSPR",
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

                    # RET: BLR, BLRL, BCTRL
                    # JMP: B, BA, BCTR
                    if asmcode.name in [
                        "BA",
                        "BCTR",
                        "BLR",
                        "BLRL",
                        "BCTRL",
                        "BL",
                        "BLA",
                        "BCA",
                        "BCL",
                        "BCLA",
                        "BCLR",
                        "BCLRL",
                        "BCCTR",
                        "BCCTRL",
                    ]:
                        cur_addr += instr_len
                        break

                    # Return from interrupt, trap handler, system call
                    if asmcode.name in ["RFI", "TWI", "TW", "SC"]:
                        cur_addr += instr_len
                        break
                    else:
                        cur_addr -= instr_len
                        instructions += 1

                except Exception as e:
                    cur_addr += instr_len
                    break

            if instructions < 10000:
                addreses[addr]["f_addr"] = cur_addr
        del vm


class Machine(MachineDefault):
    def get_miasm_machine(self):
        mn_s = (
            "ppc" + str(self.cpu.bits) + ("b" if self.cpu.endianness == "big" else "l")
        )
        return MIASMMachine(mn_s)


class MMU(MMUDefault):
    pass


class PPC32(MMU):
    PAGE_SIZE = 4096
    HTABLE_MIN_BIT_SIZE = 16
    HTABLE_MAX_BIT_SIZE = 25
    HTABLE_MIN_SIZE = 1 << HTABLE_MIN_BIT_SIZE

    def __init__(self, mmu_config):
        super(PPC32, self).__init__(mmu_config)
        if CPU.endianness == "big":
            PPC32.pte_format = ">2I"
            PPC32.extract_bits = PPC32.extract_bits_big
        else:
            PPC32.pte_format = "<2I"
            PPC32.extract_bits = PPC32.extract_bits_little

    def parse_htable_opcodes_parallel(self, addresses, frame_size, pidx, **kwargs):
        # Parse hash tables fragments and opcodes at the same time
        sleep(uniform(pidx, pidx + 1) // 1000)

        opcodes = {}
        mm = copy(self.machine.memory)
        mm.reopen()

        # Cicle over every frame
        tot_elems, iterator = addresses
        fragments = []
        opcodes = {}
        htable_size = self.machine.mmu.HTABLE_MIN_SIZE
        instr_len = CPU.processor_features["instr_len"]
        for frame_addr in tqdm(iterator, position=-pidx, total=tot_elems, leave=False):
            frame_buf = mm.get_data(frame_addr, htable_size)

            # Parse the hash table fragment
            frame_obj = self.parse_hash_table(frame_buf, htable_size, frame_addr)
            if frame_obj is not None:
                fragments.append(frame_obj)

            # Parse opcodes
            for idx, opcode in enumerate(
                iter_unpack(
                    self.machine.cpu.processor_features["opcode_unpack_fmt"], frame_buf
                )
            ):
                opcode = opcode[0]
                offset = idx * instr_len
                opcodes.update(
                    self.machine.cpu.parse_opcode(opcode, frame_addr, offset)
                )

        return fragments, opcodes

    def collect_htable_framents_opcodes(self):
        logger.info("Look for hash tables fragments and opcodes...")
        parallel_results = self.machine.apply_parallel(
            self.machine.mmu.HTABLE_MIN_SIZE, self.parse_htable_opcodes_parallel
        )

        opcodes = {}
        htables = defaultdict(list)
        logger.info("Reaggregate threads data...")
        for result in parallel_results:
            htables_th, opcodes_th = result.get()
            htables[self.machine.mmu.HTABLE_MIN_SIZE].extend(htables_th)
            opcodes.update(opcodes_th)

        return htables, opcodes

    def parse_hash_table(self, frame_buf, frame_size, frame_addr):
        ptegs = defaultdict(set)

        # Unpack records inside the frame
        for entry_idx, entry in enumerate(iter_unpack(self.pte_format, frame_buf)):
            entry_obj = self.classify_htable_entry(entry, frame_addr + entry_idx * 8)

            if entry_obj is False:
                continue

            if entry_obj is None:
                return None

            # No duplicates allowed in a PTEG
            pteg_addr = (frame_addr + entry_idx * 8) - (
                (frame_addr + entry_idx * 8) % 64
            )
            if pteg_addr in ptegs and entry_obj in ptegs[pteg_addr]:
                return None

            ptegs[pteg_addr].add(entry_obj)

        return HashTable(frame_addr, frame_size, ptegs)

    def classify_htable_entry(self, entry, entry_addr):
        # If BIT 0 Word 0 = 0 is EMPTY
        if not PPC32.extract_bits(entry[0], 0, 1):
            return False

        # ###################################################
        # PPC specs say that these bits are ignored not forced to be 0s!
        #
        # # BITs 20:22 and BIT 29 Word 1 must be 0
        # if PPC32.extract_bits(entry[1], 29, 1) or PPC32.extract_bits(entry[1], 20, 3):
        #     return None
        #####################################################

        # The physical page number must be contained in valid physical memory regions
        rpn = PPC32.extract_bits(entry[1], 0, 20)
        rpn_mod = rpn << 12
        if rpn_mod not in self.machine.memory.physpace["defined_regions"]:
            return None

        reference_bit = PPC32.extract_bits(entry[1], 23, 1)
        change_bit = PPC32.extract_bits(entry[1], 24, 1)

        # Invalid combination of R and C bit
        if not reference_bit and change_bit:
            return None

        vsid = PPC32.extract_bits(entry[0], 1, 24)
        api = PPC32.extract_bits(entry[0], 26, 6)
        h = PPC32.extract_bits(entry[0], 25, 1)
        wimg = PPC32.extract_bits(entry[1], 25, 4)
        pp = PPC32.extract_bits(entry[1], 30, 2)

        return PTE32(entry_addr, vsid, h, api, rpn, reference_bit, change_bit, wimg, pp)

    def glue_htable_fragments(self, fragments):
        logging.info("Glueing Hash tables fragments...")
        # Reorder fragments because it must be used in order
        fragments[self.HTABLE_MIN_SIZE].sort(key=lambda x: x.address)
        htables = deepcopy(fragments)
        low_frames = htables[1 << self.HTABLE_MIN_BIT_SIZE]

        # Starting from fragments with lower size, check if it is possible to form bigger ones aggregating two halves
        for i in tqdm(
            range(self.HTABLE_MIN_BIT_SIZE + 1, self.HTABLE_MAX_BIT_SIZE + 1)
        ):
            htable_size = 1 << i
            low_frames_size = 1 << (i - 1)

            for htable_idx, htable in enumerate(low_frames[:-1]):
                if htable.address % htable_size != 0:
                    continue

                # Check if the other half is present and aggregate them
                if (
                    htables[low_frames_size][htable_idx + 1].address
                    == htable.address + low_frames_size
                ):
                    nxt_htable = htables[low_frames_size][htable_idx + 1]

                    pteg_c = deepcopy(htable.ptegs)
                    pteg_c.update(nxt_htable.ptegs)

                    htables[htable_size].append(
                        HashTable(
                            address=htable.address, size=htable_size, ptegs=pteg_c
                        )
                    )

            htables[htable_size].sort(key=lambda x: x.address)
            if htables[htable_size]:
                low_frames = htables[htable_size]
            else:
                low_frames = []

        # Remove completely empty tables
        for table_size in list(htables.keys()):
            for table in list(htables[table_size]):
                if not len(table.ptegs):
                    htables[table_size].remove(table)
            if not len(htables[table_size]):
                htables.pop(table_size)

        return htables

    def filter_htables(self, htables):
        logging.info("Filtering...")
        final_candidates = defaultdict(list)  # deepcopy(htables)
        already_visited = portion.empty()
        entropies = []

        # Start from table of big size
        for table_size in reversed(list(htables.keys())):
            for table_obj in tqdm(htables[table_size]):
                # If a valid bigger table contains the little one remove the little one
                if table_obj.address in already_visited:
                    continue

                rpn_probabilities = defaultdict(int)
                total_rpn = 0
                vsids = set()
                phypages = set()

                # Calculate the probability of every RPN in the table,
                # collect all VSID of the table and all the RPN
                try:
                    for pteg in table_obj.ptegs.values():
                        for pte in pteg:
                            vsids.add(pte.vsid)
                            phypages.add(pte.rpn)
                            rpn_probabilities[pte.rpn] += 1
                            total_rpn += 1

                            # If the hash validation fails for some PTE discard the table
                            if not self.validate_entry_by_hash(
                                pte, table_obj.address, table_obj.size, pte.address
                            ):
                                raise UserWarning
                except UserWarning:
                    continue

                # HEURISTIC
                # If the table contains only one VSID is noise
                if len(vsids) == 1:
                    continue

                # HEURISTIC
                # If the table contains only one distinct physical page address
                if len(phypages) == 1:
                    continue

                # Calculate RPN entropy starting from RPN probabilities for the table
                table_entropy = 0
                for rpn_count in rpn_probabilities.values():
                    table_entropy -= rpn_count / total_rpn * log2(rpn_count / total_rpn)
                entropies.append([table_size, table_obj, table_entropy])

                already_visited |= portion.closedopen(
                    table_obj.address, table_obj.address + table_obj.size
                )
                final_candidates[table_size].append(table_obj)

        # HEURISTIC: Filter for RPN entropy: cut-off at 80% of the maximum entropy,
        # if a table has an entropy below the cut-off, discard it
        entropies.sort(key=lambda x: x[2], reverse=True)
        cutoff = entropies[0][2] * 0.8
        to_be_removed = [x for x in entropies if x[2] < cutoff]
        for table_size, table_obj, _ in to_be_removed:
            final_candidates[table_size].remove(table_obj)

        return final_candidates

    def validate_entry_by_hash(
        self, entry_obj, htable_addr, htable_size, pteg_addr_entry
    ):
        # We have only a part of the page index (9 bit)
        vsid_reduced = CPUPPC32.extract_bits_little(entry_obj.vsid, 10, 9)

        # Calculate the hash of partial data
        hash_value = vsid_reduced ^ entry_obj.api
        # Secondary hash function implemented with complement's one
        if entry_obj.h:
            hash_value = (1 << 9) - 1 - hash_value

        htabmask = (1 << (int(log2(htable_size)) - 16)) - 1
        htaborg_up = htable_addr >> 25
        htaborg_down = CPUPPC32.extract_bits_little(htable_addr >> 16, 0, 9)

        # Compute the AND and OR with SDR1 fields
        selector_middle = (hash_value & htabmask) | htaborg_down

        # Calculate the PTEG physical truncated address
        incomplete_pteg_addr = (htaborg_up << 25) + (selector_middle << 16)

        return (incomplete_pteg_addr >> 16 << 16) == (pteg_addr_entry >> 16 << 16)


class MMUShell(MMUShellDefault):
    def __init__(self, completekey="tab", stdin=None, stdout=None, machine={}):
        super(MMUShell, self).__init__(completekey, stdin, stdout, machine)

        if not self.data:
            self.data = Data(
                is_mem_parsed=False,
                is_registers_found=False,
                opcodes={},
                regs_values={},
                htables={},
            )

    def do_parse_memory(self, args):
        """Parse memory to find opcode MMU related and hash tables"""
        if self.data.is_mem_parsed:
            logger.warning("Memory already parsed")
            return

        self.parse_memory()
        self.data.is_mem_parsed = True

    def parse_memory(self):
        # Collect opcodes and hash table of the minium size
        (
            fragments,
            self.data.opcodes,
        ) = self.machine.mmu.collect_htable_framents_opcodes()

        # Glue hash table
        htables = self.machine.mmu.glue_htable_fragments(fragments)

        # Filtering results
        self.data.htables = self.machine.mmu.filter_htables(htables)

    def do_show_hashtables(self, args):
        """Show hash tables found"""
        if not self.data.is_mem_parsed:
            logger.warning("Please, parse the memory first")
            return

        table = PrettyTable()
        table.field_names = ["Address", "Size"]

        for size in sorted(self.data.htables.keys()):
            for htable in self.data.htables[size]:
                table.add_row([hex(htable.address), hex(size)])

        print(table)

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
                reg_obj = CPURegPPC.get_register_obj(register, value)
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

            reg_obj = CPURegPPC.get_register_obj(register, value)
            if reg_obj.valid and all(
                [not reg_obj.is_mmu_equivalent_to(x) for x in filtered_values[register]]
            ):
                filtered_values[register].add(reg_obj)

        self.data.regs_values = filtered_values
        self.data.is_registers_found = True

    def do_show_registers(self, args):
        """Show registers values found"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        for registers in sorted(self.data.regs_values.keys()):
            for register in self.data.regs_values[registers]:
                print(register)

    def do_show_hashtable(self, arg):
        "Parse a Hash Table of a given size"
        "Usage: show_hashtable ADDRESS size"

        arg = arg.split()
        if len(arg) < 2:
            logging.error("Missing parameter")
            return

        try:
            address = self.parse_int(arg[0])
            size = self.parse_int(arg[1])
        except ValueError:
            logging.error("Invalid format")
            return

        if address is None:
            logging.error("Invalid address format")
            return

        if address not in self.machine.memory:
            logging.error("Address not in memory address space")
            return

        valid_sizes = [
            65536,
            131072,
            262144,
            524288,
            1048576,
            2097152,
            4194304,
            8388608,
            16777216,
            33554432,
        ]
        if size not in valid_sizes:
            logging.error(f"Invalid order table. Valid sizes are {valid_sizes}")
            return

        frame_buf = self.machine.memory.get_data(address, size)
        table = self.machine.mmu.parse_hash_table(frame_buf, size, address)
        if table is None:
            logging.warning("Table not present or malformed")
        else:
            print(table)


class MMUShellGTruth(MMUShell):
    def do_show_hashtables_gtruth(self, args):
        """Show hash tables found and compare them with the ground truth"""
        if not self.data.is_mem_parsed:
            logger.warning("Please, parse the memory first")
            return

        table = PrettyTable()
        table.field_names = [
            "Address",
            "Size",
            "Found",
            "Correct size",
            "First seen",
            "Last seen",
        ]

        # Collect valid true table
        valids = {}
        for sdr1_value, sdr1_data in self.gtruth["SDR1"].items():
            sdr1_obj = SDR1(sdr1_value)
            if not sdr1_obj.valid:
                continue
            valids[sdr1_obj.address] = [
                sdr1_obj.size,
                sdr1_data["first_seen"],
                sdr1_data["last_seen"],
            ]

        # MMUShell found values
        found = {}
        for size in self.data.htables:
            for table_obj in self.data.htables[size]:
                found[table_obj.address] = [
                    table_obj.size,
                    "False positive",
                    "False positive",
                ]

        already_visited = set()
        for k, v in valids.items():
            table.add_row(
                [
                    hex(k),
                    hex(v[0]),
                    "X" if k in found else "",
                    "X"
                    if v[0]
                    == found.get(
                        k,
                        [
                            None,
                        ],
                    )[0]
                    else "",
                    v[1],
                    v[2],
                ]
            )
            already_visited.add((k, v[0]))

        fps = 0
        for k, v in found.items():
            if (k, v[0]) in already_visited:
                continue
            table.add_row([hex(k), hex(v[0]), "", "", v[1], v[2]])
            fps += 1

        print(table)
        print(f"FP: {fps}")

    def do_show_registers_gtruth(self, args):
        """Show registers value retrieved and compare with the ground truth"""
        if not self.data.is_registers_found:
            logging.info("Please, find them first!")
            return

        # Check if the last value of SDR1 was found
        last_sdr1 = SDR1(
            sorted(
                self.gtruth["SDR1"].keys(),
                key=lambda x: self.gtruth["SDR1"][x]["last_seen"],
                reverse=True,
            )[0]
        )
        print(f"Correct SDR1 value: {last_sdr1}")
        print(
            "SDR1 correct value... {}FOUND".format(
                "" if last_sdr1 in self.data.regs_values["SDR1"] else "NOT "
            )
        )

        # Found last BAT registers used by the system
        bats_found = {}
        for t in ["I", "D"]:
            for i in range(4):
                reg_name = t + "BAT" + str(i)
                batu_v, batl_v = sorted(
                    self.gtruth[reg_name].keys(),
                    key=lambda x: self.gtruth[reg_name][x][1],
                    reverse=True,
                )[0]
                bats_found[reg_name + "U"] = BATU(batu_v, reg_name + "U")
                bats_found[reg_name + "L"] = BATL(batl_v, reg_name + "L")

        # Check if values are found
        for reg_name in bats_found:
            print(
                "{} correct value... {}FOUND\t\t{}".format(
                    reg_name,
                    ""
                    if bats_found[reg_name] in self.data.regs_values[reg_name]
                    else "NOT ",
                    bats_found[reg_name],
                )
            )
