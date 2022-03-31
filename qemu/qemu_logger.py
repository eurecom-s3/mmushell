#!/usr/bin/env python3
import os
import errno
from collections import defaultdict
import argparse
import pickle
from qmp import QEMUMonitorProtocol
from signal import signal, SIGINT
from threading import Timer
from datetime import datetime
from copy import deepcopy

start_time_g = 0

class CPU:
    def __init__(self, mem_base_addr, dump_file_name, debug):
        self.regs = defaultdict(dict)
        self.debug = debug
        self.mem_base_addr = mem_base_addr
        self.dump_file_name = dump_file_name

    def parse_log_row(self, data_log, start_time):
        if self.debug:
            print(data_log)

        time_now = (datetime.now() - start_time).total_seconds()
        reg_name, reg_value = data_log.split("=")
        reg_value = int(reg_value.strip(), 16)

        if reg_value not in self.regs[reg_name]:
            self.regs[reg_name][reg_value] = (time_now, time_now)
        else:
            first_seen = self.regs[reg_name][reg_value][0]
            self.regs[reg_name][reg_value] = (first_seen, time_now)

    def dump_memory(self, qmonitor):
        # Grab the memory size and dump the memory
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]
        qmonitor.cmd("pmemsave", {"val": self.mem_base_addr,
                                  "size": memory_size,
                                  "filename": self.dump_file_name + ".dump"})


class Intel(CPU):
    def parse_log_row(self, data_log, start_time):
        if self.debug:
            print(data_log)

        time_now = (datetime.now() - start_time).total_seconds()
        reg_parts = data_log.split("|")
        reg_name, reg_value = reg_parts[0].split("=")
        reg_value = int(reg_value.strip(), 16)

        if reg_name in ("IDTR", "GDTR"):
            limit_value = int(reg_parts[1].split("=")[1].strip(), 16)
            reg_value = reg_value << 16 | limit_value

        if reg_value not in self.regs[reg_name]:
            self.regs[reg_name][reg_value] = (time_now, time_now)
        else:
            first_seen = self.regs[reg_name][reg_value][0]
            self.regs[reg_name][reg_value] = (first_seen, time_now)

    def dump_memory(self, qmonitor):
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]

        # Dump different chunks of memory
        qmonitor.cmd("pmemsave", {"val": 0x0,
                                  "size": min(memory_size, 0xC0000000),
                                  "filename": self.dump_file_name + ".dump.0"
                                  })
        if memory_size >= 0xC0000000:
            qmonitor.cmd("pmemsave", {"val": 0x100000000,
                                      "size": memory_size - 0xC0000000,
                                      "filename": self.dump_file_name + ".dump.1"
                                      })


class IntelQ35(Intel):
    def dump_memory(self, qmonitor):
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]

        # Dump different chunks of memory
        qmonitor.cmd("pmemsave", {"val": 0x0,
                                  "size": min(memory_size, 0x80000000),
                                  "filename": self.dump_file_name + ".dump.0"
                                  })
        if memory_size >= 0x80000000:
            qmonitor.cmd("pmemsave", {"val": 0x100000000,
                                      "size": memory_size - 0x80000000,
                                      "filename": self.dump_file_name + ".dump.1"
                                      })


class PPC(CPU):
    def __init__(self, mem_base_addr, dump_file_name, debug):
        super(PPC, self).__init__(mem_base_addr, dump_file_name, debug)
        dict_proto = {"U": {"value": 0, "modified": False}, "L": {"value": 0, "modified": False}}
        self._BATS = {x: deepcopy(dict_proto) for x in
                      ["DBAT0", "DBAT1", "DBAT2", "DBAT3", "IBAT0", "IBAT1", "IBAT2", "IBAT3"]}

    def parse_log_row(self, data_log, start_time):
        if self.debug:
            print(data_log)

        time_now = (datetime.now() - start_time).total_seconds()
        keys_values = data_log.split("|")
        reg_name, reg_value = keys_values[0].split("=")
        reg_value = int(reg_value.strip(), 16)

        if reg_name == "SDR1":
            _, vsid = keys_values[1].split("=")
            if reg_value not in self.regs[reg_name]:
                if vsid.strip() == "-1":
                    self.regs[reg_name][reg_value] = {"first_seen": time_now, "last_seen": time_now, "vsids": {}}
                else:
                    vsid = int(vsid.strip(), 16)
                    self.regs[reg_name][reg_value] = {"first_seen": time_now, "last_seen": time_now,
                                                      "vsids": {vsid: (time_now, time_now)}}
            else:
                self.regs[reg_name][reg_value]["last_seen"] = time_now
                if vsid.strip() != "-1":
                    vsid = int(vsid.strip(), 16)
                    if vsid not in self.regs[reg_name][reg_value]["vsids"]:
                        self.regs[reg_name][reg_value]["vsids"][vsid] = (time_now, time_now)
                    else:
                        first_seen = self.regs[reg_name][reg_value]["vsids"][vsid][0]
                        self.regs[reg_name][reg_value]["vsids"][vsid] = (first_seen, time_now)
            pass
        elif "BAT" in reg_name:
            reg_group = reg_name[0:5]
            reg_part = reg_name[5:]

            # We suppose that XBATYU and XBATYL are both updated when one of them is updated
            self._BATS[reg_group][reg_part]["value"] = reg_value
            self._BATS[reg_group][reg_part]["modified"] = True

            if self._BATS[reg_group]["U"]["modified"] and self._BATS[reg_group]["L"]["modified"]:
                regs_values = (self._BATS[reg_group]["U"]["value"], self._BATS[reg_group]["L"]["value"])
                if regs_values not in self.regs[reg_group]:
                    self.regs[reg_group][regs_values] = (time_now, time_now)
                else:
                    first_seen = self.regs[reg_group][regs_values][0]
                    self.regs[reg_group][regs_values] = (first_seen, time_now)

                self._BATS[reg_group]["U"]["modified"] = False
                self._BATS[reg_group]["L"]["modified"] = False

        else:
            if reg_value not in self.regs[reg_name]:
                self.regs[reg_name][reg_value] = (time_now, time_now)
            else:
                first_seen = self.regs[reg_name][reg_value][0]
                self.regs[reg_name][reg_value] = (first_seen, time_now)


class Arm(CPU):
    pass


class ArmVirtSecure(CPU):
    def __init__(self, mem_base_addr, dump_file_name, debug):
        super(ArmVirtSecure, self).__init__(mem_base_addr, dump_file_name, debug)
        self.secure_mem_base_addr = 0xe000000
        self.secure_memory_size = 0x01000000

    def dump_memory(self, qmonitor):
        super(ArmVirtSecure, self).dump_memory(qmonitor)

        # Dump also secure memory
        qmonitor.cmd("pmemsave", {"val": self.secure_mem_base_addr,
                                  "size": self.secure_memory_size,
                                  "filename": self.dump_file_name + "_secure.dump"
                                  })


class ARM_integratorcp(Arm):
    def dump_memory(self, qmonitor):
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]

        memory_chunks = [(0x0000000000000000, 0x000000000fffffff),
                         (0x0000000010800000, 0x0000000012ffffff),
                         (0x0000000013001000, 0x0000000013ffffff),
                         (0x0000000014800000, 0x0000000014ffffff),
                         (0x0000000015001000, 0x0000000015ffffff),
                         (0x0000000016001000, 0x0000000016ffffff),
                         (0x0000000017001000, 0x0000000017ffffff),
                         (0x0000000018001000, 0x0000000018ffffff),
                         (0x0000000019001000, 0x0000000019ffffff),
                         (0x000000001b000000, 0x000000001bffffff),
                         (0x000000001c001000, 0x000000001cffffff),
                         (0x000000001d001000, 0x00000000bfffffff),
                         (0x00000000c0001000, 0x00000000c7ffffff),
                         (0x00000000c8000010, 0x00000000c9ffffff),
                         (0x00000000ca800000, 0x00000000caffffff)]

        dumped_size = 0
        i = 0
        for i, chunk in enumerate(memory_chunks):
            dumped_chunk_size = min(memory_size - dumped_size, chunk[1] - chunk[0] + 1)
            qmonitor.cmd("pmemsave", {"val": chunk[0],
                                      "size": dumped_chunk_size,
                                      "filename": self.dump_file_name + ".dump." + str(i)
                                      })
            dumped_size += dumped_chunk_size

        if dumped_size < memory_size:
            qmonitor.cmd("pmemsave", {"val": 0xcb800000,
                                      "size": memory_size - dumped_size,
                                      "filename": self.dump_file_name + ".dump." + str(i+1)
                                      })


class ARM_raspi3(Arm):
    def dump_memory(self, qmonitor):
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]

        memory_chunks = [(0x0, 0x3f002fff),
                         (0x3f003020, 0x3f006fff),
                         (0x3f008000, 0x3f00b1ff),
                         (0x3f00b440, 0x3f00b7ff),
                         (0x3f00bc00, 0x3f0fffff),
                         (0x3f101000, 0x3f101fff),
                         (0x3f103000, 0x3f103fff),
                         (0x3f104010, 0x3f1fffff),
                         (0x3f203100, 0x3f203fff),
                         (0x3f204020, 0x3f204fff),
                         (0x3f205020, 0x3f20efff),
                         (0x3f20f080, 0x3f211fff),
                         (0x3f212008, 0x3f213fff),
                         (0x3f214100, 0x3f214fff),
                         (0x3f215100, 0x3f2fffff),
                         (0x3f300100, 0x3f5fffff),
                         (0x3f600100, 0x3f803fff),
                         (0x3f804020, 0x3f804fff),
                         (0x3f805020, 0x3f8fffff),
                         (0x3f908000, 0x3f90ffff),
                         (0x3f918000, 0x3f97ffff),
                         (0x3f981000, 0x3fdfffff),
                         (0x3fe00100, 0x3fe04fff),
                         (0x3fe05100, 0x3fffffff)]

        dumped_size = 0
        i = 0
        for i, chunk in enumerate(memory_chunks):
            dumped_chunk_size = min(memory_size - dumped_size, chunk[1] - chunk[0] + 1)
            qmonitor.cmd("pmemsave", {"val": chunk[0],
                                      "size": dumped_chunk_size,
                                      "filename": self.dump_file_name + ".dump." + str(i)
                                      })
            dumped_size += dumped_chunk_size

        if dumped_size < memory_size:
            qmonitor.cmd("pmemsave", {"val": 0xcb800000,
                                      "size": memory_size - dumped_size,
                                      "filename": self.dump_file_name + ".dump." + str(i+1)
                                      })


class RISCV(CPU):
    pass


class MIPS(CPU):
    pass


class MIPS_malta(MIPS):
    def dump_memory(self, qmonitor):
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]

        # Dump different chunks of memory
        qmonitor.cmd("pmemsave", {"val": 0x0,
                                  "size": min(memory_size, 0x10000000),
                                  "filename": self.dump_file_name + ".dump.0"
                                  })
        if memory_size >= 0x10000000:
            qmonitor.cmd("pmemsave", {"val": 0x20000000,
                                      "size": memory_size - 0x10000000,
                                      "filename": self.dump_file_name + ".dump.1"
                                      })


class MIPS_mipssim(MIPS):
    def dump_memory(self, qmonitor):
        res = qmonitor.cmd("query-memory-size-summary")
        memory_size = res["return"]["base-memory"]

        # Dump different chunks of memory
        qmonitor.cmd("pmemsave", {"val": 0x0,
                                  "size": min(memory_size, 0x1fc00000),
                                  "filename": self.dump_file_name + ".dump.0"
                                  })
        if memory_size >= 0x1fc00000:
            qmonitor.cmd("pmemsave", {"val": 0x20000000,
                                      "size": min(memory_size - 0x1fc00000, 0xe0000000),
                                      "filename": self.dump_file_name + ".dump.1"
                                      })


class POWER(CPU):
    def parse_log_row(self, data_log, start_time):
        if self.debug:
            print(data_log)

        time_now = (datetime.now() - start_time).total_seconds()
        keys_values = data_log.split("|")
        reg_name, reg_value = keys_values[0].split("=")
        reg_value = int(reg_value.strip(), 16)

        if reg_name == "SDR1":
            _, vsid = keys_values[1].split("=")
            if reg_value not in self.regs[reg_name]:
                if vsid.strip() == "-1":
                    self.regs[reg_name][reg_value] = {"first_seen": time_now, "last_seen": time_now, "vsids": {}}
                else:
                    vsid = int(vsid.strip(), 16)
                    self.regs[reg_name][reg_value] = {"first_seen": time_now, "last_seen": time_now,
                                                      "vsids": {vsid: (time_now, time_now)}}
            else:
                self.regs[reg_name][reg_value]["last_seen"] = time_now
                if vsid.strip() != "-1":
                    vsid = int(vsid.strip(), 16)
                    if vsid not in self.regs[reg_name][reg_value]["vsids"]:
                        self.regs[reg_name][reg_value]["vsids"][vsid] = (time_now, time_now)
                    else:
                        first_seen = self.regs[reg_name][reg_value]["vsids"][vsid][0]
                        self.regs[reg_name][reg_value]["vsids"][vsid] = (first_seen, time_now)
            pass

        else:
            if reg_value not in self.regs[reg_name]:
                self.regs[reg_name][reg_value] = (time_now, time_now)
            else:
                first_seen = self.regs[reg_name][reg_value][0]
                self.regs[reg_name][reg_value] = (first_seen, time_now)


parser = argparse.ArgumentParser(description='You have to call QEMU with "-qmp tcp:HOST:PORT,server,'
                                             'nowait -d fossil -D pipe_file" options and WITHOUT "-enable-kvm" option')
parser.add_argument("pipe_file", help="PIPE for QEMU log file", type=str)
parser.add_argument("qmp", help="QEMU QMP channel (host:port)", type=str)
parser.add_argument("prefix_filename", help="Prefix for dump and .regs file.", type=str)
parser.add_argument("--debug", help="Print debug info", action="store_true", default=False)
parser.add_argument("--timer", help="Shutdown machine after N seconds", type=int, default=0)
subparser = parser.add_subparsers(required=True, help="Architectures", dest="arch")
parser_intel = subparser.add_parser("intel")
parser_intelq35 = subparser.add_parser("intel_q35")
parser_ppc = subparser.add_parser("ppc")
parser_arm_virt = subparser.add_parser("arm_virt")
parser_arm_virt_secure = subparser.add_parser("arm_virt_secure")
parser_arm_vexpress_a9 = subparser.add_parser("arm_vexpress_a9")
parser_arm_vexpress_a15 = subparser.add_parser("arm_vexpress_a15")
parser_arm_integratorcp = subparser.add_parser("arm_integratorcp")
parser_arm_raspi3 = subparser.add_parser("arm_raspi3")
parser_riscv = subparser.add_parser("riscv")
parser_mips = subparser.add_parser("mips")
parser_mips_malta = subparser.add_parser("mips_malta")
parser_mips_mipssim = subparser.add_parser("mips_mipssim")
parser_power = subparser.add_parser("power")

args = parser.parse_args()

try:
    qemu_qmp = args.qmp.split(":")
    qemu_qmp[1] = int(qemu_qmp[1])
    qemu_qmp = tuple(qemu_qmp)
except Exception as e:
    parser.error("Invalid QMP channel format!")
    exit(1)


try:
    regs_file = open(args.prefix_filename + ".regs", "wb")
except Exception as e:
    print(e)
    print("Unable to open output file!")
    exit(1)

if args.arch == "intel":
    log_class = Intel(0, args.prefix_filename, args.debug)
elif args.arch == "intel_q35":
    log_class = IntelQ35(0, args.prefix_filename, args.debug)

elif args.arch == "ppc":
    log_class = PPC(0, args.prefix_filename, args.debug)

elif args.arch == "arm_virt":
    log_class = Arm(0x40000000, args.prefix_filename, args.debug)
elif args.arch == "arm_virt_secure":
    log_class = ArmVirtSecure(0x40000000, args.prefix_filename, args.debug)
elif args.arch == "arm_vexpress_a9":
    log_class = Arm(0x60000000, args.prefix_filename, args.debug)
elif args.arch == "arm_vexpress_a15":
    log_class = Arm(0x80000000, args.prefix_filename, args.debug)
elif args.arch == "arm_integratorcp":
    log_class = ARM_integratorcp(0x00000000, args.prefix_filename, args.debug)
elif args.arch == "arm_raspi3":
    log_class = ARM_raspi3(0x00000000, args.prefix_filename, args.debug)

elif args.arch == "riscv":
    log_class = RISCV(0x80000000, args.prefix_filename, args.debug)

elif args.arch == "mips":
    log_class = MIPS(0, args.prefix_filename, args.debug)
elif args.arch == "mips_malta":
    log_class = MIPS_malta(0, args.prefix_filename, args.debug)
elif args.arch == "mips_mipssim":
    log_class = MIPS_mipssim(0, args.prefix_filename, args.debug)

elif args.arch == "power":
    log_class = POWER(0, args.prefix_filename, args.debug)

else:
    print("Invalid architecture")
    exit(1)

qemu_fifo = args.pipe_file
qemu_monitor = None
fifo = None


def ctrl_c_handler(sig, frame):
    uptime = (datetime.now() - start_time_g).total_seconds()
    print("\n\nDump the memory, save registers and shutdown the machine")

    # Dump the memory
    log_class.dump_memory(qemu_monitor)

    # Save data
    log_class.regs["uptime"] = uptime
    pickle.dump(log_class.regs, regs_file)
    qemu_monitor.close()
    regs_file.close()

    print("Done!")
    exit(0)


def timer_handler():
    os.kill(os.getpid(), SIGINT)


# Open the QEMU log FIFO
try:
    os.mkfifo(qemu_fifo)
except OSError as oe:
    if oe.errno != errno.EEXIST:
        print(oe)
        exit(1)


print("Waiting for QEMU logs in FIFO...")
already_receive = False
fifo = open(qemu_fifo)
start_time_g = datetime.now()

for data in fifo:
    if not already_receive:
        print("QEMU connected!")
        already_receive = True

        # Try to open the QMP channel
        try:
            qemu_monitor = QEMUMonitorProtocol(qemu_qmp)
            qemu_monitor.connect()
        except Exception as e:
            regs_file.close()
            print(e)
            print("Impossible to connect to QEMU QMP channel!")
            exit(1)
        print("QEMU QMP connected!")

        signal(SIGINT, ctrl_c_handler)
        if args.timer > 0:
            t = Timer(args.timer, timer_handler)
            t.start()
            print("After {} seconds it dump the memory, save the registers, "
                  "and shutdown the machine. So wait...".format(str(args.timer)))
        else:
            print("Press CTRL-C to dump the memory, save the registers, and shutdown the machine")

    log_class.parse_log_row(data, start_time_g)
