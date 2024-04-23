#!/usr/bin/env python3
import argparse
import logging
import importlib
import yaml

from cerberus import Validator

# Set logging configuration
logger = logging.getLogger(__name__)

# Schema for YAML configuration file
machine_yaml_schema = {
    "cpu": {
        "required": True,
        "type": "dict",
        "schema": {
            "architecture": {"required": True, "type": "string", "min": 1},
            "endianness": {"required": True, "type": "string", "min": 1},
            "bits": {"required": True, "type": "integer", "allowed": [32, 64]},
            "processor_features": {"required": False, "type": "dict"},
            "registers_values": {
                "required": False,
                "type": "dict",
                "keysrules": {"type": "string", "min": 1},
                "valuesrules": {"type": "integer"},
            },
        },
    },
    "mmu": {
        "required": True,
        "type": "dict",
        "schema": {"mode": {"required": True, "type": "string", "min": 1}},
    },
    "memspace": {
        "required": True,
        "type": "dict",
        "schema": {
            "ram": {
                "required": True,
                "type": "list",
                "minlength": 1,
                "schema": {
                    "type": "dict",
                    "schema": {
                        "start": {
                            "required": True,
                            "type": "integer",
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                        },
                        "end": {
                            "required": True,
                            "type": "integer",
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                        },
                        "dumpfile": {"required": True, "type": "string", "min": 0},
                    },
                },
            },
            "not_ram": {
                "required": True,
                "type": "list",
                "minlength": 1,
                "schema": {
                    "type": "dict",
                    "schema": {
                        "start": {
                            "required": True,
                            "type": "integer",
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                        },
                        "end": {
                            "required": True,
                            "type": "integer",
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                        },
                    },
                },
            },
        },
    },
}


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "MACHINE_CONFIG",
        help="YAML file describing the machine",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--gtruth",
        help="Ground truth from QEMU registers",
        type=argparse.FileType("rb", 0),
        default=None,
    )
    parser.add_argument(
        "--session",
        help="Data file of a previous MMUShell session",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug", help="Enable debug output", action="store_true", default=False
    )
    args = parser.parse_args()

    # Set logging system
    fmt = "%(msg)s"
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)

    # Load the machine configuration YAML file
    try:
        machine_config = yaml.load(args.MACHINE_CONFIG, Loader=yaml.FullLoader)
        args.MACHINE_CONFIG.close()
    except Exception as e:
        logger.fatal("Malformed YAML file: {}".format(e))
        exit(1)

    # Validate YAML schema
    yaml_validator = Validator(allow_unknown=True)
    if not yaml_validator.validate(machine_config, machine_yaml_schema):
        logger.fatal("Invalid YAML file. Error:" + str(yaml_validator.errors))
        exit(1)

    # Create the Machine class
    try:
        architecture_module = importlib.import_module(
            "architectures." + machine_config["cpu"]["architecture"]
        )
    except ModuleNotFoundError:
        logger.fatal("Unkown architecture!")
        exit(1)

    # Create a Machine starting from the parsed configuration
    machine = architecture_module.Machine.from_machine_config(machine_config)

    # Launch the interactive shell
    if args.gtruth:
        shell = architecture_module.MMUShellGTruth(machine=machine)
    else:
        shell = architecture_module.MMUShell(machine=machine)

    # Load ground truth (if passed)
    if args.gtruth:
        shell.load_gtruth(args.gtruth)

    # Load previous data (if passed)
    if args.session:
        shell.reload_data_from_file(args.session)

    shell.cmdloop()


if __name__ == "__main__":
    main()
