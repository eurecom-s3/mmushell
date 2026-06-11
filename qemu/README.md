## Description

The patch `qemu_v5.0.0.patch` allows to get the ground-truth MMU values of running virtual machines through memory dumps.

## Installation

If you use Debian/Ubuntu please install the following packages :

```
build-essential git pkg-config libgtk-3-dev python3 python3-dev python3-pip python3-venv
```

Remember to always run `qemu_logger.py` or `run_qemu` script in the python3 venv created by the installer script (`build_qemu`).
