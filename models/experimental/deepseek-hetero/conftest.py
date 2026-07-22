# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""pytest bootstrap for the ``deepseek-hetero`` package.

The directory name contains a hyphen, so this tree is intentionally NOT importable
as ``models.experimental.deepseek-hetero.*``. Instead, modules import each other as
top-level names (``import bridge``, ``from transport.pcie import PcieTransport``) with
this directory on ``sys.path``.

pytest imports this conftest during collection; inserting the package root here makes
those top-level imports resolve for every test under ``tests/``. Standalone entry
points (e.g. ``demo/run_hetero.py``) perform the equivalent insert themselves.
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
