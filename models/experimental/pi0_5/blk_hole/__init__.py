# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-slice PCC + D2D tests for the BH-Galaxy pipeline on a 4-chip Blackhole.

Each multi-chip slice class from ``tt/tt_bh_glx`` is tested INDIVIDUALLY on one
chip of a 4-chip mesh (native MeshShape(4,1), or a 4-chip corner of the 32-chip
galaxy), against its PyTorch reference at the same granularity. See
``mesh_setup_bh.open_blackhole_mesh`` for the auto-detecting mesh harness.
"""
