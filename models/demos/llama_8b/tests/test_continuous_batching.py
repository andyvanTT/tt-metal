# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Llama-8B continuous-batching fork.

TODO: add a hardware-in-the-loop test that exercises non-contiguous
``empty_slots`` (e.g. slots [0, 2, 5]) through the forked
``prefill_forward_text`` / ``decode_forward`` paths and verifies that
request rows are packed positionally while the KV-cache page table is
still indexed by the request's own slot.

A pure-Python regression test for ``_get_prefill_user_page_table`` can be
added here once the helper is refactored to not depend on ``ttnn`` tensors.
"""
