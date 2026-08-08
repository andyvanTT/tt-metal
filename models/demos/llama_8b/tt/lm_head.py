# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


def apply_lofi_lm_head(model_instance):
    """Re-configure a constructed LM head for LoFi math fidelity.

    The original branch changed LMHead.__init__ to build the compute kernel
    config with LoFi instead of HiFi2. The config is consumed lazily on the
    first forward pass, so mutating it on an already-constructed module is
    equivalent and avoids forking the whole LMHead class.
    """
    model_instance.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
