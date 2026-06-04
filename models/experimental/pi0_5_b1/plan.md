# Plan for single transformer block
- single block consists of the following
```
        normed = rms_norm(
            hidden_states,
            self.input_layernorm_weight,
            self.config.rms_norm_eps,
        )

        # Attention with residual
        attn_output, new_cache = self.attention.forward(
            normed,
            cos,
            sin,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        hidden_states = hidden_states + attn_output

        # Pre-MLP norm
        normed = rms_norm(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )

        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        hidden_states = hidden_states + mlp_output

```

## steps
1. rms norm
2. attention layer
3. residual add
4. rms norm
5. feed forward layer
6. residual add

## micro-op structure(from deepseekv3b)
1. op.py, python kernel caller
2. micro_ops/kernels_<op>_kernel.cpp, caller to the op, splits 3 cores
3. unified_kernels/kernels_<op>.hpp, definition/template of kernel
4. api/compute/op.h, blackhole llk intrinsics
