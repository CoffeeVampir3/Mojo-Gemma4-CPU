# Kernel And Modeling Cleanup Prototype

I kept this prototype outside production code. I did not inspect existing files in `prototypes`; the new files use a `codex_` prefix.

## Production Repetition Observed

The kernels repeat the same shape in most modules:

- A row or head primitive does the real SIMD work.
- A `BurstKernel` wrapper stores the same pointer set plus `start` and `end`.
- `execute()` loops over `start..end`.
- `over_range()` rebuilds the same struct with only range changed.
- The dispatcher repeats `DispatchBuffer`, rank loop, `recommended_workers`, `tile_dispatch` or `worker_range`, dispatch, and join.

The modeling layer repeats another layer of ceremony:

- Scratch bindings by `Island` and string name are gathered manually in every phase.
- `degree` and `max_worker_count` are threaded through every call.
- RMS parameters are recomputed at each call site.
- `RankBuffers` are rebuilt by hand for every allreduce and broadcast.
- GEMV/RMS/allreduce calls dominate layer code, making the actual model schedule harder to scan.

## Prototype 1: Generic Ranked Range Dispatch

`codex_kernel_dispatch_cleanup.mojo` sketches a conservative kernel-side refactor:

- `WorkRange` packages `start/end`.
- `RangePartitionedKernel` gives kernels a default `with_range()` method, so each kernel only needs a `set_range()` method.
- `RankedRangeJob` separates dispatch policy from the kernel payload.
- `dispatch_ranked_range()` centralizes rank iteration, inline threshold handling, worker selection, buffer filling, dispatch, and joins.

The example rewrites scalar multiply as:

```mojo
var job = ScalarMulJob[hidden, tp](src, dst, scalar, seq_len)
dispatch_ranked_range[tp=tp, max_worker_count=max_worker_count](job, pools)
```

That is the part worth lifting into production first. It avoids trying to get clever with reflection and keeps NUMA-rank behavior explicit.

## Prototype 2: Model Call Facade

`codex_model_call_cleanup.mojo` sketches a caller-side facade:

```mojo
BF16Ops[degree, max_worker_count].rms[hidden=C.HIDDEN](
    x_main, x_residual, body.input_norm.binding(layer_ctx), 1, self.pools)

BF16Ops[degree, max_worker_count].allreduce(x_residual, C.HIDDEN, self.pools)

BF16Ops[degree, max_worker_count].gemv[
    rows=intermediate_per_rank, cols=C.HIDDEN,
](x_residual, body.gate_proj.binding(ctx), gate, pools)
```

The goal is not to hide the model schedule. It is to remove repeated mechanical details that are already implied by `degree` and `max_worker_count`.

## Why This Shape

This follows the Mojo docs patterns that fit the current codebase:

- Use traits with comptime members for shared behavior that varies by conformer.
- Use parameterized structs/functions to keep specialization explicit.
- Avoid materializing complex comptime values in hot paths.
- Keep pointers and origins explicit at the boundary where bindings become `RankBuffers`.
- Keep the BurstPool NUMA model intact: dispatch remains rank-local and joins remain explicit.

## Migration Order I Would Use

1. Land the `RangePartitionedKernel` and `dispatch_ranked_range()` helper behind production tests.
2. Convert one low-risk kernel, likely `dispatch_scalar_mul` or `dispatch_gelu_gate_up`.
3. Convert GEMV and RMS once the helper has survived a simple path.
4. Add `BF16Ops` or a similarly named model facade after kernel dispatch signatures stabilize.
5. Then split attention and FFN model phases into small typed views so scratch binding names are centralized.

## Open Risks

- Attention kernels need a worker id in the payload, so the generic dispatcher must support worker-aware factories. The prototype does.
- Some kernels have rank-specific totals (`valid_len[r]`) and some have one total for all ranks. The `RankedRangeJob.total(rank)` hook covers both.
- Reflection could remove even the `set_range()` method, but I would not start there. The current code benefits from obvious, register-passable payload structs.
- The model facade should stay thin. If it starts owning scheduling policy, layer code will become harder to reason about rather than easier.
