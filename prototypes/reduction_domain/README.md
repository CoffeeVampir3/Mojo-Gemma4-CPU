# Reduction Domain Prototypes

These are intentionally small probes for the collective/reduction design. They are not production modules.

`writeback_contract_proto.mojo` isolates the operation/writeback split. The operation is open-set (`Sum`, `Product`) and the writeback is a separate contract (`StoreReduced`, `AddIntoDest`).

`attention_contract_proto.mojo` models placement as a layer contract rather than a single atomic TP/CP/EP descriptor. It captures the awkward Gemma-style case where local attention has no context axis, while full attention with two heads cannot shard heads and must fall back to context.

`dispatch_policy_proto.mojo` models byte-driven dispatch with safety capabilities. It makes the important distinction that an in-place store may be small in bytes but still not legal for one-worker-per-domain small dispatch.

`burst_kernel_contract_proto.mojo` puts the operation/writeback split inside a `BurstKernel`, matching the mailbox constraint without carrying a runtime callable in the payload.

`batched_variadic_array_proto.mojo` groups a homogeneous set of reductions behind one dispatch. The call site accepts a variadic list of `ReductionDesc` values, builds one descriptor array, and dispatches one kernel carrying only a config pointer.

`composite_variadic_groups_proto.mojo` groups heterogeneous reduction groups behind one dispatch. A variadic type list of `GroupPlan[...]` values composes multiple op/writeback contracts over the same descriptor buffer.

`function_parameter_scalar_proto.mojo` is the minimal function-parameter sanity check. It proves `@parameter` functions can be passed directly as compile-time function parameters.

`function_group_plan_proto.mojo` removes the `ReductionOp` and `Writeback` traits from the composite group design. Each group is parameterized by plain functions: `combine`, `writeback`, `reads_old_dest`, `start`, and `count`.

`function_work_step_proto.mojo` pushes the idea further: a group is just a function-backed dispatch step over a descriptor index. This is the most general shape so far, because the function body owns the operation semantics.

`function_real_burst_proto.mojo` checks the function-step shape against the real `threading.threading_traits.BurstKernel` trait. Run it with `-I .` from the repo root.

`function_capture_step_proto.mojo` checks local `@parameter` bodies. Capturing compile-time values works. Capturing a runtime local in the composite kernel failed LLVM lowering, so runtime phase state should be carried through descriptor/config memory instead of closure capture.

`origin_exact_config_proto.mojo` tests the closest exact-origin config shape. A fully generic version with mutable source/destination origins in the `BurstKernel` payload failed at `JobSet.add(...)`: Mojo treats the stored payload's mutable origin parameters as writable references that may alias the mutable job set. Changing `JobSet.add` to consume the payload did not help, and even a raw-address payload still failed while the kernel type retained generic mutable origin parameters. Constraining those origins to concrete `MutExternalOrigin` compiles, which is useful but too narrow as a general API.

`origin_readonly_config_proto.mojo` tests the practical compromise: keep exact origins for read-only source pointers by converting mutable input pointers to `ImmutOrigin(...)`, and erase only write-side pointers plus escaping config addresses. This compiles through `JobSet` because the kernel payload carries only immutable caller origins. That is the pattern now applied to `kernels/reductions.mojo`.

`origin_lifetime_token_proto.mojo` tests carrying immutable exact-origin tokens for write destinations while using raw addresses for the actual write pointer inside `execute()`. This preserves a lifetime tie to stack-backed destinations without putting a mutable origin in the job payload. It also shows configs themselves can be passed to jobs through exact immutable config origins instead of `MutAnyOrigin`, while still using a post-dispatch `_ = cfg.n` keepalive because mailbox copies escape the compiler's lifetime model.

`origin_direct_job_storage_proto.mojo` tests whether exact mutable origins are impossible in job payloads or only impossible across `add(mut self, job)`. Direct assignment into a local job buffer compiles with an exact mutable destination origin, and dispatching that buffer compiles too. The abstraction boundary is the problem: a helper method that both mutates the job set and receives a job carrying mutable origins triggers exclusivity. A fully in-place config with two same-origin mutable pointers still fails before dispatch, so sources need immutable origins or ranks need distinct origins.

`origin_rank_buffer_builder_proto.mojo` tests the same builder issue at the rank-buffer layer. A read buffer whose pointers are `ImmutOrigin(...)` can be populated from stack storage through `insert_next()`. A write buffer with mutable stack-origin pointers needs direct field assignment, not an `insert_next(mut self, ptr)` method, for the same reason `JobSet.add` is painful.

`origin_slot_jobset_proto.mojo` tests a better primitive for exact mutable-origin jobs: `JobSet.reserve()` returns a pointer to the next job slot, and the caller assigns the job into that slot directly. This keeps `JobSet` encapsulation for count/dispatch while avoiding the `add(mut self, job)` aliasing boundary.

`reduction_extraction_primitives_proto.mojo` tests low-risk extraction helpers suggested by the current `kernels/reductions.mojo`: `frozen_ptr()` for immutable stack config pointers, `copy_elements()` for typed element-count copies, and `reduce_sources_to()` for the shared vectorized "sum TP sources then store/cast to destination dtype" loop used by both reduce-store and reduce-to-scratch.

Attempted rank-phase dispatcher extraction with captured/local job builders and top-level function builders was not promising. Captured builders run into closure capture convention problems around writable pointers, top-level function builders hit function-literal type mismatches in generic signatures, and generic `K(config, rank, start, end)` construction is not accepted. The safer abstraction line appears to be lower-level primitives plus small explicit phase loops.
