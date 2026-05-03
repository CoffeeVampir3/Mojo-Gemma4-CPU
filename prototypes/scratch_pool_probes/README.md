# ScratchPool / LinearBorrowPool Design Probes

These probes exhaustively explore whether the `ScratchPool`/`ScratchLease` abstraction in `modeling/linear_borrow_pool.mojo` can be meaningfully improved given Mojo's aliasing rules.

## Current design (linear_borrow_pool.mojo)

- `ScratchPool`: LIFO bump allocator. Tracks `offset` into a pre-allocated region.
- `ScratchLease`: `@explicit_destroy` linear token. Must be consumed via `^.release()`. Holds `offset`, `byte_size`, and a `MutAnyOrigin` pointer back to the pool's offset field.
- `as_ptr[T]()` and `view[E, S]()` derive their origin from the lease reference — use-after-release is caught at compile time.
- LIFO ordering is enforced at runtime only (abort on violation).
- Caller must pass `scratch_base` on every `as_ptr`/`view` call — the lease stores only the offset, not the absolute address.

## What the probes tested

### probe_01: Scoped lease with baked-in base address
**Result: works, strict improvement.** Storing `addr` (base + offset) instead of bare `offset` eliminates the `scratch_base` argument at every call site. Zero cost — the addition happens once at borrow time.

### probe_02: Nested LIFO with method-based borrow
**Result: works.** `pool.borrow[T, count]()` as a method works cleanly. The key finding: a *free function* `borrow_from[origin](mut pool)` cannot infer the MutOrigin parameter from a `mut` argument — Mojo can't propagate that. Must use a method or accept explicit origin specification.

### probe_03: Watermark pool (no per-lease release)
**Result: works, but loses origin-safety.** `save()/restore()` is simpler for phase-structured code. All buffers within a phase are batch-freed. However, `RegionPtr` returns `MutAnyOrigin` — there's no lease lifetime to tie pointers to. This means no compile-time use-after-free detection. Appropriate only if the caller already has a strong phase discipline and doesn't need the compiler to enforce it.

### probe_04: Fully static comptime layout
**Result: works, but no isolation.** When the scratch layout is fully known at compile time, you can compute all offsets as `comptime` values and access them via `ptr_at[T, offset]()`. No allocator, no runtime overhead. But all pointers are `MutAnyOrigin` and there's no concept of "this region is no longer valid" — overlapping phases freely alias. Only useful when the layout is static AND you don't need the compiler to enforce non-aliasing.

### probe_05: Origin safety limits with BurstKernel dispatch
**Result: origin-erased kernels work fine.** The `origin_lifetime_token_proto.mojo` pattern (carry an immutable token for lifetime extension, reconstruct the mutable pointer from a raw address inside `execute()`) works with the pool lease pattern. An erased kernel (`WriteKernelErased` with just `dst_addr: Int`) is simpler and compiles through `dispatch()` without aliasing issues. The origin safety lives at the lease level, not the kernel level.

### probe_06: Hybrid watermark + phase lease
**Result: works, interesting middle ground.** `PhaseLease` captures a watermark at creation and restores it on release. Buffers within a phase are allocated via `lease.bump[T, count]()` with origin tied to the lease ref. This gives: (1) batch release of a whole phase, (2) origin-safety on individual buffer pointers within the phase, (3) persistent buffers outside the phase survive. The API is `begin_phase(pool) -> PhaseLease`, `lease.bump[T, N]() -> ptr`, `lease^.release()`.

### probe_07: Compile-time LIFO via exact pool origin
**Result: compiles, but doesn't actually enforce LIFO order.** `ExactLease[pool_origin: MutOrigin]` with `UnsafePointer[StackPool, pool_origin]` compiles and works. However, Mojo's lifetime checker doesn't enforce destruction order of sibling variables — you can still release in the wrong order and get a runtime abort. The exact origin buys nothing over `MutAnyOrigin` for LIFO enforcement. It would only matter if the pool itself were destroyed while leases existed, which is already prevented by the `MutAnyOrigin` pointer keeping the pool live.

### probe_08: Exact vs erased origin with dispatch
**Result: both work identically with BurstKernel.** The exact-origin lease and the erased-origin lease both compile through `dispatch()` when the kernel carries only addresses (not origin-bearing pointers to the pool). Since kernels need `Int` addresses anyway (to cross the mailbox boundary), the exact pool origin in the lease buys nothing for dispatch compatibility.

### probe_09: Real usage pattern edge cases
**Result: all patterns work with the erased (current) design.** Phase reuse, persistent + temp interleaving, kernel dispatch from lease buffers — all work cleanly with `MutAnyOrigin` on the pool offset pointer. The origin safety comes from `as_ptr[o: MutOrigin](ref [o] self)` tying pointer origin to the lease reference, not from the pool-writeback pointer.

## Conclusions

1. **Baking the base address into the lease is a strict improvement.** Store `addr = base + offset` instead of bare `offset`. Eliminates a parameter from every `as_ptr`/`view` call. The current design requires callers to pass `scratch_base` repeatedly, which is error-prone (wrong base = silent corruption) and verbose.

2. **Exact pool origins on the lease don't help.** Mojo doesn't enforce destruction order of sibling variables, so compile-time LIFO enforcement is impossible with the current language. The runtime abort in `release()` is the correct check. The `MutAnyOrigin` pointer to the pool's offset field is the right trade-off.

3. **The hybrid watermark+lease (probe 06) is interesting but doesn't solve a real problem the current design has.** It's better for code that has clear phase boundaries and allocates many temp buffers per phase. But the current LIFO lease already handles this — you just release in reverse. The watermark saves counting releases but adds a new concept.

4. **The fully static layout (probe 04) is appealing for models with known scratch layout.** If every layer's scratch usage is comptime-known (as it is for Gemma4 via `calculate_peak_scratch`), you could compute all offsets at compile time and skip the allocator entirely. But you lose the compiler's origin-based use-after-free protection, which is the main value of the lease abstraction.

5. **The current design is sound.** The key insight is that origin safety comes from `as_ptr[o](ref [o] self)` — the pointer's origin is the lease reference's origin. This is correct and doesn't need changing. The `@explicit_destroy` + manual `^.release()` is ugly but necessary: Mojo's ASAP destruction would release in declaration order (FIFO), not LIFO, so automatic destruction would be wrong.

## Recommended changes

The only change worth making to the current `linear_borrow_pool.mojo`:

- **Store absolute address instead of offset.** Change `ScratchLease.offset` to `ScratchLease.addr` (= `scratch_base + offset`), bake the base at borrow time. Remove `scratch_base` from `as_ptr` and `view` signatures. This is a pure simplification with no downside — the pool already knows the base, so the lease can too.

Everything else is either equivalent complexity or trades compile-time safety for syntactic convenience.
