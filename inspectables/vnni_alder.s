	.att_syntax
	.file	"vnni_inspect.mojo"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI0_0:
	.byte	128
	.byte	128
	.byte	128
	.byte	128
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_1:
	.zero	4,128
	.text
	.globl	main
	.prefalign	4, .Lfunc_end0, nop
	.type	main,@function
main:
.Lmain$local:
	.type	.Lmain$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movl	%edi, %ebp
	callq	KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice@PLT
	testq	%rax, %rax
	jne	.LBB0_2
	leaq	static_string_a61c3395ab9379d9(%rip), %rdi
	movq	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_0"@GOTPCREL(%rip), %rdx
	movq	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_1"@GOTPCREL(%rip), %rcx
	movl	$7, %esi
	callq	KGEN_CompilerRT_GetOrCreateGlobal@PLT
.LBB0_2:
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$1, (%rsp)
	movq	$1, 8(%rsp)
	movq	$4, 16(%rsp)
	vmovd	1, %xmm0
	vpbroadcastd	.LCPI0_1(%rip), %xmm1
	vpxor	%xmm1, %xmm0, %xmm0
	vpbroadcastd	%xmm0, %ymm2
	vpxor	%xmm0, %xmm0, %xmm0
	{vex}	vpdpbusd	1, %ymm2, %ymm0
	vmovd	5, %xmm2
	vpxor	%xmm1, %xmm2, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	{vex}	vpdpbusd	33, %ymm2, %ymm0
	vmovd	9, %xmm2
	vpxor	%xmm1, %xmm2, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	{vex}	vpdpbusd	65, %ymm2, %ymm0
	vmovd	13, %xmm2
	vpxor	%xmm1, %xmm2, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	{vex}	vpdpbusd	97, %ymm2, %ymm0
	vmovd	17, %xmm2
	vpxor	%xmm1, %xmm2, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	{vex}	vpdpbusd	129, %ymm2, %ymm0
	vmovd	21, %xmm2
	vpxor	%xmm1, %xmm2, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	{vex}	vpdpbusd	161, %ymm2, %ymm0
	vmovd	25, %xmm2
	vpxor	%xmm1, %xmm2, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	{vex}	vpdpbusd	193, %ymm2, %ymm0
	vmovd	29, %xmm2
	vpxor	%xmm1, %xmm2, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	225, %ymm1, %ymm0
	vmovdqu	%ymm0, 4
	movq	%rsp, %rax
	#APP
	#NO_APP
	leaq	8(%rsp), %rax
	#APP
	#NO_APP
	leaq	16(%rsp), %rax
	#APP
	#NO_APP
	vzeroupper
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.size	.Lmain$local, .Lfunc_end0-main
	.cfi_endproc

	.prefalign	4, .Lfunc_end1, nop
	.type	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_0",@function
"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_0":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice@PLT
.Lfunc_end1:
	.size	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_0", .Lfunc_end1-"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_0"
	.cfi_endproc

	.prefalign	4, .Lfunc_end2, nop
	.type	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_1",@function
"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_1":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice@PLT
.Lfunc_end2:
	.size	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_1", .Lfunc_end2-"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"vnni_inspect::main()\"_closure_1"
	.cfi_endproc

	.type	static_string_a61c3395ab9379d9,@object
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
static_string_a61c3395ab9379d9:
	.asciz	"Runtime"
	.size	static_string_a61c3395ab9379d9, 8

	.section	".note.GNU-stack","",@progbits
