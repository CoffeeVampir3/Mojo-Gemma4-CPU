	.att_syntax
	.file	"moe_act_inspect.mojo"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_0:
	.long	0x80000000
.LCPI0_1:
	.long	0xc2ae0000
.LCPI0_2:
	.long	0x3fb8aa3b
.LCPI0_3:
	.long	0x3f317218
.LCPI0_4:
	.long	0x3e2c2268
.LCPI0_5:
	.long	0x3eff3b64
.LCPI0_6:
	.long	0x3f7ff972
.LCPI0_7:
	.long	0x3f800000
.LCPI0_8:
	.long	4294967170
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
	movq	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_0"@GOTPCREL(%rip), %rdx
	movq	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_1"@GOTPCREL(%rip), %rcx
	movl	$7, %esi
	callq	KGEN_CompilerRT_GetOrCreateGlobal@PLT
.LBB0_2:
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$4, 8(%rsp)
	movq	$4, 16(%rsp)
	vmovups	4, %zmm10
	vbroadcastss	.LCPI0_0(%rip), %zmm1
	vxorps	%zmm1, %zmm10, %zmm0
	vbroadcastss	.LCPI0_1(%rip), %zmm2
	vmaxps	%zmm2, %zmm0, %zmm0
	vbroadcastss	.LCPI0_2(%rip), %zmm4
	vmulps	%zmm4, %zmm0, %zmm3
	vrndscaleps	$12, %zmm3, %zmm3
	vcvttps2dq	%zmm3, %zmm9
	vcvtdq2ps	%zmm9, %zmm8
	vbroadcastss	.LCPI0_3(%rip), %zmm5
	vfnmadd213ps	%zmm0, %zmm5, %zmm8
	vbroadcastss	.LCPI0_4(%rip), %zmm0
	vbroadcastss	.LCPI0_5(%rip), %zmm6
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm8, %zmm11
	vbroadcastss	.LCPI0_6(%rip), %zmm7
	vfmadd213ps	%zmm7, %zmm8, %zmm11
	vbroadcastss	.LCPI0_7(%rip), %zmm3
	vfmadd213ps	%zmm3, %zmm8, %zmm11
	vpbroadcastd	.LCPI0_8(%rip), %zmm8
	vpmaxsd	%zmm8, %zmm9, %zmm9
	vpslld	$23, %zmm9, %zmm12
	vpbroadcastd	.LCPI0_7(%rip), %zmm9
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 4
	vmovups	68, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 68
	vmovups	132, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 132
	vmovups	196, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 196
	vmovups	4, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 4
	vmovups	68, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 68
	vmovups	132, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 132
	vmovups	196, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 196
	vmovups	260, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 260
	vmovups	324, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 324
	vmovups	388, %zmm10
	vxorps	%zmm1, %zmm10, %zmm11
	vmaxps	%zmm2, %zmm11, %zmm11
	vmulps	%zmm4, %zmm11, %zmm12
	vrndscaleps	$12, %zmm12, %zmm12
	vcvttps2dq	%zmm12, %zmm12
	vcvtdq2ps	%zmm12, %zmm13
	vfnmadd213ps	%zmm11, %zmm5, %zmm13
	vmovaps	%zmm0, %zmm11
	vfmadd213ps	%zmm6, %zmm13, %zmm11
	vfmadd213ps	%zmm7, %zmm13, %zmm11
	vfmadd213ps	%zmm3, %zmm13, %zmm11
	vpmaxsd	%zmm8, %zmm12, %zmm12
	vpslld	$23, %zmm12, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm12
	vfmadd213ps	%zmm3, %zmm11, %zmm12
	vdivps	%zmm12, %zmm10, %zmm11
	vmulps	%zmm11, %zmm10, %zmm10
	vmovups	%zmm10, 388
	vmovups	452, %zmm10
	vxorps	%zmm1, %zmm10, %zmm1
	vmaxps	%zmm2, %zmm1, %zmm1
	vmulps	%zmm4, %zmm1, %zmm2
	vrndscaleps	$12, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vcvtdq2ps	%zmm2, %zmm4
	vfnmadd213ps	%zmm1, %zmm5, %zmm4
	vfmadd213ps	%zmm6, %zmm4, %zmm0
	vfmadd213ps	%zmm7, %zmm4, %zmm0
	vfmadd213ps	%zmm3, %zmm4, %zmm0
	vpmaxsd	%zmm8, %zmm2, %zmm1
	vpslld	$23, %zmm1, %zmm1
	vpaddd	%zmm9, %zmm1, %zmm1
	vfmadd213ps	%zmm3, %zmm0, %zmm1
	vdivps	%zmm1, %zmm10, %zmm0
	vmulps	%zmm0, %zmm10, %zmm0
	vmovups	%zmm0, 452
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
	.type	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_0",@function
"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_0":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice@PLT
.Lfunc_end1:
	.size	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_0", .Lfunc_end1-"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_0"
	.cfi_endproc

	.prefalign	4, .Lfunc_end2, nop
	.type	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_1",@function
"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_1":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice@PLT
.Lfunc_end2:
	.size	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_1", .Lfunc_end2-"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"moe_act_inspect::main()\"_closure_1"
	.cfi_endproc

	.type	static_string_a61c3395ab9379d9,@object
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
static_string_a61c3395ab9379d9:
	.asciz	"Runtime"
	.size	static_string_a61c3395ab9379d9, 8

	.section	".note.GNU-stack","",@progbits
