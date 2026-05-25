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
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$80, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
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
	movl	$228, %r14d
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$4, (%rsp)
	movq	$4, 8(%rsp)
	vmovups	4, %ymm10
	vbroadcastss	.LCPI0_0(%rip), %ymm0
	vxorps	%ymm0, %ymm10, %ymm2
	vbroadcastss	.LCPI0_1(%rip), %ymm14
	vmaxps	%ymm14, %ymm2, %ymm4
	vbroadcastss	.LCPI0_2(%rip), %ymm2
	vmulps	%ymm2, %ymm4, %ymm3
	vroundps	$12, %ymm3, %ymm3
	vcvttps2dq	%ymm3, %ymm9
	vcvtdq2ps	%ymm9, %ymm8
	vbroadcastss	.LCPI0_3(%rip), %ymm3
	vfnmadd213ps	%ymm4, %ymm3, %ymm8
	vbroadcastss	.LCPI0_4(%rip), %ymm1
	vbroadcastss	.LCPI0_5(%rip), %ymm4
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm8, %ymm11
	vbroadcastss	.LCPI0_6(%rip), %ymm5
	vfmadd213ps	%ymm5, %ymm8, %ymm11
	vbroadcastss	.LCPI0_7(%rip), %ymm6
	vfmadd213ps	%ymm6, %ymm8, %ymm11
	vpbroadcastd	.LCPI0_8(%rip), %ymm7
	vpmaxsd	%ymm7, %ymm9, %ymm9
	vpslld	$23, %ymm9, %ymm12
	vpbroadcastd	.LCPI0_7(%rip), %ymm9
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 4
	vmovups	36, %ymm10
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 36
	vmovups	68, %ymm10
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 68
	vmovups	100, %ymm10
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 100
	vmovups	132, %ymm10
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 132
	vmovups	164, %ymm10
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 164
	vmovups	196, %ymm10
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 196
	vmovups	228, %ymm10
	vmovups	%ymm0, 48(%rsp)
	vxorps	%ymm0, %ymm10, %ymm11
	vmaxps	%ymm14, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm11, %ymm3, %ymm13
	vmovaps	%ymm1, %ymm11
	vfmadd213ps	%ymm4, %ymm13, %ymm11
	vfmadd213ps	%ymm5, %ymm13, %ymm11
	vfmadd213ps	%ymm6, %ymm13, %ymm11
	vmovdqu	%ymm7, 16(%rsp)
	vpmaxsd	%ymm7, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm6, %ymm11, %ymm12
	vdivps	%ymm12, %ymm10, %ymm11
	vmulps	%ymm11, %ymm10, %ymm10
	vmovups	%ymm10, 228
	xorl	%eax, %eax
	vmovups	48(%rsp), %ymm7
	vbroadcastss	.LCPI0_1(%rip), %ymm2
	vbroadcastss	.LCPI0_2(%rip), %ymm3
	vbroadcastss	.LCPI0_3(%rip), %ymm5
	vbroadcastss	.LCPI0_5(%rip), %ymm1
	vbroadcastss	.LCPI0_6(%rip), %ymm0
	vbroadcastss	.LCPI0_7(%rip), %ymm8
	.p2align	4
.LBB0_3:
	vmovups	-224(%r14,%rax), %ymm13
	vmovups	-192(%r14,%rax), %ymm12
	vxorps	%ymm7, %ymm13, %ymm10
	vmaxps	%ymm2, %ymm10, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vroundps	$12, %ymm11, %ymm14
	vmovups	-160(%r14,%rax), %ymm11
	vcvttps2dq	%ymm14, %ymm14
	vcvtdq2ps	%ymm14, %ymm15
	vfnmadd213ps	%ymm10, %ymm5, %ymm15
	vbroadcastss	.LCPI0_4(%rip), %ymm6
	vmovaps	%ymm6, %ymm4
	vfmadd213ps	%ymm1, %ymm15, %ymm4
	vfmadd213ps	%ymm0, %ymm15, %ymm4
	vfmadd213ps	%ymm8, %ymm15, %ymm4
	vmovups	-128(%r14,%rax), %ymm10
	vmovdqu	16(%rsp), %ymm15
	vpmaxsd	%ymm15, %ymm14, %ymm14
	vpslld	$23, %ymm14, %ymm14
	vpaddd	%ymm9, %ymm14, %ymm14
	vfmadd213ps	%ymm8, %ymm4, %ymm14
	vdivps	%ymm14, %ymm13, %ymm4
	vmulps	%ymm4, %ymm13, %ymm4
	vmovups	%ymm4, -224(%r14,%rax)
	vxorps	%ymm7, %ymm12, %ymm4
	vmaxps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm3, %ymm4, %ymm13
	vroundps	$12, %ymm13, %ymm13
	vcvttps2dq	%ymm13, %ymm13
	vcvtdq2ps	%ymm13, %ymm14
	vfnmadd213ps	%ymm4, %ymm5, %ymm14
	vmovaps	%ymm6, %ymm4
	vfmadd213ps	%ymm1, %ymm14, %ymm4
	vfmadd213ps	%ymm0, %ymm14, %ymm4
	vfmadd213ps	%ymm8, %ymm14, %ymm4
	vpmaxsd	%ymm15, %ymm13, %ymm13
	vpslld	$23, %ymm13, %ymm13
	vpaddd	%ymm9, %ymm13, %ymm13
	vfmadd213ps	%ymm8, %ymm4, %ymm13
	vdivps	%ymm13, %ymm12, %ymm4
	vmulps	%ymm4, %ymm12, %ymm4
	vmovups	%ymm4, -192(%r14,%rax)
	vxorps	%ymm7, %ymm11, %ymm4
	vmaxps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm3, %ymm4, %ymm12
	vroundps	$12, %ymm12, %ymm12
	vcvttps2dq	%ymm12, %ymm12
	vcvtdq2ps	%ymm12, %ymm13
	vfnmadd213ps	%ymm4, %ymm5, %ymm13
	vmovaps	%ymm6, %ymm4
	vfmadd213ps	%ymm1, %ymm13, %ymm4
	vfmadd213ps	%ymm0, %ymm13, %ymm4
	vfmadd213ps	%ymm8, %ymm13, %ymm4
	vpmaxsd	%ymm15, %ymm12, %ymm12
	vpslld	$23, %ymm12, %ymm12
	vpaddd	%ymm9, %ymm12, %ymm12
	vfmadd213ps	%ymm8, %ymm4, %ymm12
	vdivps	%ymm12, %ymm11, %ymm4
	vmulps	%ymm4, %ymm11, %ymm4
	vmovups	%ymm4, -160(%r14,%rax)
	vxorps	%ymm7, %ymm10, %ymm4
	vmaxps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm3, %ymm4, %ymm11
	vroundps	$12, %ymm11, %ymm11
	vcvttps2dq	%ymm11, %ymm11
	vcvtdq2ps	%ymm11, %ymm12
	vfnmadd213ps	%ymm4, %ymm5, %ymm12
	vmovaps	%ymm6, %ymm4
	vfmadd213ps	%ymm1, %ymm12, %ymm4
	vfmadd213ps	%ymm0, %ymm12, %ymm4
	vfmadd213ps	%ymm8, %ymm12, %ymm4
	vpmaxsd	%ymm15, %ymm11, %ymm11
	vpslld	$23, %ymm11, %ymm11
	vpaddd	%ymm9, %ymm11, %ymm11
	vfmadd213ps	%ymm8, %ymm4, %ymm11
	vdivps	%ymm11, %ymm10, %ymm4
	vmulps	%ymm4, %ymm10, %ymm4
	vmovups	%ymm4, -128(%r14,%rax)
	vmovups	-96(%r14,%rax), %ymm4
	vxorps	%ymm7, %ymm4, %ymm10
	vmaxps	%ymm2, %ymm10, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vroundps	$12, %ymm11, %ymm11
	vcvttps2dq	%ymm11, %ymm11
	vcvtdq2ps	%ymm11, %ymm12
	vfnmadd213ps	%ymm10, %ymm5, %ymm12
	vmovaps	%ymm6, %ymm10
	vfmadd213ps	%ymm1, %ymm12, %ymm10
	vfmadd213ps	%ymm0, %ymm12, %ymm10
	vfmadd213ps	%ymm8, %ymm12, %ymm10
	vpmaxsd	%ymm15, %ymm11, %ymm11
	vpslld	$23, %ymm11, %ymm11
	vpaddd	%ymm9, %ymm11, %ymm11
	vfmadd213ps	%ymm8, %ymm10, %ymm11
	vdivps	%ymm11, %ymm4, %ymm10
	vmulps	%ymm4, %ymm10, %ymm4
	vmovups	%ymm4, -96(%r14,%rax)
	vmovups	-64(%r14,%rax), %ymm4
	vxorps	%ymm7, %ymm4, %ymm10
	vmaxps	%ymm2, %ymm10, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vroundps	$12, %ymm11, %ymm11
	vcvttps2dq	%ymm11, %ymm11
	vcvtdq2ps	%ymm11, %ymm12
	vfnmadd213ps	%ymm10, %ymm5, %ymm12
	vmovaps	%ymm6, %ymm10
	vfmadd213ps	%ymm1, %ymm12, %ymm10
	vfmadd213ps	%ymm0, %ymm12, %ymm10
	vfmadd213ps	%ymm8, %ymm12, %ymm10
	vpmaxsd	%ymm15, %ymm11, %ymm11
	vpslld	$23, %ymm11, %ymm11
	vpaddd	%ymm9, %ymm11, %ymm11
	vfmadd213ps	%ymm8, %ymm10, %ymm11
	vdivps	%ymm11, %ymm4, %ymm10
	vmulps	%ymm4, %ymm10, %ymm4
	vmovups	%ymm4, -64(%r14,%rax)
	vmovups	-32(%r14,%rax), %ymm4
	vxorps	%ymm7, %ymm4, %ymm10
	vmaxps	%ymm2, %ymm10, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vroundps	$12, %ymm11, %ymm11
	vcvttps2dq	%ymm11, %ymm11
	vcvtdq2ps	%ymm11, %ymm12
	vfnmadd213ps	%ymm10, %ymm5, %ymm12
	vmovaps	%ymm6, %ymm10
	vfmadd213ps	%ymm1, %ymm12, %ymm10
	vfmadd213ps	%ymm0, %ymm12, %ymm10
	vfmadd213ps	%ymm8, %ymm12, %ymm10
	vpmaxsd	%ymm15, %ymm11, %ymm11
	vpslld	$23, %ymm11, %ymm11
	vpaddd	%ymm9, %ymm11, %ymm11
	vfmadd213ps	%ymm8, %ymm10, %ymm11
	vdivps	%ymm11, %ymm4, %ymm10
	vmulps	%ymm4, %ymm10, %ymm4
	vmovups	%ymm4, -32(%r14,%rax)
	vmovups	(%r14,%rax), %ymm4
	vxorps	%ymm7, %ymm4, %ymm10
	vmaxps	%ymm2, %ymm10, %ymm10
	vmulps	%ymm3, %ymm10, %ymm11
	vroundps	$12, %ymm11, %ymm11
	vcvttps2dq	%ymm11, %ymm11
	vcvtdq2ps	%ymm11, %ymm12
	vfnmadd213ps	%ymm10, %ymm5, %ymm12
	vfmadd213ps	%ymm1, %ymm12, %ymm6
	vfmadd213ps	%ymm0, %ymm12, %ymm6
	vfmadd213ps	%ymm8, %ymm12, %ymm6
	vpmaxsd	%ymm15, %ymm11, %ymm11
	vpslld	$23, %ymm11, %ymm11
	vpaddd	%ymm9, %ymm11, %ymm11
	vfmadd213ps	%ymm8, %ymm6, %ymm11
	vdivps	%ymm11, %ymm4, %ymm10
	vmulps	%ymm4, %ymm10, %ymm4
	vmovups	%ymm4, (%r14,%rax)
	addq	$256, %rax
	cmpq	$512, %rax
	jne	.LBB0_3
	movq	%rsp, %rax
	#APP
	#NO_APP
	leaq	8(%rsp), %rax
	#APP
	#NO_APP
	vzeroupper
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$80, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
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
