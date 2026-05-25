	.att_syntax
	.file	"i8_quantize_inspect.mojo"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_0:
	.long	0xc3000000
.LCPI0_1:
	.long	0x42fe0000
.LCPI0_2:
	.long	0x7fffffff
.LCPI0_3:
	.long	0x2edbe6ff
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
	subq	$32, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movl	%edi, %ebp
	callq	KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice@PLT
	testq	%rax, %rax
	jne	.LBB0_2
	leaq	static_string_a61c3395ab9379d9(%rip), %rdi
	movq	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_0"@GOTPCREL(%rip), %rdx
	movq	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_1"@GOTPCREL(%rip), %rcx
	movl	$7, %esi
	callq	KGEN_CompilerRT_GetOrCreateGlobal@PLT
.LBB0_2:
	movl	$452, %r14d
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$4, 16(%rsp)
	movq	$1, 24(%rsp)
	xorl	%eax, %eax
	vbroadcastss	.LCPI0_0(%rip), %zmm0
	vbroadcastss	.LCPI0_1(%rip), %zmm1
	.p2align	4
.LBB0_3:
	vrndscaleps	$12, -448(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -451(%r14,%rax)
	vrndscaleps	$12, -384(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -435(%r14,%rax)
	vrndscaleps	$12, -320(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -419(%r14,%rax)
	vrndscaleps	$12, -256(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -403(%r14,%rax)
	vrndscaleps	$12, -192(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -387(%r14,%rax)
	vrndscaleps	$12, -128(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -371(%r14,%rax)
	vrndscaleps	$12, -64(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -355(%r14,%rax)
	vrndscaleps	$12, (%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -339(%r14,%rax)
	subq	$-128, %rax
	cmpq	$4096, %rax
	jne	.LBB0_3
	xorl	%eax, %eax
	.p2align	4
.LBB0_5:
	vrndscaleps	$12, -448(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -451(%r14,%rax)
	vrndscaleps	$12, -384(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -435(%r14,%rax)
	vrndscaleps	$12, -320(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -419(%r14,%rax)
	vrndscaleps	$12, -256(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -403(%r14,%rax)
	vrndscaleps	$12, -192(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -387(%r14,%rax)
	vrndscaleps	$12, -128(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -371(%r14,%rax)
	vrndscaleps	$12, -64(%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -355(%r14,%rax)
	vrndscaleps	$12, (%r14,%rax,4), %zmm2
	vmaxps	%zmm0, %zmm2, %zmm2
	vminps	%zmm1, %zmm2, %zmm2
	vcvttps2dq	%zmm2, %zmm2
	vpmovdb	%zmm2, -339(%r14,%rax)
	subq	$-128, %rax
	cmpq	$1024, %rax
	jne	.LBB0_5
	vxorps	%xmm7, %xmm7, %xmm7
	xorl	%eax, %eax
	vbroadcastss	.LCPI0_2(%rip), %zmm10
	vxorps	%xmm9, %xmm9, %xmm9
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm3, %xmm3, %xmm3
	vpxor	%xmm2, %xmm2, %xmm2
	.p2align	4
.LBB0_7:
	vandps	-448(%r14,%rax), %zmm10, %zmm11
	vcmpunordps	%zmm7, %zmm7, %k1
	vmaxps	%zmm7, %zmm11, %zmm7
	vandps	-384(%r14,%rax), %zmm10, %zmm12
	vmovaps	%zmm11, %zmm7 {%k1}
	vcmpunordps	%zmm9, %zmm9, %k1
	vmaxps	%zmm9, %zmm12, %zmm9
	vmovaps	%zmm12, %zmm9 {%k1}
	vandps	-320(%r14,%rax), %zmm10, %zmm11
	vcmpunordps	%zmm8, %zmm8, %k1
	vmaxps	%zmm8, %zmm11, %zmm8
	vmovaps	%zmm11, %zmm8 {%k1}
	vandps	-256(%r14,%rax), %zmm10, %zmm11
	vcmpunordps	%zmm6, %zmm6, %k1
	vmaxps	%zmm6, %zmm11, %zmm6
	vmovaps	%zmm11, %zmm6 {%k1}
	vandps	-192(%r14,%rax), %zmm10, %zmm11
	vcmpunordps	%zmm5, %zmm5, %k1
	vmaxps	%zmm5, %zmm11, %zmm5
	vmovaps	%zmm11, %zmm5 {%k1}
	vandps	-128(%r14,%rax), %zmm10, %zmm11
	vcmpunordps	%zmm4, %zmm4, %k1
	vmaxps	%zmm4, %zmm11, %zmm4
	vmovaps	%zmm11, %zmm4 {%k1}
	vandps	-64(%r14,%rax), %zmm10, %zmm11
	vcmpunordps	%zmm3, %zmm3, %k1
	vmaxps	%zmm3, %zmm11, %zmm3
	vandps	(%r14,%rax), %zmm10, %zmm12
	vmovaps	%zmm11, %zmm3 {%k1}
	vcmpunordps	%zmm2, %zmm2, %k1
	vmaxps	%zmm2, %zmm12, %zmm2
	vmovaps	%zmm12, %zmm2 {%k1}
	addq	$512, %rax
	cmpq	$16384, %rax
	jne	.LBB0_7
	vmaxps	%zmm7, %zmm9, %zmm10
	vcmpunordps	%zmm7, %zmm7, %k1
	vmovaps	%zmm9, %zmm10 {%k1}
	vmaxps	%zmm10, %zmm8, %zmm7
	vcmpunordps	%zmm10, %zmm10, %k1
	vmovaps	%zmm8, %zmm7 {%k1}
	vmaxps	%zmm7, %zmm6, %zmm8
	vcmpunordps	%zmm7, %zmm7, %k1
	vmovaps	%zmm6, %zmm8 {%k1}
	vmaxps	%zmm8, %zmm5, %zmm6
	vcmpunordps	%zmm8, %zmm8, %k1
	vmovaps	%zmm5, %zmm6 {%k1}
	vmaxps	%zmm6, %zmm4, %zmm5
	vcmpunordps	%zmm6, %zmm6, %k1
	vmovaps	%zmm4, %zmm5 {%k1}
	vmaxps	%zmm5, %zmm3, %zmm4
	vcmpunordps	%zmm5, %zmm5, %k1
	vmovaps	%zmm3, %zmm4 {%k1}
	vmaxps	%zmm4, %zmm2, %zmm3
	vcmpunordps	%zmm4, %zmm4, %k1
	vmovaps	%zmm2, %zmm3 {%k1}
	vextractf64x4	$1, %zmm3, %ymm2
	vmaxps	%ymm3, %ymm2, %ymm4
	vcmpunordps	%ymm3, %ymm3, %k1
	vmovaps	%ymm2, %ymm4 {%k1}
	vextractf128	$1, %ymm4, %xmm2
	vmaxps	%xmm4, %xmm2, %xmm3
	vcmpunordps	%xmm4, %xmm4, %k1
	vmovaps	%xmm2, %xmm3 {%k1}
	vshufpd	$1, %xmm3, %xmm3, %xmm2
	vmaxps	%xmm3, %xmm2, %xmm4
	vcmpunordps	%xmm3, %xmm3, %k1
	vmovaps	%xmm2, %xmm4 {%k1}
	vmovshdup	%xmm4, %xmm2
	vmaxss	%xmm4, %xmm2, %xmm3
	vcmpunordss	%xmm4, %xmm4, %k1
	vmovss	%xmm2, %xmm3, %xmm3 {%k1}
	vmovss	.LCPI0_3(%rip), %xmm2
	vmaxss	%xmm3, %xmm2, %xmm2
	vmovss	.LCPI0_1(%rip), %xmm3
	vdivss	%xmm2, %xmm3, %xmm3
	vbroadcastss	%xmm3, %zmm3
	xorl	%eax, %eax
	.p2align	4
.LBB0_9:
	vmulps	-448(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -451(%r14,%rax)
	vmulps	-384(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -435(%r14,%rax)
	vmulps	-320(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -419(%r14,%rax)
	vmulps	-256(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -403(%r14,%rax)
	vmulps	-192(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -387(%r14,%rax)
	vmulps	-128(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -371(%r14,%rax)
	vmulps	-64(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -355(%r14,%rax)
	vmulps	(%r14,%rax,4), %zmm3, %zmm4
	vrndscaleps	$12, %zmm4, %zmm4
	vmaxps	%zmm0, %zmm4, %zmm4
	vminps	%zmm1, %zmm4, %zmm4
	vcvttps2dq	%zmm4, %zmm4
	vpmovdb	%zmm4, -339(%r14,%rax)
	subq	$-128, %rax
	cmpq	$4096, %rax
	jne	.LBB0_9
	vmovss	%xmm2, 12(%rsp)
	leaq	12(%rsp), %rax
	#APP
	#NO_APP
	leaq	16(%rsp), %rax
	#APP
	#NO_APP
	leaq	24(%rsp), %rax
	#APP
	#NO_APP
	vzeroupper
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$32, %rsp
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
	.type	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_0",@function
"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_0":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice@PLT
.Lfunc_end1:
	.size	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_0", .Lfunc_end1-"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_0"
	.cfi_endproc

	.prefalign	4, .Lfunc_end2, nop
	.type	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_1",@function
"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_1":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice@PLT
.Lfunc_end2:
	.size	"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_1", .Lfunc_end2-"std::builtin::_startup::__wrap_and_execute_main[def() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"i8_quantize_inspect::main()\"_closure_1"
	.cfi_endproc

	.type	static_string_a61c3395ab9379d9,@object
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
static_string_a61c3395ab9379d9:
	.asciz	"Runtime"
	.size	static_string_a61c3395ab9379d9, 8

	.section	".note.GNU-stack","",@progbits
