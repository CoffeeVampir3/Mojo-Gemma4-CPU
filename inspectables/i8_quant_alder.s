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
	movl	$57, %r14d
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$4, 16(%rsp)
	movq	$1, 24(%rsp)
	xorl	%eax, %eax
	vbroadcastss	.LCPI0_0(%rip), %ymm0
	vbroadcastss	.LCPI0_1(%rip), %ymm1
	.p2align	4
.LBB0_3:
	vroundps	$12, -53(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -56(%r14,%rax)
	vroundps	$12, -21(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -48(%r14,%rax)
	vroundps	$12, 11(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -40(%r14,%rax)
	vroundps	$12, 43(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -32(%r14,%rax)
	vroundps	$12, 75(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -24(%r14,%rax)
	vroundps	$12, 107(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -16(%r14,%rax)
	vroundps	$12, 139(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, -8(%r14,%rax)
	vroundps	$12, 171(%r14,%rax,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, (%r14,%rax)
	addq	$64, %rax
	cmpq	$4096, %rax
	jne	.LBB0_3
	movq	16(%rsp), %rax
	movq	24(%rsp), %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB0_5:
	vroundps	$12, (%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, (%rcx,%rdx)
	vroundps	$12, 32(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 8(%rcx,%rdx)
	vroundps	$12, 64(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 16(%rcx,%rdx)
	vroundps	$12, 96(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 24(%rcx,%rdx)
	vroundps	$12, 128(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 32(%rcx,%rdx)
	vroundps	$12, 160(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 40(%rcx,%rdx)
	vroundps	$12, 192(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 48(%rcx,%rdx)
	vroundps	$12, 224(%rax,%rdx,4), %ymm2
	vmaxps	%ymm0, %ymm2, %ymm2
	vminps	%ymm1, %ymm2, %ymm2
	vcvttps2dq	%ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackssdw	%xmm3, %xmm2, %xmm2
	vpacksswb	%xmm2, %xmm2, %xmm2
	vmovq	%xmm2, 56(%rcx,%rdx)
	addq	$64, %rdx
	cmpq	$1024, %rdx
	jne	.LBB0_5
	vxorps	%xmm6, %xmm6, %xmm6
	xorl	%edx, %edx
	vbroadcastss	.LCPI0_2(%rip), %ymm9
	vxorps	%xmm10, %xmm10, %xmm10
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm4, %xmm4, %xmm4
	vpxor	%xmm3, %xmm3, %xmm3
	vpxor	%xmm2, %xmm2, %xmm2
	.p2align	4
.LBB0_7:
	vandps	(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm6, %ymm11, %ymm12
	vcmpunordps	%ymm6, %ymm6, %ymm6
	vblendvps	%ymm6, %ymm11, %ymm12, %ymm6
	vandps	32(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm10, %ymm11, %ymm12
	vcmpunordps	%ymm10, %ymm10, %ymm10
	vblendvps	%ymm10, %ymm11, %ymm12, %ymm10
	vandps	64(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm8, %ymm11, %ymm12
	vcmpunordps	%ymm8, %ymm8, %ymm8
	vblendvps	%ymm8, %ymm11, %ymm12, %ymm8
	vandps	96(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm7, %ymm11, %ymm12
	vcmpunordps	%ymm7, %ymm7, %ymm7
	vblendvps	%ymm7, %ymm11, %ymm12, %ymm7
	vandps	128(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm5, %ymm11, %ymm12
	vcmpunordps	%ymm5, %ymm5, %ymm5
	vblendvps	%ymm5, %ymm11, %ymm12, %ymm5
	vandps	160(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm4, %ymm11, %ymm12
	vcmpunordps	%ymm4, %ymm4, %ymm4
	vblendvps	%ymm4, %ymm11, %ymm12, %ymm4
	vandps	192(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm3, %ymm11, %ymm12
	vcmpunordps	%ymm3, %ymm3, %ymm3
	vblendvps	%ymm3, %ymm11, %ymm12, %ymm3
	vandps	224(%rax,%rdx), %ymm9, %ymm11
	vmaxps	%ymm2, %ymm11, %ymm12
	vcmpunordps	%ymm2, %ymm2, %ymm2
	vblendvps	%ymm2, %ymm11, %ymm12, %ymm2
	addq	$256, %rdx
	cmpq	$16384, %rdx
	jne	.LBB0_7
	vmaxps	%ymm6, %ymm10, %ymm9
	vcmpunordps	%ymm6, %ymm6, %ymm6
	vblendvps	%ymm6, %ymm10, %ymm9, %ymm6
	vmaxps	%ymm6, %ymm8, %ymm9
	vcmpunordps	%ymm6, %ymm6, %ymm6
	vblendvps	%ymm6, %ymm8, %ymm9, %ymm6
	vmaxps	%ymm6, %ymm7, %ymm8
	vcmpunordps	%ymm6, %ymm6, %ymm6
	vblendvps	%ymm6, %ymm7, %ymm8, %ymm6
	vmaxps	%ymm6, %ymm5, %ymm7
	vcmpunordps	%ymm6, %ymm6, %ymm6
	vblendvps	%ymm6, %ymm5, %ymm7, %ymm5
	vmaxps	%ymm5, %ymm4, %ymm6
	vcmpunordps	%ymm5, %ymm5, %ymm5
	vblendvps	%ymm5, %ymm4, %ymm6, %ymm4
	vmaxps	%ymm4, %ymm3, %ymm5
	vcmpunordps	%ymm4, %ymm4, %ymm4
	vblendvps	%ymm4, %ymm3, %ymm5, %ymm3
	vmaxps	%ymm3, %ymm2, %ymm4
	vcmpunordps	%ymm3, %ymm3, %ymm3
	vblendvps	%ymm3, %ymm2, %ymm4, %ymm2
	vextractf128	$1, %ymm2, %xmm3
	vmaxps	%xmm2, %xmm3, %xmm4
	vcmpunordps	%xmm2, %xmm2, %xmm2
	vblendvps	%xmm2, %xmm3, %xmm4, %xmm2
	vshufpd	$1, %xmm2, %xmm2, %xmm3
	vmaxps	%xmm2, %xmm3, %xmm4
	vcmpunordps	%xmm2, %xmm2, %xmm2
	vblendvps	%xmm2, %xmm3, %xmm4, %xmm2
	vmovshdup	%xmm2, %xmm3
	vmaxss	%xmm2, %xmm3, %xmm4
	vcmpunordss	%xmm2, %xmm2, %xmm2
	vblendvps	%xmm2, %xmm3, %xmm4, %xmm2
	vmovss	.LCPI0_3(%rip), %xmm3
	vmaxss	%xmm2, %xmm3, %xmm2
	vmovss	.LCPI0_1(%rip), %xmm3
	vdivss	%xmm2, %xmm3, %xmm3
	vbroadcastss	%xmm3, %ymm3
	xorl	%edx, %edx
	.p2align	4
.LBB0_9:
	vmulps	(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, (%rcx,%rdx)
	vmulps	32(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 8(%rcx,%rdx)
	vmulps	64(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 16(%rcx,%rdx)
	vmulps	96(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 24(%rcx,%rdx)
	vmulps	128(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 32(%rcx,%rdx)
	vmulps	160(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 40(%rcx,%rdx)
	vmulps	192(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 48(%rcx,%rdx)
	vmulps	224(%rax,%rdx,4), %ymm3, %ymm4
	vroundps	$12, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm4, %ymm4
	vminps	%ymm1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vextracti128	$1, %ymm4, %xmm5
	vpackssdw	%xmm5, %xmm4, %xmm4
	vpacksswb	%xmm4, %xmm4, %xmm4
	vmovq	%xmm4, 56(%rcx,%rdx)
	addq	$64, %rdx
	cmpq	$4096, %rdx
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
