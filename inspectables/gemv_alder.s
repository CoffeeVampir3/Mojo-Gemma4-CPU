	.att_syntax
	.file	"gemv_inspect.mojo"
	.text
	.prefalign	4, .Lfunc_end0, nop
	.type	main_closure_0,@function
main_closure_0:
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice@PLT
.Lfunc_end0:
	.size	main_closure_0, .Lfunc_end0-main_closure_0
	.cfi_endproc

	.prefalign	4, .Lfunc_end1, nop
	.type	main_closure_1,@function
main_closure_1:
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice@PLT
.Lfunc_end1:
	.size	main_closure_1, .Lfunc_end1-main_closure_1
	.cfi_endproc

	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI2_0:
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
.LCPI2_1:
	.long	0x43000000
.LCPI2_2:
	.zero	4,128
	.text
	.globl	main
	.prefalign	4, .Lfunc_end2, nop
	.type	main,@function
main:
.Lmain$local:
	.type	.Lmain$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$40, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movl	%edi, %ebp
	callq	KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice@PLT
	testq	%rax, %rax
	jne	.LBB2_2
	leaq	static_string_a61c3395ab9379d9(%rip), %rdi
	movq	main_closure_0@GOTPCREL(%rip), %rdx
	movq	main_closure_1@GOTPCREL(%rip), %rcx
	movl	$7, %esi
	callq	KGEN_CompilerRT_GetOrCreateGlobal@PLT
.LBB2_2:
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$1, (%rsp)
	movq	$1, 8(%rsp)
	movq	$4, 16(%rsp)
	movq	$4, 24(%rsp)
	movq	$4, 32(%rsp)
	vpxor	%xmm2, %xmm2, %xmm2
	movl	$64, %edx
	movl	$1057, %eax
	movl	$1, %ecx
	vpbroadcastd	.LCPI2_2(%rip), %xmm4
	movabsq	$-9223372036854775808, %rsi
	vpxor	%xmm3, %xmm3, %xmm3
	vpxor	%xmm1, %xmm1, %xmm1
	vpxor	%xmm0, %xmm0, %xmm0
	.p2align	4
.LBB2_3:
	movq	$-256, %rdi
	movq	%rcx, %r8
	.p2align	4
.LBB2_4:
	vmovd	(%r8), %xmm5
	vpxor	%xmm4, %xmm5, %xmm5
	vpbroadcastd	%xmm5, %ymm5
	{vex}	vpdpbusd	-32(%rax,%rdi,4), %ymm5, %ymm2
	{vex}	vpdpbusd	(%rax,%rdi,4), %ymm5, %ymm3
	{vex}	vpdpbusd	992(%rax,%rdi,4), %ymm5, %ymm1
	{vex}	vpdpbusd	1024(%rax,%rdi,4), %ymm5, %ymm0
	addq	$4, %r8
	addq	$16, %rdi
	jne	.LBB2_4
	leaq	64(%rdx), %rdi
	addq	$-1024, %rdx
	addq	$2048, %rax
	addq	$64, %rcx
	cmpq	%rsi, %rdx
	movq	%rdi, %rdx
	ja	.LBB2_3
	vcvtdq2ps	%ymm2, %ymm2
	vmovups	4, %ymm4
	vbroadcastss	.LCPI2_1(%rip), %ymm5
	vfnmadd231ps	%ymm5, %ymm4, %ymm2
	vmulps	%ymm2, %ymm4, %ymm2
	vmovups	%ymm2, 4
	vcvtdq2ps	%ymm3, %ymm2
	vmovups	36, %ymm3
	vfnmadd231ps	%ymm5, %ymm3, %ymm2
	vmulps	%ymm2, %ymm3, %ymm2
	vmovups	%ymm2, 36
	vcvtdq2ps	%ymm1, %ymm1
	vmovups	68, %ymm2
	vfnmadd231ps	%ymm5, %ymm2, %ymm1
	vmulps	%ymm1, %ymm2, %ymm1
	vmovups	%ymm1, 68
	vcvtdq2ps	%ymm0, %ymm0
	vmovups	100, %ymm1
	vfnmadd231ps	%ymm5, %ymm1, %ymm0
	vmulps	%ymm0, %ymm1, %ymm0
	vmovups	%ymm0, 100
	movq	%rsp, %rax
	#APP
	#NO_APP
	leaq	8(%rsp), %rax
	#APP
	#NO_APP
	leaq	16(%rsp), %rax
	#APP
	#NO_APP
	leaq	24(%rsp), %rax
	#APP
	#NO_APP
	leaq	32(%rsp), %rax
	#APP
	#NO_APP
	vzeroupper
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$40, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.size	.Lmain$local, .Lfunc_end2-main
	.cfi_endproc

	.type	static_string_a61c3395ab9379d9,@object
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
static_string_a61c3395ab9379d9:
	.asciz	"Runtime"
	.size	static_string_a61c3395ab9379d9, 8

	.section	".note.GNU-stack","",@progbits
