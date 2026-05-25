	.att_syntax
	.file	"vnni_inspect.mojo"
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
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
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
	vmovd	1, %xmm1
	vpbroadcastd	.LCPI2_1(%rip), %xmm0
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpxor	%xmm2, %xmm2, %xmm2
	vpdpbusd	1, %zmm1, %zmm2
	vmovd	5, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	65, %zmm1, %zmm2
	vmovd	9, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	129, %zmm1, %zmm2
	vmovd	13, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	193, %zmm1, %zmm2
	vmovd	17, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	257, %zmm1, %zmm2
	vmovd	21, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	321, %zmm1, %zmm2
	vmovd	25, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	385, %zmm1, %zmm2
	vmovd	29, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %zmm1
	vpdpbusd	449, %zmm1, %zmm2
	vmovdqu64	%zmm2, 4
	vmovd	1, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	vpxor	%xmm2, %xmm2, %xmm2
	{vex}	vpdpbusd	1, %ymm1, %ymm2
	vmovd	5, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	33, %ymm1, %ymm2
	vmovd	9, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	65, %ymm1, %ymm2
	vmovd	13, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	97, %ymm1, %ymm2
	vmovd	17, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	129, %ymm1, %ymm2
	vmovd	21, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	161, %ymm1, %ymm2
	vmovd	25, %xmm1
	vpxor	%xmm0, %xmm1, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	{vex}	vpdpbusd	193, %ymm1, %ymm2
	vmovd	29, %xmm1
	vpxor	%xmm0, %xmm1, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	{vex}	vpdpbusd	225, %ymm0, %ymm2
	vmovdqu	%ymm2, 4
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
