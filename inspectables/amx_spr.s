	.att_syntax
	.file	"amx_inspect.mojo"
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

	.globl	main
	.prefalign	4, .Lfunc_end2, nop
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
	jne	.LBB2_2
	leaq	static_string_a61c3395ab9379d9(%rip), %rdi
	movq	main_closure_0@GOTPCREL(%rip), %rdx
	movq	main_closure_1@GOTPCREL(%rip), %rcx
	movl	$7, %esi
	callq	KGEN_CompilerRT_GetOrCreateGlobal@PLT
.LBB2_2:
	movl	$1, %r14d
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movq	$1, (%rsp)
	movq	$1, 8(%rsp)
	movq	$1, 16(%rsp)
	movq	$4, 24(%rsp)
	tilezero	%tmm4
	tilezero	%tmm5
	tilezero	%tmm6
	tilezero	%tmm7
	xorl	%ecx, %ecx
	movl	$1024, %eax
	movl	$64, %edx
	movabsq	$-9223372036854775808, %rsi
	movl	$1, %edi
	.p2align	4
.LBB2_3:
	leaq	64(%rcx), %r8
	leaq	(%r14,%rcx), %r9
	tileloadd	(%r9,%rax), %tmm0
	leaq	16384(%r14,%rcx), %r9
	tileloadd	(%r9,%rax), %tmm1
	tileloadd	(%rdi,%rdx), %tmm2
	tileloadd	(%rdi,%rdx), %tmm3
	tdpbssd	%tmm2, %tmm0, %tmm4
	tdpbssd	%tmm2, %tmm1, %tmm5
	tdpbssd	%tmm3, %tmm0, %tmm6
	tdpbssd	%tmm3, %tmm1, %tmm7
	addq	$-960, %rcx
	addq	$1024, %rdi
	cmpq	%rsi, %rcx
	movq	%r8, %rcx
	ja	.LBB2_3
	movl	$64, %eax
	movl	$4, %ecx
	tilestored	%tmm4, (%rcx,%rax)
	movl	$1028, %ecx
	tilestored	%tmm5, (%rcx,%rax)
	movl	$2052, %ecx
	tilestored	%tmm6, (%rcx,%rax)
	movl	$3076, %ecx
	tilestored	%tmm7, (%rcx,%rax)
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
