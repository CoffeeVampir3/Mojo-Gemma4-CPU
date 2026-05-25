	.att_syntax
	.file	"fwht_inspect.mojo"
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

	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI2_0:
	.long	0x3f800000
	.long	0xbf800000
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI2_1:
	.long	0x3f800000
	.long	0x3f800000
	.long	0xbf800000
	.long	0xbf800000
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI2_2:
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.LCPI2_3:
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0x3f800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.long	0xbf800000
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI2_4:
	.long	0x3e000000
.LCPI2_5:
	.long	0x3db504f3
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
	pushq	%rax
	.cfi_def_cfa_offset 32
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
	movq	$4, (%rsp)
	vmovups	4, %zmm3
	vmovups	68, %zmm4
	vmovups	132, %zmm5
	vmovups	196, %zmm1
	vshufps	$177, %zmm1, %zmm1, %zmm2
	vbroadcastsd	.LCPI2_0(%rip), %zmm0
	vfmadd231ps	%zmm1, %zmm0, %zmm2
	vshufpd	$85, %zmm2, %zmm2, %zmm6
	vbroadcastf32x4	.LCPI2_1(%rip), %zmm1
	vfmadd231ps	%zmm2, %zmm1, %zmm6
	vpermpd	$78, %zmm6, %zmm7
	vbroadcastf64x4	.LCPI2_2(%rip), %zmm2
	vfmadd231ps	%zmm6, %zmm2, %zmm7
	vshufps	$177, %zmm5, %zmm5, %zmm6
	vfmadd231ps	%zmm5, %zmm0, %zmm6
	vshufpd	$85, %zmm6, %zmm6, %zmm5
	vfmadd231ps	%zmm6, %zmm1, %zmm5
	vpxor	%xmm6, %xmm6, %xmm6
	vpermpd	$78, %zmm5, %zmm6
	vfmadd231ps	%zmm5, %zmm2, %zmm6
	vshufps	$177, %zmm4, %zmm4, %zmm5
	vfmadd231ps	%zmm4, %zmm0, %zmm5
	vshufpd	$85, %zmm5, %zmm5, %zmm4
	vfmadd231ps	%zmm5, %zmm1, %zmm4
	vpxor	%xmm5, %xmm5, %xmm5
	vpermpd	$78, %zmm4, %zmm5
	vfmadd231ps	%zmm4, %zmm2, %zmm5
	vshufps	$177, %zmm3, %zmm3, %zmm4
	vfmadd231ps	%zmm3, %zmm0, %zmm4
	vshufpd	$85, %zmm4, %zmm4, %zmm3
	vfmadd231ps	%zmm4, %zmm1, %zmm3
	vpxor	%xmm4, %xmm4, %xmm4
	vpermpd	$78, %zmm3, %zmm4
	vfmadd231ps	%zmm3, %zmm2, %zmm4
	vshuff64x2	$78, %zmm4, %zmm4, %zmm8
	vmovaps	.LCPI2_3(%rip), %zmm3
	vfmadd231ps	%zmm4, %zmm3, %zmm8
	vshuff64x2	$78, %zmm5, %zmm5, %zmm4
	vfmadd231ps	%zmm5, %zmm3, %zmm4
	vshuff64x2	$78, %zmm6, %zmm6, %zmm5
	vfmadd231ps	%zmm6, %zmm3, %zmm5
	vshuff64x2	$78, %zmm7, %zmm7, %zmm6
	vfmadd231ps	%zmm7, %zmm3, %zmm6
	vaddps	%zmm4, %zmm8, %zmm7
	vsubps	%zmm4, %zmm8, %zmm4
	vaddps	%zmm6, %zmm5, %zmm8
	vsubps	%zmm6, %zmm5, %zmm5
	vaddps	%zmm8, %zmm7, %zmm6
	vsubps	%zmm8, %zmm7, %zmm7
	vaddps	%zmm5, %zmm4, %zmm8
	vsubps	%zmm5, %zmm4, %zmm4
	vbroadcastss	.LCPI2_4(%rip), %zmm5
	vmulps	%zmm5, %zmm6, %zmm6
	vmulps	%zmm5, %zmm8, %zmm8
	vmulps	%zmm5, %zmm7, %zmm7
	vmulps	%zmm5, %zmm4, %zmm9
	vmovups	%zmm6, 4
	vmovups	%zmm8, 68
	vmovups	%zmm7, 132
	vmovups	%zmm9, 196
	vmovups	4, %zmm6
	vmovups	68, %zmm8
	vmovups	132, %zmm10
	vmovups	260, %zmm11
	vmovups	324, %zmm7
	vmovups	388, %zmm5
	vmovups	452, %zmm4
	vshufps	$177, %zmm4, %zmm4, %zmm12
	vfmadd231ps	%zmm4, %zmm0, %zmm12
	vshufpd	$85, %zmm12, %zmm12, %zmm13
	vfmadd231ps	%zmm12, %zmm1, %zmm13
	vpxor	%xmm4, %xmm4, %xmm4
	vpermpd	$78, %zmm13, %zmm4
	vfmadd231ps	%zmm13, %zmm2, %zmm4
	vshufps	$177, %zmm5, %zmm5, %zmm12
	vfmadd231ps	%zmm5, %zmm0, %zmm12
	vshufpd	$85, %zmm12, %zmm12, %zmm13
	vfmadd231ps	%zmm12, %zmm1, %zmm13
	vpxor	%xmm5, %xmm5, %xmm5
	vpermpd	$78, %zmm13, %zmm5
	vfmadd231ps	%zmm13, %zmm2, %zmm5
	vshufps	$177, %zmm7, %zmm7, %zmm12
	vfmadd231ps	%zmm7, %zmm0, %zmm12
	vshufpd	$85, %zmm12, %zmm12, %zmm13
	vfmadd231ps	%zmm12, %zmm1, %zmm13
	vpxor	%xmm7, %xmm7, %xmm7
	vpermpd	$78, %zmm13, %zmm7
	vfmadd231ps	%zmm13, %zmm2, %zmm7
	vshufps	$177, %zmm11, %zmm11, %zmm12
	vfmadd231ps	%zmm11, %zmm0, %zmm12
	vshufpd	$85, %zmm12, %zmm12, %zmm11
	vfmadd231ps	%zmm12, %zmm1, %zmm11
	vpxor	%xmm12, %xmm12, %xmm12
	vpermpd	$78, %zmm11, %zmm12
	vfmadd231ps	%zmm11, %zmm2, %zmm12
	vshufps	$177, %zmm9, %zmm9, %zmm11
	vfmadd231ps	%zmm9, %zmm0, %zmm11
	vshufpd	$85, %zmm11, %zmm11, %zmm9
	vfmadd231ps	%zmm11, %zmm1, %zmm9
	vpxor	%xmm11, %xmm11, %xmm11
	vpermpd	$78, %zmm9, %zmm11
	vfmadd231ps	%zmm9, %zmm2, %zmm11
	vshufps	$177, %zmm10, %zmm10, %zmm9
	vfmadd231ps	%zmm10, %zmm0, %zmm9
	vshufpd	$85, %zmm9, %zmm9, %zmm10
	vfmadd231ps	%zmm9, %zmm1, %zmm10
	vpxor	%xmm9, %xmm9, %xmm9
	vpermpd	$78, %zmm10, %zmm9
	vfmadd231ps	%zmm10, %zmm2, %zmm9
	vshufps	$177, %zmm8, %zmm8, %zmm10
	vfmadd231ps	%zmm8, %zmm0, %zmm10
	vshufpd	$85, %zmm10, %zmm10, %zmm8
	vfmadd231ps	%zmm10, %zmm1, %zmm8
	vpxor	%xmm10, %xmm10, %xmm10
	vpermpd	$78, %zmm8, %zmm10
	vfmadd231ps	%zmm8, %zmm2, %zmm10
	vshufps	$177, %zmm6, %zmm6, %zmm8
	vfmadd231ps	%zmm0, %zmm6, %zmm8
	vshufpd	$85, %zmm8, %zmm8, %zmm0
	vfmadd231ps	%zmm8, %zmm1, %zmm0
	vpermpd	$78, %zmm0, %zmm1
	vfmadd231ps	%zmm0, %zmm2, %zmm1
	vshuff64x2	$78, %zmm1, %zmm1, %zmm0
	vfmadd231ps	%zmm1, %zmm3, %zmm0
	vshuff64x2	$78, %zmm10, %zmm10, %zmm1
	vfmadd231ps	%zmm10, %zmm3, %zmm1
	vshuff64x2	$78, %zmm9, %zmm9, %zmm2
	vfmadd231ps	%zmm9, %zmm3, %zmm2
	vshuff64x2	$78, %zmm11, %zmm11, %zmm6
	vfmadd231ps	%zmm11, %zmm3, %zmm6
	vshuff64x2	$78, %zmm12, %zmm12, %zmm8
	vfmadd231ps	%zmm12, %zmm3, %zmm8
	vshuff64x2	$78, %zmm7, %zmm7, %zmm9
	vfmadd231ps	%zmm7, %zmm3, %zmm9
	vshuff64x2	$78, %zmm5, %zmm5, %zmm7
	vfmadd231ps	%zmm5, %zmm3, %zmm7
	vshuff64x2	$78, %zmm4, %zmm4, %zmm5
	vfmadd231ps	%zmm4, %zmm3, %zmm5
	vaddps	%zmm1, %zmm0, %zmm3
	vsubps	%zmm1, %zmm0, %zmm0
	vaddps	%zmm6, %zmm2, %zmm1
	vsubps	%zmm6, %zmm2, %zmm2
	vaddps	%zmm9, %zmm8, %zmm4
	vsubps	%zmm9, %zmm8, %zmm6
	vaddps	%zmm5, %zmm7, %zmm8
	vsubps	%zmm5, %zmm7, %zmm5
	vaddps	%zmm1, %zmm3, %zmm7
	vsubps	%zmm1, %zmm3, %zmm1
	vaddps	%zmm2, %zmm0, %zmm3
	vsubps	%zmm2, %zmm0, %zmm0
	vaddps	%zmm8, %zmm4, %zmm2
	vsubps	%zmm8, %zmm4, %zmm4
	vaddps	%zmm5, %zmm6, %zmm8
	vsubps	%zmm5, %zmm6, %zmm5
	vaddps	%zmm2, %zmm7, %zmm6
	vsubps	%zmm2, %zmm7, %zmm2
	vaddps	%zmm8, %zmm3, %zmm7
	vsubps	%zmm8, %zmm3, %zmm3
	vaddps	%zmm4, %zmm1, %zmm8
	vsubps	%zmm4, %zmm1, %zmm1
	vaddps	%zmm5, %zmm0, %zmm4
	vsubps	%zmm5, %zmm0, %zmm0
	vbroadcastss	.LCPI2_5(%rip), %zmm5
	vmulps	%zmm5, %zmm6, %zmm6
	vmulps	%zmm5, %zmm7, %zmm7
	vmulps	%zmm5, %zmm8, %zmm8
	vmulps	%zmm5, %zmm4, %zmm4
	vmulps	%zmm5, %zmm2, %zmm2
	vmulps	%zmm5, %zmm3, %zmm3
	vmulps	%zmm5, %zmm1, %zmm1
	vmulps	%zmm5, %zmm0, %zmm0
	vmovups	%zmm6, 4
	vmovups	%zmm7, 68
	vmovups	%zmm8, 132
	vmovups	%zmm4, 196
	vmovups	%zmm2, 260
	vmovups	%zmm3, 324
	vmovups	%zmm1, 388
	vmovups	%zmm0, 452
	movq	%rsp, %rax
	#APP
	#NO_APP
	vzeroupper
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$8, %rsp
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
