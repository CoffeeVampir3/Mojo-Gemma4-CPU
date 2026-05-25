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
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI2_3:
	.long	0x3e000000
.LCPI2_4:
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
	subq	$248, %rsp
	.cfi_def_cfa_offset 272
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
	movq	$4, 104(%rsp)
	vmovups	4, %ymm0
	vmovups	36, %ymm6
	vmovups	68, %ymm7
	vmovups	100, %ymm9
	vmovups	132, %ymm2
	vmovups	164, %ymm5
	vmovups	196, %ymm4
	vmovups	228, %ymm1
	vshufps	$177, %ymm1, %ymm1, %ymm10
	vbroadcastsd	.LCPI2_0(%rip), %ymm8
	vfmadd231ps	%ymm1, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm3
	vbroadcastf128	.LCPI2_1(%rip), %ymm1
	vfmadd231ps	%ymm10, %ymm1, %ymm3
	vshufps	$177, %ymm4, %ymm4, %ymm10
	vfmadd231ps	%ymm4, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm4
	vfmadd231ps	%ymm10, %ymm1, %ymm4
	vshufps	$177, %ymm5, %ymm5, %ymm10
	vfmadd231ps	%ymm5, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm5
	vfmadd231ps	%ymm10, %ymm1, %ymm5
	vshufps	$177, %ymm2, %ymm2, %ymm10
	vfmadd231ps	%ymm2, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm2
	vfmadd231ps	%ymm10, %ymm1, %ymm2
	vshufps	$177, %ymm9, %ymm9, %ymm10
	vfmadd231ps	%ymm9, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm9
	vfmadd231ps	%ymm10, %ymm1, %ymm9
	vshufps	$177, %ymm7, %ymm7, %ymm10
	vfmadd231ps	%ymm7, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm7
	vfmadd231ps	%ymm10, %ymm1, %ymm7
	vshufps	$177, %ymm6, %ymm6, %ymm10
	vfmadd231ps	%ymm6, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm6
	vfmadd231ps	%ymm10, %ymm1, %ymm6
	vshufps	$177, %ymm0, %ymm0, %ymm10
	vfmadd231ps	%ymm0, %ymm8, %ymm10
	vshufpd	$5, %ymm10, %ymm10, %ymm0
	vfmadd231ps	%ymm10, %ymm1, %ymm0
	vxorps	%xmm10, %xmm10, %xmm10
	vpermpd	$78, %ymm0, %ymm10
	vmovaps	.LCPI2_2(%rip), %ymm11
	vfmadd231ps	%ymm0, %ymm11, %ymm10
	vxorps	%xmm0, %xmm0, %xmm0
	vpermpd	$78, %ymm6, %ymm0
	vfmadd231ps	%ymm6, %ymm11, %ymm0
	vxorps	%xmm6, %xmm6, %xmm6
	vpermpd	$78, %ymm7, %ymm6
	vfmadd231ps	%ymm7, %ymm11, %ymm6
	vxorps	%xmm7, %xmm7, %xmm7
	vpermpd	$78, %ymm9, %ymm7
	vfmadd231ps	%ymm9, %ymm11, %ymm7
	vxorps	%xmm9, %xmm9, %xmm9
	vpermpd	$78, %ymm2, %ymm9
	vfmadd231ps	%ymm2, %ymm11, %ymm9
	vxorps	%xmm2, %xmm2, %xmm2
	vpermpd	$78, %ymm5, %ymm2
	vfmadd231ps	%ymm5, %ymm11, %ymm2
	vxorps	%xmm5, %xmm5, %xmm5
	vpermpd	$78, %ymm4, %ymm5
	vfmadd231ps	%ymm4, %ymm11, %ymm5
	vxorps	%xmm4, %xmm4, %xmm4
	vpermpd	$78, %ymm3, %ymm4
	vfmadd231ps	%ymm3, %ymm11, %ymm4
	vaddps	%ymm0, %ymm10, %ymm3
	vsubps	%ymm0, %ymm10, %ymm0
	vaddps	%ymm7, %ymm6, %ymm10
	vsubps	%ymm7, %ymm6, %ymm6
	vaddps	%ymm2, %ymm9, %ymm7
	vsubps	%ymm2, %ymm9, %ymm2
	vaddps	%ymm4, %ymm5, %ymm9
	vsubps	%ymm4, %ymm5, %ymm4
	vaddps	%ymm3, %ymm10, %ymm5
	vsubps	%ymm10, %ymm3, %ymm3
	vaddps	%ymm6, %ymm0, %ymm10
	vsubps	%ymm6, %ymm0, %ymm0
	vaddps	%ymm7, %ymm9, %ymm6
	vsubps	%ymm9, %ymm7, %ymm7
	vaddps	%ymm4, %ymm2, %ymm9
	vsubps	%ymm4, %ymm2, %ymm2
	vaddps	%ymm6, %ymm5, %ymm4
	vsubps	%ymm6, %ymm5, %ymm5
	vaddps	%ymm9, %ymm10, %ymm6
	vsubps	%ymm9, %ymm10, %ymm9
	vaddps	%ymm7, %ymm3, %ymm10
	vsubps	%ymm7, %ymm3, %ymm3
	vaddps	%ymm2, %ymm0, %ymm7
	vsubps	%ymm2, %ymm0, %ymm0
	vbroadcastss	.LCPI2_3(%rip), %ymm2
	vmulps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm2, %ymm6, %ymm6
	vmulps	%ymm2, %ymm10, %ymm11
	vmulps	%ymm2, %ymm7, %ymm7
	vmulps	%ymm2, %ymm5, %ymm5
	vmulps	%ymm2, %ymm9, %ymm9
	vmulps	%ymm2, %ymm3, %ymm3
	vmulps	%ymm2, %ymm0, %ymm10
	vmovups	%ymm4, 4
	vmovups	%ymm6, 36
	vmovups	%ymm11, 68
	vmovups	%ymm7, 100
	vmovups	%ymm5, 132
	vmovups	%ymm9, 164
	vmovups	%ymm3, 196
	vmovups	%ymm10, 228
	vmovups	132, %ymm2
	vmovups	164, %ymm12
	vmovups	196, %ymm15
	vmovups	260, %ymm14
	vmovups	292, %ymm0
	vmovups	324, %ymm3
	vmovups	356, %ymm4
	vmovups	388, %ymm7
	vmovups	420, %ymm6
	vmovups	452, %ymm5
	vmovups	484, %ymm9
	vshufps	$177, %ymm9, %ymm9, %ymm11
	vfmadd231ps	%ymm9, %ymm8, %ymm11
	vmovups	%ymm11, (%rsp)
	vshufps	$177, %ymm5, %ymm5, %ymm9
	vfmadd231ps	%ymm5, %ymm8, %ymm9
	vmovups	%ymm9, 64(%rsp)
	vshufps	$177, %ymm6, %ymm6, %ymm5
	vfmadd231ps	%ymm6, %ymm8, %ymm5
	vmovups	%ymm5, 32(%rsp)
	vshufps	$177, %ymm7, %ymm7, %ymm6
	vfmadd231ps	%ymm7, %ymm8, %ymm6
	vshufps	$177, %ymm4, %ymm4, %ymm7
	vfmadd231ps	%ymm4, %ymm8, %ymm7
	vshufps	$177, %ymm3, %ymm3, %ymm9
	vfmadd231ps	%ymm3, %ymm8, %ymm9
	vshufps	$177, %ymm0, %ymm0, %ymm11
	vfmadd231ps	%ymm0, %ymm8, %ymm11
	vshufps	$177, %ymm14, %ymm14, %ymm13
	vfmadd231ps	%ymm14, %ymm8, %ymm13
	vshufps	$177, %ymm10, %ymm10, %ymm14
	vfmadd231ps	%ymm10, %ymm8, %ymm14
	vshufps	$177, %ymm15, %ymm15, %ymm10
	vfmadd231ps	%ymm15, %ymm8, %ymm10
	vshufps	$177, %ymm12, %ymm12, %ymm15
	vfmadd231ps	%ymm12, %ymm8, %ymm15
	vshufps	$177, %ymm2, %ymm2, %ymm12
	vfmadd231ps	%ymm2, %ymm8, %ymm12
	vmovups	100, %ymm0
	vshufps	$177, %ymm0, %ymm0, %ymm5
	vfmadd231ps	%ymm0, %ymm8, %ymm5
	vmovups	68, %ymm0
	vshufps	$177, %ymm0, %ymm0, %ymm4
	vfmadd231ps	%ymm0, %ymm8, %ymm4
	vmovups	36, %ymm0
	vshufps	$177, %ymm0, %ymm0, %ymm3
	vfmadd231ps	%ymm0, %ymm8, %ymm3
	vmovups	4, %ymm0
	vshufps	$177, %ymm0, %ymm0, %ymm2
	vfmadd231ps	%ymm8, %ymm0, %ymm2
	vmovupd	(%rsp), %ymm0
	vshufpd	$5, %ymm0, %ymm0, %ymm8
	vfmadd231ps	%ymm0, %ymm1, %ymm8
	vmovups	%ymm8, (%rsp)
	vmovupd	64(%rsp), %ymm0
	vshufpd	$5, %ymm0, %ymm0, %ymm8
	vfmadd231ps	%ymm0, %ymm1, %ymm8
	vmovups	%ymm8, 64(%rsp)
	vmovupd	32(%rsp), %ymm0
	vshufpd	$5, %ymm0, %ymm0, %ymm8
	vfmadd231ps	%ymm0, %ymm1, %ymm8
	vmovups	%ymm8, 32(%rsp)
	vshufpd	$5, %ymm6, %ymm6, %ymm0
	vfmadd231ps	%ymm6, %ymm1, %ymm0
	vmovups	%ymm0, 144(%rsp)
	vshufpd	$5, %ymm7, %ymm7, %ymm6
	vfmadd231ps	%ymm7, %ymm1, %ymm6
	vshufpd	$5, %ymm9, %ymm9, %ymm7
	vfmadd231ps	%ymm9, %ymm1, %ymm7
	vshufpd	$5, %ymm11, %ymm11, %ymm8
	vfmadd231ps	%ymm11, %ymm1, %ymm8
	vshufpd	$5, %ymm13, %ymm13, %ymm9
	vfmadd231ps	%ymm13, %ymm1, %ymm9
	vshufpd	$5, %ymm14, %ymm14, %ymm13
	vfmadd231ps	%ymm14, %ymm1, %ymm13
	vshufpd	$5, %ymm10, %ymm10, %ymm14
	vfmadd231ps	%ymm10, %ymm1, %ymm14
	vshufpd	$5, %ymm15, %ymm15, %ymm0
	vfmadd231ps	%ymm15, %ymm1, %ymm0
	vshufpd	$5, %ymm12, %ymm12, %ymm15
	vfmadd231ps	%ymm12, %ymm1, %ymm15
	vshufpd	$5, %ymm5, %ymm5, %ymm11
	vfmadd231ps	%ymm5, %ymm1, %ymm11
	vshufpd	$5, %ymm4, %ymm4, %ymm12
	vfmadd231ps	%ymm4, %ymm1, %ymm12
	vshufpd	$5, %ymm3, %ymm3, %ymm4
	vfmadd231ps	%ymm3, %ymm1, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm3
	vfmadd231ps	%ymm2, %ymm1, %ymm3
	vpermpd	$78, %ymm3, %ymm1
	vmovaps	.LCPI2_2(%rip), %ymm5
	vfmadd231ps	%ymm3, %ymm5, %ymm1
	vmovups	%ymm1, 112(%rsp)
	vxorps	%xmm1, %xmm1, %xmm1
	vpermpd	$78, %ymm4, %ymm1
	vfmadd231ps	%ymm4, %ymm5, %ymm1
	vmovups	%ymm1, 208(%rsp)
	vxorps	%xmm1, %xmm1, %xmm1
	vpermpd	$78, %ymm12, %ymm1
	vfmadd231ps	%ymm12, %ymm5, %ymm1
	vmovups	%ymm1, 176(%rsp)
	vpermpd	$78, %ymm11, %ymm10
	vfmadd231ps	%ymm11, %ymm5, %ymm10
	vxorps	%xmm11, %xmm11, %xmm11
	vpermpd	$78, %ymm15, %ymm11
	vfmadd231ps	%ymm15, %ymm5, %ymm11
	vxorps	%xmm3, %xmm3, %xmm3
	vpermpd	$78, %ymm0, %ymm3
	vfmadd231ps	%ymm0, %ymm5, %ymm3
	vxorps	%xmm12, %xmm12, %xmm12
	vpermpd	$78, %ymm14, %ymm12
	vfmadd231ps	%ymm14, %ymm5, %ymm12
	vxorps	%xmm4, %xmm4, %xmm4
	vpermpd	$78, %ymm13, %ymm4
	vfmadd231ps	%ymm13, %ymm5, %ymm4
	vxorps	%xmm13, %xmm13, %xmm13
	vpermpd	$78, %ymm9, %ymm13
	vfmadd231ps	%ymm9, %ymm5, %ymm13
	vxorps	%xmm2, %xmm2, %xmm2
	vpermpd	$78, %ymm8, %ymm2
	vfmadd231ps	%ymm8, %ymm5, %ymm2
	vxorps	%xmm8, %xmm8, %xmm8
	vpermpd	$78, %ymm7, %ymm8
	vfmadd231ps	%ymm7, %ymm5, %ymm8
	vxorps	%xmm7, %xmm7, %xmm7
	vpermpd	$78, %ymm6, %ymm7
	vfmadd231ps	%ymm6, %ymm5, %ymm7
	vmovups	144(%rsp), %ymm0
	vxorps	%xmm6, %xmm6, %xmm6
	vpermpd	$78, %ymm0, %ymm6
	vfmadd231ps	%ymm0, %ymm5, %ymm6
	vmovups	32(%rsp), %ymm0
	vxorps	%xmm9, %xmm9, %xmm9
	vpermpd	$78, %ymm0, %ymm9
	vfmadd231ps	%ymm0, %ymm5, %ymm9
	vmovups	64(%rsp), %ymm0
	vxorps	%xmm14, %xmm14, %xmm14
	vpermpd	$78, %ymm0, %ymm14
	vfmadd231ps	%ymm0, %ymm5, %ymm14
	vmovups	(%rsp), %ymm0
	vxorps	%xmm15, %xmm15, %xmm15
	vpermpd	$78, %ymm0, %ymm15
	vfmadd231ps	%ymm0, %ymm5, %ymm15
	vmovups	112(%rsp), %ymm1
	vmovups	208(%rsp), %ymm0
	vaddps	%ymm0, %ymm1, %ymm5
	vsubps	%ymm0, %ymm1, %ymm0
	vmovups	%ymm0, (%rsp)
	vmovups	176(%rsp), %ymm1
	vaddps	%ymm1, %ymm10, %ymm0
	vsubps	%ymm10, %ymm1, %ymm1
	vaddps	%ymm3, %ymm11, %ymm10
	vsubps	%ymm3, %ymm11, %ymm3
	vaddps	%ymm4, %ymm12, %ymm11
	vsubps	%ymm4, %ymm12, %ymm4
	vaddps	%ymm2, %ymm13, %ymm12
	vsubps	%ymm2, %ymm13, %ymm2
	vmovups	%ymm2, 32(%rsp)
	vaddps	%ymm7, %ymm8, %ymm13
	vsubps	%ymm7, %ymm8, %ymm7
	vaddps	%ymm6, %ymm9, %ymm8
	vsubps	%ymm9, %ymm6, %ymm6
	vaddps	%ymm15, %ymm14, %ymm9
	vsubps	%ymm15, %ymm14, %ymm14
	vaddps	%ymm0, %ymm5, %ymm15
	vsubps	%ymm0, %ymm5, %ymm0
	vmovups	%ymm0, 64(%rsp)
	vmovups	(%rsp), %ymm0
	vaddps	%ymm1, %ymm0, %ymm5
	vsubps	%ymm1, %ymm0, %ymm2
	vaddps	%ymm11, %ymm10, %ymm0
	vsubps	%ymm11, %ymm10, %ymm10
	vaddps	%ymm4, %ymm3, %ymm11
	vsubps	%ymm4, %ymm3, %ymm3
	vaddps	%ymm13, %ymm12, %ymm4
	vsubps	%ymm13, %ymm12, %ymm12
	vmovups	32(%rsp), %ymm1
	vaddps	%ymm7, %ymm1, %ymm13
	vsubps	%ymm7, %ymm1, %ymm7
	vaddps	%ymm9, %ymm8, %ymm1
	vsubps	%ymm9, %ymm8, %ymm8
	vaddps	%ymm6, %ymm14, %ymm9
	vsubps	%ymm14, %ymm6, %ymm6
	vaddps	%ymm0, %ymm15, %ymm14
	vsubps	%ymm0, %ymm15, %ymm0
	vmovups	%ymm0, (%rsp)
	vaddps	%ymm5, %ymm11, %ymm15
	vsubps	%ymm11, %ymm5, %ymm5
	vmovups	64(%rsp), %ymm0
	vaddps	%ymm0, %ymm10, %ymm11
	vsubps	%ymm10, %ymm0, %ymm10
	vaddps	%ymm3, %ymm2, %ymm0
	vsubps	%ymm3, %ymm2, %ymm2
	vaddps	%ymm1, %ymm4, %ymm3
	vsubps	%ymm1, %ymm4, %ymm1
	vaddps	%ymm9, %ymm13, %ymm4
	vsubps	%ymm9, %ymm13, %ymm9
	vaddps	%ymm8, %ymm12, %ymm13
	vsubps	%ymm8, %ymm12, %ymm8
	vaddps	%ymm6, %ymm7, %ymm12
	vsubps	%ymm6, %ymm7, %ymm6
	vaddps	%ymm3, %ymm14, %ymm7
	vsubps	%ymm3, %ymm14, %ymm3
	vaddps	%ymm4, %ymm15, %ymm14
	vsubps	%ymm4, %ymm15, %ymm4
	vaddps	%ymm13, %ymm11, %ymm15
	vsubps	%ymm13, %ymm11, %ymm11
	vaddps	%ymm0, %ymm12, %ymm13
	vsubps	%ymm12, %ymm0, %ymm0
	vmovups	%ymm0, 64(%rsp)
	vmovups	(%rsp), %ymm0
	vaddps	%ymm1, %ymm0, %ymm12
	vsubps	%ymm1, %ymm0, %ymm0
	vmovups	%ymm0, (%rsp)
	vaddps	%ymm5, %ymm9, %ymm0
	vsubps	%ymm9, %ymm5, %ymm1
	vmovups	%ymm1, 32(%rsp)
	vaddps	%ymm8, %ymm10, %ymm9
	vsubps	%ymm8, %ymm10, %ymm1
	vmovups	%ymm1, 144(%rsp)
	vaddps	%ymm6, %ymm2, %ymm10
	vsubps	%ymm6, %ymm2, %ymm1
	vmovups	%ymm1, 112(%rsp)
	vbroadcastss	.LCPI2_4(%rip), %ymm6
	vmulps	%ymm6, %ymm7, %ymm7
	vmulps	%ymm6, %ymm14, %ymm14
	vmulps	%ymm6, %ymm15, %ymm15
	vmulps	%ymm6, %ymm13, %ymm13
	vmulps	%ymm6, %ymm12, %ymm12
	vmulps	%ymm6, %ymm0, %ymm0
	vmulps	%ymm6, %ymm9, %ymm9
	vmulps	%ymm6, %ymm10, %ymm10
	vmulps	%ymm6, %ymm3, %ymm1
	vmulps	%ymm6, %ymm4, %ymm2
	vmulps	%ymm6, %ymm11, %ymm3
	vmulps	64(%rsp), %ymm6, %ymm4
	vmulps	(%rsp), %ymm6, %ymm5
	vmulps	32(%rsp), %ymm6, %ymm8
	vmulps	144(%rsp), %ymm6, %ymm11
	vmulps	112(%rsp), %ymm6, %ymm6
	vmovups	%ymm7, 4
	vmovups	%ymm14, 36
	vmovups	%ymm15, 68
	vmovups	%ymm13, 100
	vmovups	%ymm12, 132
	vmovups	%ymm0, 164
	vmovups	%ymm9, 196
	vmovups	%ymm10, 228
	vmovups	%ymm1, 260
	vmovups	%ymm2, 292
	vmovups	%ymm3, 324
	vmovups	%ymm4, 356
	vmovups	%ymm5, 388
	vmovups	%ymm8, 420
	vmovups	%ymm11, 452
	vmovups	%ymm6, 484
	leaq	104(%rsp), %rax
	#APP
	#NO_APP
	vzeroupper
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$248, %rsp
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
