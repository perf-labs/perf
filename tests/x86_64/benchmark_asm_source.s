	.text
	.globl asm_bench
	.type asm_bench, @function
asm_bench:
	mov eax, 42
	ret
	.size asm_bench, .-asm_bench
