.intel_syntax noprefix
branch_penalty:
test edi, 1
jz done
add eax, 1; done: nop
