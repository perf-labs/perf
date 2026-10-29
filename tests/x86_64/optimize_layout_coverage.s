    .text
    .p2align 4
    .globl g
    .type g, @function
g:
    movl %edi, %eax
    ret
    .size g, .-g
    .globl payload
    .type payload, @notype
payload:
    movl $42, %eax
    ret
    .section .note.GNU-stack,"",@progbits
