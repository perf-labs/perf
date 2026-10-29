    .text
    .p2align 4
.Lret1:
    movl $1, %eax
    ret
    .globl h
    .type h, @function
h:
    jmp .Ljmp
    .size h, .-h
    .globl f
    .type f, @function
f:
    testl %edi, %edi
    jne h
    je .Lafter
.Ljmp:
    jmp .Lret1
.Lafter:
    movl $2, %eax
    ret
    .size f, .-f
    .section .note.GNU-stack,"",@progbits
