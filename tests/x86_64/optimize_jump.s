    .text
    .globl hot
    .type hot, @function
hot:
    endbr64
    andl $15, %edi
    leaq .Ltable(%rip), %rdx
    movslq (%rdx,%rdi,4), %rax
    addq %rdx, %rax
    jmp *%rax
.L0:
    movl $11, %eax
    ret
.L1:
    movl $22, %eax
    ret
.L2:
    movl $33, %eax
    ret
.L3:
    movl $44, %eax
    ret
.L4:
    movl $55, %eax
    ret
.L5:
    movl $66, %eax
    ret
.L6:
    movl $77, %eax
    ret
.L7:
    movl $88, %eax
    ret
.L8:
    movl $99, %eax
    ret
.L9:
    movl $111, %eax
    ret
.L10:
    movl $122, %eax
    ret
.L11:
    movl $133, %eax
    ret
.L12:
    movl $144, %eax
    ret
.L13:
    movl $155, %eax
    ret
.L14:
    movl $166, %eax
    ret
.L15:
    movl $177, %eax
    ret
    .size hot, .-hot
    .section .rodata
    .align 4
.Ltable:
    .long .L0 - .Ltable
    .long .L1 - .Ltable
    .long .L2 - .Ltable
    .long .L3 - .Ltable
    .long .L4 - .Ltable
    .long .L5 - .Ltable
    .long .L6 - .Ltable
    .long .L7 - .Ltable
    .long .L8 - .Ltable
    .long .L9 - .Ltable
    .long .L10 - .Ltable
    .long .L11 - .Ltable
    .long .L12 - .Ltable
    .long .L13 - .Ltable
    .long .L14 - .Ltable
    .long .L15 - .Ltable
    .section .note.GNU-stack,"",@progbits
