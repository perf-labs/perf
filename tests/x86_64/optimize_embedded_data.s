    .text
    .p2align 4
    .globl hot
    .type hot, @function
hot:
    jmp .Lcode
.Ldata:
    .byte 0xe9, 0x01, 0x02, 0x03, 0x04
.Lcode:
    lea .Ldata(%rip), %rax
    movl (%rax), %ecx
    cmpl $0x030201e9, %ecx
    setne %al
    movzbl %al, %eax
    ret
    .size hot, .-hot

    .globl main
    .type main, @function
main:
    call hot
    ret
    .size main, .-main

    .section .note.GNU-stack,"",@progbits
