    .text
    .p2align 4
.macro cold name
    .globl \name
    .type \name, @function
\name:
    movl $1, %eax
    addl $2, %eax
    addl $3, %eax
    addl $4, %eax
    ret
    .size \name, .-\name
.endm
    cold c00
    cold c01
    cold c02
    cold c03
    cold c04
    cold c05
    cold c06
    cold c07
    cold c08
    cold c09
    cold c10
    cold c11
    cold c12

    .globl h
    .type h, @function
h:
    jmp t
    .size h, .-h

    .globl t
    .type t, @function
t:
    movl $7, %eax
    ret
    .size t, .-t

    .globl main
    .type main, @function
main:
    call h
    cmpl $7, %eax
    setne %al
    movzbl %al, %eax
    ret
    .size main, .-main

    .section .note.GNU-stack,"",@progbits
