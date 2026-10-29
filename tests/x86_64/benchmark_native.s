.intel_syntax noprefix
benchmark_native:
add eax, ebx
sub eax, ebx
add eax, 1
sub eax, 1
add eax, 2
inc eax
dec eax
neg eax
not eax
add rax, rbx
sub rax, rbx
neg rax
not rax
shl eax, 1
shr eax, 1
sar eax, 1
shl rax, 1
shr rax, 1
sar rax, 1
imul eax, ebx
paddd xmm0, xmm1
paddq xmm0, xmm1
lea eax, [eax + 8]
lea eax, [eax + 4]
lea eax, [eax + 2]
