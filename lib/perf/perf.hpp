// The MIT License (MIT)
//
// Copyright (c) 2026 Kris Jusiak <kris@jusiak.net>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef PERF_HPP
#define PERF_HPP

#ifndef PERF_LABEL
#if defined(__clang__)
#define PERF_LABEL(name)                                  \
    asm volatile(                                         \
        ".pushsection .perf.label, \"awR?\", @progbits\n" \
        ".quad 0f\n"                                      \
        ".asciz \"" #name "\"\n"                          \
        ".popsection\n"                                   \
        "0:\n"                                            \
    ); name:
#else
#define PERF_LABEL(name)                                  \
    asm volatile goto(                                    \
        ".pushsection .perf.label, \"awR?\", @progbits\n" \
        ".quad %l0\n"                                     \
        ".asciz \"" #name "\"\n"                          \
        ".popsection\n"                                   \
        :::: name                                         \
    ); name:
#endif
#endif

#endif // PERF_HPP
