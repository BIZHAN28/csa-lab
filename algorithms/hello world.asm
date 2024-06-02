section .data
hello_len: 14
hello: "Hello, world!\n"
section .code
.start:
    # I/O
    movi x15, 0
    movi x0, hello_len
    ld x0, x0
    movi x1, hello
.loop:
    je x0, .break
    ld x2, x1
    out x15, x2
    inc x1
    dec x0
    jmp .loop
.break:
    hlt
