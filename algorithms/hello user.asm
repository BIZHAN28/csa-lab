section .data
hello_len: 23
hello: "\n - What is your name?\n"
bye_len: 11
bye: "\n - Hello, "
buf_size: 100
buf_len: 0
buf: "                                                                                                    "
section .code
.start:
    # I/O
    movi x15, 0
    movi x0, hello_len
    ld x0, x0
    movi x1, hello
.loop1:
    je x0, .next1
    ld x3, x1
    out x15, x3
    inc x1
    dec x0
    jmp .loop1
.next1:
    movi x0, buf_size
    ld x0, x0
    movi x1, buf
.loop2:
    je x0, .next2
    in x2, x15
    je x2, .next2
    st x1, x2
    inc x1
    dec x0
    jmp .loop2
.next2:
    movi x1, buf_size
    ld x1, x1
    sub x1, x1, x0
    movi x0, buf_len
    st x0, x1
    movi x0, bye_len
    ld x0, x0
    movi x1, bye
.loop3:
    je x0, .next3
    ld x3, x1
    out x15, x3
    inc x1
    dec x0
    jmp .loop3
.next3:
    movi x0, buf_len
    ld x0, x0
    movi x1, buf
.loop4:
    je x0, .next4
    ld x3, x1
    out x15, x3
    inc x1
    dec x0
    jmp .loop4
.next4:
    hlt
