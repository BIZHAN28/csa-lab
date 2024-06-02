section .data
buf_size: 100
buf_len: 0
buf: "          "
section .code
.start:
    # res
    movi x1, 0
    movi x3, 3
    movi x5, 5
    # bound
    movi x15, 10
.loop:
    je x15, .break
    dec x15
    mod x13, x15, x3
    je x13, .add
    mod x13, x15, x5
    je x13, .add
    jmp .loop
.add:
    add x1, x1, x15
    jmp .loop
.break:
    movi x10, 10
    movi x5, buf
.num_loop:
    mod x13, x1, x10
    div x1, x1, x10
    # ord('0') = 48
    movi x3, 48 
    add x13, x13, x3
    st x5, x13
    inc x5
    je x1, .print
    jmp .num_loop
.print:
    movi x9, buf
    dec x9
.print_loop:
    sub x8, x5, x9
    je x8, .hlt
    movi x13, 0
    ld x4, x5
    out x13, x4
    dec x5
    jmp .print_loop
.hlt:
    hlt

