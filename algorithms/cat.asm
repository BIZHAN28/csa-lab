section .code
.start:
    movi x1, 0
.loop:
    in x0, x1
    je x0, .exit
    out x1, x0
    jmp .loop
.exit:
    hlt

