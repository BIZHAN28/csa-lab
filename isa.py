from __future__ import annotations

from enum import Enum


class Opcode(str, Enum):
    NOP = "nop"
    IN = "in"
    OUT = "out"
    MOV = "mov"
    MOVI = "movi"
    LD = "ld"
    ST = "st"
    INC = "inc"
    DEC = "dec"
    NEG = "neg"
    ADD = "add"
    SUB = "sub"
    DIV = "div"
    MOD = "mod"
    AND = "and"
    OR = "or"
    NOT = "not"
    JG = "jg"
    JE = "je"
    JUMP = "jmp"
    HLT = "hlt"

    def __str__(self):
        return self.name


class Instruction:
    def __init__(self, opcode: Opcode, args: list[str] | None = None):
        if args is None:
            args = []
        self.opcode: Opcode = opcode
        self.args: list[str] = args

    def __str__(self):
        return f"({self.opcode} {self.args})"

    def __repr__(self):
        return self.__str__()

