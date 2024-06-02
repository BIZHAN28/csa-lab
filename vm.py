#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import sys
from enum import Enum

from isa import Instruction, Opcode
from util import Register, is_register

COLORED = False

class Signal(str, Enum):
    # Memory
    WriteMem = "write_mem"
    ReadMem = "read_mem"
    # IO
    WriteIO = "write_io"
    ReadIO = "read_io"
    # ALU multiplexer latch
    SelectSourceRegisterRight = "reg_alu_right"  # source 2
    IncRight = "inc_alu_right"
    DecRight = "dec_alu_right"
    ZeroRight = "zero_alu_right"
    # Reg writing
    ReadImmidate = "read_imm"
    # ALU
    ReadALU = "read_alu"
    SumALU = "sum_alu"
    SubALU = "sub_alu"
    NegALU = "neg_alu"
    DivALU = "div_alu"
    ModALU = "mod_alu"
    AndALU = "and_alu"
    OrALU = "or_alu"
    NotALU = "not_alu"
    CmpALU = "cmp_alu"
    # Controlling
    PCJumpTypeJE = "je"
    PCJumpTypeJG = "jg"
    PCJumpTypeJump = "jump"
    PCJumpTypeNext = "next_pc"
    MicroProgramCounterZero = "mpc_zero"
    MicroProgramCounterOpcode = "mpc_jump"
    MicroProgramCounterNext = "mpc_next"
    # latch
    LatchPC = "latch_pc"
    LatchMPCounter = "latch_mpc"


OPCODES_IMPLS = {
    Signal.SumALU: lambda a, b: a + b,
    Signal.SubALU: lambda a, b: a - b,
    # Signal.IncRight: lambda a, _: a + 1,
    # Signal.DecRight: lambda a, _: a - 1,
    Signal.DivALU: lambda a, b: a // b,
    Signal.ModALU: lambda a, b: a % b,
    Signal.AndALU: lambda a, b: a and b,
    Signal.OrALU: lambda a, b: a or b,
    Signal.NotALU: lambda a, _: not a,
}


class ALU:
    negative: bool = False
    zero: bool = False

    def process(self, sig: Signal, left: int, right: int) -> int:
        ans: int = 0
        if sig == Signal.CmpALU:
            calc = left - right
            self.negative = calc < 0
            self.zero = calc == 0
            ans = left
        elif sig in OPCODES_IMPLS:
            ans = OPCODES_IMPLS[sig](left, right)
            self.negative = ans < 0
            self.zero = ans == 0
        else:
            raise ValueError(f"Invalid signal: {sig}")
        return ans


class IO:
    def __init__(self, charset: list[int]):
        self.charset: list[int] = charset
        self.output: str = ""

    def read_byte(self) -> int:
        if len(self.charset) == 0:
            return 0

        value = self.charset[0]
        self.charset = self.charset[1:]

        return value

    def write_byte(self, value: int):
        self.output += chr(value)

    def __repr__(self):
        chars = ""
        for c in self.charset:
            chars += f"{c}  "
        return f"{self.charset} {self.output}"


MEM_SIZE = 2**10


class DataPath:
    alu: ALU = ALU()

    def __init__(
        self,
        start: int,
        code: dict[int, Instruction],
        data: dict[int, int],
        pm_io: dict[int, IO],
    ):
        self.data_mem: list[int] = [0 for _ in range(MEM_SIZE)]
        self.code_mem: list[Instruction] = [Instruction(Opcode.NOP) for _ in range(MEM_SIZE)]
        self.regs: dict[Register, int] = dict()
        self.pm_io = pm_io
        for reg in range(0, 16):
            self.regs[Register(f"x{reg}")] = 0

        for k, v in data.items():
            self.data_mem[int(k)] = v
        for k, v in code.items():
            self.code_mem[int(k)] = v

    def latch_reg(self, reg: Register, value: int):
        self.regs[reg] = value

    def load_reg(self, reg: Register) -> int:
        return self.regs[reg]

    def get_zero_flag(self) -> bool:
        return self.alu.zero

    def get_negative_flag(self) -> bool:
        return self.alu.negative

    def ld(self, memory_address: Register, target_register: Register):
        # выбираем значение из регистра как адрес
        # и защелкиваем значение из памяти в регистр назначения
        self.latch_reg(target_register, self.data_mem[self.load_reg(memory_address)])

    def st(self, data_reg: Register, addr_reg: Register):
        self.data_mem[self.load_reg(addr_reg)] = self.load_reg(data_reg)

    def input(self, io_address: Register, target_register: Register):
        self.latch_reg(target_register, self.pm_io[self.load_reg(io_address)].read_byte())

    def out(self, data_reg: Register, addr_reg: Register):
        self.pm_io[self.load_reg(addr_reg)].write_byte(self.load_reg(data_reg))


class HLTError(KeyError):
    pass


class ControlUnit:
    tick_counter = 0
    # fmt: off
    m_program = (
        (Signal.MicroProgramCounterOpcode, Signal.LatchMPCounter), # 0 - Instruction fetch
        # NOP
        (Signal.PCJumpTypeNext, Signal.LatchPC, Signal.MicroProgramCounterZero, Signal.LatchMPCounter), # 1
        # IN
        (Signal.ReadIO,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 2
        # OUT
        (Signal.WriteIO,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 3
        # MOV
        (Signal.OrALU, Signal.ZeroRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 4
        # MOVI
        (Signal.ReadImmidate,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 5
        # LD
        (Signal.ReadMem,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 6
        # ST
        (Signal.WriteMem,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 7
        # INC
        (Signal.SumALU, Signal.IncRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 8
        # DEC
        (Signal.SumALU, Signal.DecRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 9
        # NEG
        (Signal.NegALU, Signal.ZeroRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 10
        # ADD
        (Signal.SumALU, Signal.SelectSourceRegisterRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 11
        # SUB
        (Signal.SubALU, Signal.SelectSourceRegisterRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 12
        # DIV
        (Signal.DivALU, Signal.SelectSourceRegisterRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 13
        # MOD
        (Signal.ModALU, Signal.SelectSourceRegisterRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 14
        # AND
        (Signal.AndALU, Signal.SelectSourceRegisterRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 15
        # OR
        (Signal.OrALU, Signal.SelectSourceRegisterRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 16
        # NOT
        (Signal.NotALU, Signal.ZeroRight, Signal.ReadALU,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter,
         Signal.PCJumpTypeNext, Signal.LatchPC), # 17
        # JG
        (Signal.CmpALU, Signal.ZeroRight, Signal.PCJumpTypeJG, Signal.LatchPC,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter), # 18
        # JE
        (Signal.CmpALU, Signal.ZeroRight, Signal.PCJumpTypeJE, Signal.LatchPC,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter), # 19
        # JUMP
        (Signal.PCJumpTypeJump, Signal.LatchPC,
         Signal.MicroProgramCounterZero, Signal.LatchMPCounter), # 20
    )
    # fmt: on

    @staticmethod
    def opcode_to_mc(opcode: Opcode) -> int:
        try:
            return {
                Opcode.NOP: 1,
                Opcode.IN: 2,
                Opcode.OUT: 3,
                Opcode.MOV: 4,
                Opcode.MOVI: 5,
                Opcode.LD: 6,
                Opcode.ST: 7,
                Opcode.INC: 8,
                Opcode.DEC: 9,
                Opcode.NEG: 10,
                Opcode.ADD: 11,
                Opcode.SUB: 12,
                Opcode.DIV: 13,
                Opcode.MOD: 14,
                Opcode.AND: 15,
                Opcode.OR: 16,
                Opcode.NOT: 17,
                Opcode.JG: 18,
                Opcode.JE: 19,
                Opcode.JUMP: 20,
            }[opcode]
        except KeyError:
            raise HLTError() from None

    @staticmethod
    def parse_instruction_args(instr: Instruction):
        regs = []
        imm = None
        for a in instr.args:
            if is_register(a):
                regs.append(Register(a))
            else:
                imm = int(a)
        match instr.opcode:
            case Opcode.JE | Opcode.JG:
                return {
                    "sr1": regs[0] if len(regs) > 0 else None,
                    "sr2": regs[1] if len(regs) > 1 else None,
                    "imm": imm,
                }
            case _:
                return {
                    "dr": regs[0] if len(regs) > 0 else None,
                    "sr1": regs[1] if len(regs) > 1 else None,
                    "sr2": regs[2] if len(regs) > 2 else None,
                    "imm": imm,
                }

    def __init__(self, dp: DataPath):
        self.data_path = dp
        self.pc: int = 0
        self.mpc: int = 0
        self.prev_mpc: int = 0
        self.tick_counter = 0

    def on_signal_latch_pc(self, microcode: tuple):
        if Signal.PCJumpTypeNext in microcode:
            self.pc += 1
        elif Signal.PCJumpTypeJump in microcode:
            self.pc = self.parse_instruction_args(self.data_path.code_mem[self.pc])["imm"]
        elif Signal.PCJumpTypeJE in microcode:
            self.pc = (
                self.parse_instruction_args(self.data_path.code_mem[self.pc])["imm"]
                if self.data_path.get_zero_flag()
                else self.pc + 1
            )
        elif Signal.PCJumpTypeJG in microcode:
            self.pc = (
                self.parse_instruction_args(self.data_path.code_mem[self.pc])["imm"]
                if self.data_path.get_negative_flag()
                else self.pc + 1
            )
        else:
            raise ValueError("Nothing chosen on PC jump")

    def on_signal_latch_mpc(self, microcode: tuple[Signal]):
        if Signal.MicroProgramCounterNext in microcode:
            self.mpc += 1
        elif Signal.MicroProgramCounterZero in microcode:
            self.mpc = 0
        elif Signal.MicroProgramCounterOpcode in microcode:
            self.mpc = self.opcode_to_mc(self.data_path.code_mem[self.pc].opcode)
        else:
            raise ValueError("Nothing chosen on MPC jump")

    def tick(self):
        self.tick_counter += 1

    def get_ticks(self):
        return self.tick_counter

    def dispatch_micro_instruction(self, microcode: tuple):
        alu_out: int | None = None
        for signal in microcode:
            match signal:
                case Signal.LatchMPCounter:
                    self.on_signal_latch_mpc(microcode)
                case Signal.LatchPC:
                    self.on_signal_latch_pc(microcode)
                case Signal.WriteIO:
                    data_reg: Register = Register(self.data_path.code_mem[self.pc].args[1])
                    addr_reg: Register = Register(self.data_path.code_mem[self.pc].args[0])
                    self.data_path.out(data_reg, addr_reg)
                case Signal.ReadIO:
                    reg_to: Register = Register(self.data_path.code_mem[self.pc].args[0])
                    io_address = Register(self.data_path.code_mem[self.pc].args[1])
                    self.data_path.input(io_address, reg_to)
                case Signal.WriteMem:
                    data_reg: Register = Register(self.data_path.code_mem[self.pc].args[1])
                    addr_reg: Register = Register(self.data_path.code_mem[self.pc].args[0])
                    self.data_path.st(data_reg, addr_reg)
                case Signal.ReadMem:
                    reg_to: Register = Register(self.data_path.code_mem[self.pc].args[0])
                    memory_address = Register(self.data_path.code_mem[self.pc].args[1])
                    self.data_path.ld(memory_address, reg_to)
                case (
                    Signal.SumALU
                    | Signal.SubALU
                    | Signal.NegALU
                    | Signal.DivALU
                    | Signal.ModALU
                    | Signal.AndALU
                    | Signal.OrALU
                    | Signal.NotALU
                    | Signal.CmpALU
                ):
                    if Signal.SelectSourceRegisterRight in microcode:
                        alu_right = self.data_path.load_reg(
                            self.parse_instruction_args(self.data_path.code_mem[self.pc])["sr2"]
                        )
                    elif Signal.IncRight in microcode:
                        alu_right = 1
                    elif Signal.DecRight in microcode:
                        alu_right = -1
                    elif Signal.ZeroRight in microcode:
                        alu_right = 0
                    else:
                        raise ValueError("Nothing chosen on right alu input")
                    alu_out = self.data_path.alu.process(
                        signal,
                        self.data_path.load_reg(self.parse_instruction_args(self.data_path.code_mem[self.pc])["sr1"]),
                        alu_right,
                    )
                case Signal.ReadImmidate:
                    imm = self.parse_instruction_args(self.data_path.code_mem[self.pc])["imm"]
                    reg_to: Register = Register(self.data_path.code_mem[self.pc].args[0])
                    self.data_path.latch_reg(reg_to, imm)
                case Signal.ReadALU:
                    reg_to: Register = Register(self.data_path.code_mem[self.pc].args[0])
                    assert alu_out is not None, "ALU out is None"
                    self.data_path.latch_reg(reg_to, alu_out)
                case _:
                    pass

    def simulation(self):
        try:
            while True:
                if self.mpc == 0:
                    logging.debug("%s", "NEW INSTRUCTION")
                    logging.debug("%s", "------------------")
                self.prev_mpc = self.mpc
                self.dispatch_micro_instruction(self.m_program[self.mpc])
                self.tick()
                logging.debug("%s", self)  # ?
        except HLTError:
            logging.debug("Program has ended with halt")
            return
        except EOFError:
            logging.exception("Input buffer is empty!")
            return

    def __repr__(self):
        regs_str = ""
        for reg, value in self.data_path.regs.items():
            regs_str += f"({reg} = {value})"
        state_repr = (
            f"TICK {self.tick_counter} PREV_MPC: {self.prev_mpc} CUR_MPC: {self.mpc} \n REGS: [{regs_str.strip()}] \n"
        )
        instr = self.data_path.code_mem[self.pc]
        opcode = instr.opcode
        instr_repr = str(opcode)
        instr_repr += f" {instr.args}"
        if COLORED:
            return f"\033[0;32m{state_repr} \033[0;33m{instr_repr}\033[0m"
        return f"{state_repr} {instr_repr}"


def computer(code_dictionary, tokens: list[int]):
    data_mem: dict[int, int] = code_dictionary["data_mem"]
    start: int = code_dictionary["start"]
    tmp_code_mem = code_dictionary["code_mem"]
    code_mem: dict[int, Instruction] = {}
    for k, v in tmp_code_mem.items():
        code_mem[int(k)] = Instruction(Opcode(v["opcode"]), v["args"])

    pm_io = {0: IO(tokens), 1: IO([])}
    dp = DataPath(start, code_mem, data_mem, pm_io)
    cu = ControlUnit(dp)
    logging.debug("%s", cu)
    cu.simulation()

    for addr, io in pm_io.items():
        if len(io.output) != 0:
            print(f"PMIO on {addr} : '{io.output}'")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format="%(levelname)-7s %(module)s:%(funcName)-13s %(message)s")
    if len(sys.argv) == 2:
        code_dict = json.load(open(sys.argv[1], encoding="utf-8"))
        computer(code_dict, [0])
    elif len(sys.argv) == 3:
        code_dict = json.load(open(sys.argv[1], encoding="utf-8"))
        with open(sys.argv[2], encoding="utf-8") as file:
            input_token = [ord(i) for i in file.read()]
            input_token.append(0)
        computer(code_dict, input_token)
    else:
        raise Exception("Wrong arguments: machine.py <code_file> <optional_input_file>")
