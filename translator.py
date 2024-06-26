#!/usr/bin/env python3

from __future__ import annotations

import json
import sys

from isa import Instruction, Opcode
from preprocessor import preprocessing
from util import is_register, to_hex


def transform_data_into_structure(data: str):
    data_mem = dict()
    variables: dict[str, int] = {}
    address_counter = 0

    for line in data.split("\n"):
        line = line.strip()
        var_description = line.split(":")
        assert len(var_description) == 2, "Incorrect assignment"
        name, value = var_description[0].strip(), var_description[1].strip()
        assert name[-1] != " ", 'Spaces after variable name are not allowed, add ":"'
        assert name not in variables, "Variable already defined."

        if value[0] == '"':
            processed_value = bytes(value[1:-1], "utf-8").decode("unicode_escape")
            ascii_values = [len(processed_value)]
            ascii_values += [ord(char) for char in processed_value]
            variables[name] = address_counter
            for char in ascii_values:
                data_mem[int(address_counter)] = int(char)
                address_counter += 1
        else:
            variables[name] = address_counter
            if value.isdigit() or (value[0] == "-" and value[1:].isdigit()):
                data_mem[address_counter] = int(value)
            else:
                data_mem[address_counter] = to_hex(value)
            address_counter += 1

    return data_mem, variables


def transform_text_into_structure( text: str, variables: dict[str, int]):
    assert text.find(".start:") != -1, ".start label not found"
    labels: dict[str, int] = {}
    start_address: int = -1
    address_counter: int = 0
    command_mem: dict[int, Instruction] = {}

    # Label handling
    for instr in text.split("\n"):
        decoding = instr.split(" ")
        if decoding[0][0] == ".":
            current_label = decoding[0]
            assert len(decoding) == 1, "Error parsing label."
            assert current_label[-1] == ":", 'Label format error, excepted ":" mark'
            assert current_label.find(":"), 'Label format error, excepted ":" mark'
            current_label = decoding[0][1:-1]
            if current_label == "start":
                start_address = address_counter
            labels[current_label] = address_counter
        else:
            address_counter += 1

    address_counter = 0

    # Instruction handling
    for num, instr in enumerate(text.split("\n")):
        try:
            decoding = instr.split(" ")
            if decoding[0][0] != ".":
                assert Opcode(decoding[0].lower()) is not None, "No such opcode"
                cur_opcode = Opcode(decoding[0].lower())
                current_instruction = None
                command_arguments = decoding[1:]
                if cur_opcode == Opcode.IN:
                    assert len(command_arguments) == 2, "in must have two arguments - register and port address"
                    assert is_register(command_arguments[0]), "in first argument should be register"
                    assert is_register(command_arguments[1]), "in second argument should be register"
                    current_instruction = Instruction(cur_opcode, command_arguments)

                if cur_opcode == Opcode.OUT:
                    assert len(command_arguments) == 2, "out must have two arguments - register and port address"
                    assert is_register(command_arguments[0]), "out first argument should be register"
                    assert is_register(command_arguments[1]), "out second argument should be register"
                    current_instruction = Instruction(cur_opcode, command_arguments)

                if cur_opcode in [Opcode.HLT, Opcode.NOP]:
                    assert len(command_arguments) == 0, "hlt/nop should not have arguments"
                    current_instruction = Instruction(cur_opcode)

                if cur_opcode in [Opcode.JE, Opcode.JG]:
                    assert len(command_arguments) == 2, "condotional branch instruction should have 2 arguments - register and label"
                    assert is_register(command_arguments[0]), "conditional branch first argument should be register"
                    assert labels[command_arguments[1][1:]] is not None, "No such label"
                    current_instruction = Instruction(cur_opcode, [command_arguments[0], str(labels[command_arguments[1][1:]])])

                if cur_opcode in [Opcode.JUMP]:
                    assert len(command_arguments) == 1, "branch instruction should have 1 argument - label"
                    assert labels[command_arguments[0][1:]] is not None, "No such label"
                    current_instruction = Instruction(cur_opcode, [str(labels[command_arguments[0][1:]])])

                if cur_opcode == Opcode.MOV:
                    assert len(command_arguments) == 2, "mov should have arguments two arguments"
                    assert is_register(command_arguments[0]), "mov first argument should be register"
                    if is_register(command_arguments[1]):
                        current_instruction = Instruction(cur_opcode, command_arguments)
                    else:
                        raise Exception("mov second argument should be register")

                if cur_opcode == Opcode.MOVI:
                    assert len(command_arguments) == 2, "movi should have arguments two arguments"
                    assert is_register(command_arguments[0]), "movi first argument should be register"
                    if command_arguments[1].isdigit() or (command_arguments[1][0] == "-" and command_arguments[1][1:].isdigit()):
                        current_instruction = Instruction(cur_opcode, command_arguments)
                    elif command_arguments[1][0] == "0" and command_arguments[1][1] == "x":
                        current_instruction = Instruction(cur_opcode, command_arguments)
                    elif variables[command_arguments[1]] is not None:
                        current_instruction = Instruction(
                            cur_opcode, [command_arguments[0], str(variables[command_arguments[1]])]
                        )
                    else:
                        raise Exception("movi second argument can be: int, variable")

                if cur_opcode in [Opcode.INC, Opcode.DEC, Opcode.NEG]:
                    assert len(command_arguments) == 1, "inc/dec/neg must have only one argument - register"
                    assert is_register(command_arguments[0]), "inc/dec/neg first argument should be register"
                    command_arguments.append(command_arguments[0])
                    current_instruction = Instruction(cur_opcode, command_arguments)

                if cur_opcode in [Opcode.ADD, Opcode.MOD, Opcode.DIV, Opcode.SUB]:
                    assert len(command_arguments) == 3, "add/mod/div/sub should have three registers as args"
                    assert is_register(command_arguments[0]), "add/mod/div/sub args are registers"
                    assert is_register(command_arguments[1]), "add/mod/div/sub args are registers"
                    assert is_register(command_arguments[2]), "add/mod/div/sub args are registers"
                    command_arguments = [command_arguments[0], command_arguments[1], command_arguments[2]]
                    current_instruction = Instruction(cur_opcode, command_arguments)

                if cur_opcode == Opcode.LD:
                    assert len(command_arguments) == 2, "ld must have 2 arguments"
                    assert is_register(command_arguments[0]), "Not registers in arguments"
                    if is_register(command_arguments[1]):
                        current_instruction = Instruction(cur_opcode, [command_arguments[0], command_arguments[1]])
                    else:
                        current_instruction = Instruction(cur_opcode, [command_arguments[0], command_arguments[1]])

                if cur_opcode == Opcode.ST:
                    assert len(command_arguments) == 2, "st must have 2 arguments"
                    assert is_register(command_arguments[0]), "Not registers in arguments"
                    assert is_register(command_arguments[1]), "Not registers in arguments"
                    current_instruction = Instruction(cur_opcode, [command_arguments[0], command_arguments[1]])

                if cur_opcode == Opcode.NOT:
                    assert len(command_arguments) == 2, "not must have 2 arguments"
                    assert is_register(command_arguments[0]), "Not registers in arguments"
                    assert is_register(command_arguments[1]), "Not registers in arguments"
                    current_instruction = Instruction(cur_opcode, [command_arguments[0], command_arguments[1]])

                assert current_instruction is not None, "Instruction parsing error"
                command_mem[address_counter] = current_instruction
                address_counter += 1
        except AssertionError as e:
            print(f"Error in line {num + 1}: {e}")
            print(f"Line: {instr}")
            raise

    return start_address, command_mem


def perform_translator(source: str, target: str) -> dict:
    code = preprocessing(source)
    data_mem: dict[int, int] = {}
    variables = {}

    text_index = code.find("section .code")

    assert text_index != -1, "No .code section"
    text_start, text_stop = text_index + len("section .code") + 1, None
    data_index = code.find("section .data")
    if data_index == -1:
        text_stop = len(code)
    else:
        data_start, data_stop = data_index + len("section .data") + 1, None
        if data_index < text_index:
            data_stop = text_index - 1
            text_stop = len(code)
        else:
            text_stop = data_index - 1
            data_stop = len(code)
        data_mem, variables = transform_data_into_structure(code[data_start:data_stop])

    start, command_mem = transform_text_into_structure(code[text_start:text_stop], variables)
    ans = dict({"data_mem": data_mem, "start": start, "code_mem": command_mem})
    with open(target, "w", encoding="utf-8") as out_file:
        json.dump(ans, out_file, indent=4, default=lambda o: o.__dict__)

    return ans


def main(args):
    assert len(args) == 2, "Usage: translation.py <source> <output>"
    source = args[0]

    with open(source, encoding="utf-8") as file:
        code = file.read()

    result = perform_translator(code, args[1])

    loc = len(code.split("\n"))
    print(f"source LoC: {loc} instr: {len(result['code_mem'])}")


if __name__ == "__main__":
    sys.path.append("")
    main(sys.argv[1:])

