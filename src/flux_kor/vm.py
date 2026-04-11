"""
유체 VM — 최소 바이트코드 가상머신

Korean-first runtime의 바닥에 있는 언어 중립 실행 엔진.
경어 수준이 CAP_REQUIRE 옵코드로 권한을 검사하고,
SOV→CPS 컴파일된 바이트코드를 실행한다.

레지스터 기반 VM:
  R0-R7: 범용 레지스터
  CAP: 현재 권한 수준
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any


class Opcode(IntEnum):
    """유체 바이트코드 옵코드"""
    # 산술
    ADD = 0x01
    SUB = 0x02
    MUL = 0x03
    DIV = 0x04
    MOD = 0x05
    POW = 0x06

    # 이동/저장
    LOAD_IMM = 0x10       # LOAD_IMM reg, imm
    LOAD = 0x11           # LOAD reg, addr
    STORE = 0x12          # STORE addr, reg
    MOV = 0x13            # MOV dst, src

    # 제어 흐름
    JMP = 0x20
    JZ = 0x21             # JZ addr (jump if zero)
    JNZ = 0x22            # JNZ addr (jump if not zero)
    CMP = 0x23            # CMP reg, reg
    CALL = 0x24
    RET = 0x25
    HALT = 0x26

    # 권한/경어
    CAP_REQUIRE = 0x30    # CAP_REQUIRE level — 권한 수준 확인
    CAP_SET = 0x31        # CAP_SET level

    # 에이전트 간 통신
    SEND = 0x40           # SEND agent, message
    BROADCAST = 0x41      # BROADCAST message

    # 내장 함수
    FACTORIAL = 0x50
    SUM_RANGE = 0x51

    # 디버그
    PRINT = 0xF0
    NOP = 0x00


class VMError(Exception):
    """VM 실행 오류"""
    pass


class CapabilityError(VMError):
    """권한 부족 오류 — 경어 수준 불일치"""
    pass


class VM:
    """유체 가상머신

    경어 수준 기반 권한 시스템을 내장한 레지스터 VM.
    CAP_REQUIRE 옵코드로 현재 컨텍스트의 경어 수준을 검사한다.
    """

    NUM_REGISTERS = 8

    # 경어 수준 → 권한 값 (높을수록 높은 권한)
    HONORIFIC_CAP: dict[int, int] = {
        4: 0b1000,  # 하십시오체 (admin)
        3: 0b0100,  # 해요체 (standard user)
        2: 0b0010,  # 해체 (peer)
        1: 0b0001,  # 해라체 (internal/system)
    }

    def __init__(self) -> None:
        self.registers: list[int] = [0] * self.NUM_REGISTERS
        self.pc: int = 0
        self.code: list[tuple] = []
        self.cap_level: int = 1  # 기본: 해라체 (system)
        self.call_stack: list[int] = []
        self.halted: bool = False
        self.output: list[str] = []
        self.messages: list[tuple[str, str]] = []

    def reset(self) -> None:
        self.registers = [0] * self.NUM_REGISTERS
        self.pc = 0
        self.call_stack = []
        self.halted = False
        self.output = []
        self.messages = []

    def set_cap_level(self, level: int) -> None:
        """권한 수준 설정"""
        if level not in self.HONORIFIC_CAP:
            raise VMError(f"잘못된 권한 수준: {level}")
        self.cap_level = level

    def _check_cap(self, required: int) -> None:
        """권한 검사 — 현재 CAP이 필요한 수준을 만족하는지"""
        current_cap = self.HONORIFIC_CAP.get(self.cap_level, 0)
        required_cap = self.HONORIFIC_CAP.get(required, 0)
        if current_cap < required_cap:
            raise CapabilityError(
                f"권한 부족: 현재={self.cap_level}, 필요={required} "
                f"(CAP 0b{current_cap:04b} < 0b{required_cap:04b})"
            )

    def _reg(self, idx: int) -> int:
        if not 0 <= idx < self.NUM_REGISTERS:
            raise VMError(f"레지스터 범위 초과: R{idx}")
        return self.registers[idx]

    def _set_reg(self, idx: int, val: int) -> None:
        if not 0 <= idx < self.NUM_REGISTERS:
            raise VMError(f"레지스터 범위 초과: R{idx}")
        self.registers[idx] = val

    def _korean_numeral(self, n: int) -> str:
        """숫자를 한자어 표기로 변환"""
        numerals = {
            0: "영", 1: "일", 2: "이", 3: "삼", 4: "사",
            5: "오", 6: "육", 7: "칠", 8: "팔", 9: "구", 10: "십",
        }
        if n in numerals:
            return numerals[n]
        return str(n)

    def execute(self, bytecode: list[tuple] | None = None) -> list[str]:
        """바이트코드 실행 — 출력 라인 목록 반환"""
        if bytecode is not None:
            self.code = bytecode
        self.pc = 0
        self.halted = False

        while not self.halted and self.pc < len(self.code):
            instr = self.code[self.pc]
            op = instr[0]
            args = instr[1:]

            if op == Opcode.NOP:
                self.pc += 1

            elif op == Opcode.HALT:
                self.halted = True

            elif op == Opcode.LOAD_IMM:
                reg, imm = int(args[0]), int(args[1])
                self._set_reg(reg, imm)
                self.pc += 1

            elif op == Opcode.MOV:
                dst, src = int(args[0]), int(args[1])
                self._set_reg(dst, self._reg(src))
                self.pc += 1

            elif op == Opcode.STORE:
                addr, reg = int(args[0]), int(args[1])
                # 단순 메모리 (addr → 레지스터 맵핑으로 대체)
                self._set_reg(addr, self._reg(reg))
                self.pc += 1

            elif op == Opcode.ADD:
                dst, a, b = int(args[0]), int(args[1]), int(args[2])
                self._set_reg(dst, self._reg(a) + self._reg(b))
                self.pc += 1

            elif op == Opcode.SUB:
                dst, a, b = int(args[0]), int(args[1]), int(args[2])
                self._set_reg(dst, self._reg(a) - self._reg(b))
                self.pc += 1

            elif op == Opcode.MUL:
                dst, a, b = int(args[0]), int(args[1]), int(args[2])
                self._set_reg(dst, self._reg(a) * self._reg(b))
                self.pc += 1

            elif op == Opcode.DIV:
                dst, a, b = int(args[0]), int(args[1]), int(args[2])
                divisor = self._reg(b)
                if divisor == 0:
                    raise VMError("0으로 나눌 수 없습니다")
                self._set_reg(dst, self._reg(a) // divisor)
                self.pc += 1

            elif op == Opcode.POW:
                dst, a, b = int(args[0]), int(args[1]), int(args[2])
                self._set_reg(dst, self._reg(a) ** self._reg(b))
                self.pc += 1

            elif op == Opcode.CMP:
                a, b = int(args[0]), int(args[1])
                val_a, val_b = self._reg(a), self._reg(b)
                self._set_reg(0, 0)
                if val_a < val_b:
                    self._set_reg(0, -1)
                elif val_a > val_b:
                    self._set_reg(0, 1)
                self.pc += 1

            elif op == Opcode.JZ:
                addr = int(args[0])
                if self._reg(0) == 0:
                    self.pc = addr
                else:
                    self.pc += 1

            elif op == Opcode.JNZ:
                addr = int(args[0])
                if self._reg(0) != 0:
                    self.pc = addr
                else:
                    self.pc += 1

            elif op == Opcode.JMP:
                self.pc = int(args[0])

            elif op == Opcode.CALL:
                addr = int(args[0])
                self.call_stack.append(self.pc + 1)
                self.pc = addr

            elif op == Opcode.RET:
                if self.call_stack:
                    self.pc = self.call_stack.pop()
                else:
                    raise VMError("반환 주소가 없습니다")

            elif op == Opcode.CAP_REQUIRE:
                required_level = int(args[0])
                self._check_cap(required_level)
                self.pc += 1

            elif op == Opcode.CAP_SET:
                new_level = int(args[0])
                self.set_cap_level(new_level)
                self.pc += 1

            elif op == Opcode.FACTORIAL:
                reg = int(args[0])
                n = self._reg(reg)
                self._set_reg(reg, math.factorial(n))
                self.pc += 1

            elif op == Opcode.SUM_RANGE:
                dst, reg_a, reg_b = int(args[0]), int(args[1]), int(args[2])
                a, b = self._reg(reg_a), self._reg(reg_b)
                self._set_reg(dst, sum(range(a, b + 1)))
                self.pc += 1

            elif op == Opcode.SEND:
                agent, message = str(args[0]), str(args[1])
                self.messages.append(("send", agent, message))
                self.output.append(f"→ {agent}: {message}")
                self.pc += 1

            elif op == Opcode.BROADCAST:
                message = str(args[0])
                self.messages.append(("broadcast", "*", message))
                self.output.append(f"📢 전체: {message}")
                self.pc += 1

            elif op == Opcode.PRINT:
                val = args[0]
                self.output.append(str(val))
                self.pc += 1

            else:
                raise VMError(f"알 수 없는 옵코드: {op}")

        return self.output

    def dump_registers(self) -> str:
        """레지스터 상태 출력"""
        lines = ["레지스터 상태:"]
        for i, v in enumerate(self.registers):
            lines.append(f"  R{i}({self._korean_numeral(i)}) = {v}")
        return "\n".join(lines)
