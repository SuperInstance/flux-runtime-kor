"""
어셈블리 → 바이트코드 인코더 — 한국어 기능 내장 인코딩

한국어 어셈블리 니모닉을 FLUX 바이트코드로 인코딩하는 모듈.
한국어 레지스터 이름, 한자어 숫자, 경어 수준 접두사를 지원.

지원 옵코드:
  제어:  NOP, JMP, JZ, JNZ, CALL, RET, HALT
  산술:  IADD, ISUB, IMUL, IDIV, IMOD, INEG, INC, DEC
  이동:  MOV, LOAD, STORE, MOVI, PUSH, POP
  비교:  CMP, JE, JNE
  입출력: PRINT
  권한:  CAP_REQUIRE
  통신:  TELL, ASK, DELEGATE, BROADCAST
  신뢰:  TRUST_CHECK

한국어 기능:
  - 레지스터 이름: 영(0) 일(1) 이(2) 삼(3) 사(4) 오(5) 육(6) 칠(7) 또는 R0~R63
  - 한자어 숫자: 영일이삼사오육칠팔구십백천만
  - 경어 수준: 각 명령어 블록에 CAP_REQUIRE 접두사 삽입
  - 어셈블리 니모닉: 한국어명과 영문명 모두 지원
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
# FLUX 확장 옵코드
# ═══════════════════════════════════════════════════════════════

class FluxOpcode(IntEnum):
    """FLUX 바이트코드 옵코드 — 확장 집합

    기존 vm.Opcode와 호환되면서 새로운 연산을 추가한 옵코드 집합.
    인코더는 이 열거형을 사용하여 바이트코드를 생성.
    """
    # ── 제어 흐름 ──
    NOP         = 0x00
    JMP         = 0x01
    JZ          = 0x02
    JNZ         = 0x03
    CALL        = 0x04
    RET         = 0x05
    HALT        = 0x06

    # ── 정수 산술 ──
    IADD        = 0x10    # IADD dst, a, b       — 정수 덧셈
    ISUB        = 0x11    # ISUB dst, a, b       — 정수 뺄셈
    IMUL        = 0x12    # IMUL dst, a, b       — 정수 곱셈
    IDIV        = 0x13    # IDIV dst, a, b       — 정수 나눗셈
    IMOD        = 0x14    # IMOD dst, a, b       — 정수 나머지
    INEG        = 0x15    # INEG dst, a          — 부정 (음수화)
    INC         = 0x16    # INC dst              — 1 증가
    DEC         = 0x17    # DEC dst              — 1 감소

    # ── 데이터 이동 ──
    MOV         = 0x20    # MOV dst, src          — 레지스터 간 이동
    LOAD        = 0x21    # LOAD dst, addr        — 메모리에서 로드
    STORE       = 0x22    # STORE addr, src       — 메모리에 저장
    MOVI        = 0x23    # MOVI dst, imm         — 즉시값 로드
    PUSH        = 0x24    # PUSH src              — 스택에 푸시
    POP         = 0x25    # POP dst               — 스택에서 팝

    # ── 비교 ──
    CMP         = 0x30    # CMP a, b              — 비교 (R0에 플래그)
    JE          = 0x31    # JE addr               — 같으면 점프
    JNE         = 0x32    # JNE addr              — 다르면 점프

    # ── 입출력 ──
    PRINT       = 0x40    # PRINT val             — 출력

    # ── 권한 / 경어 ──
    CAP_REQUIRE = 0x50    # CAP_REQUIRE level     — 권한 수준 요구

    # ── 에이전트 통신 ──
    TELL        = 0x60    # TELL dest, msg        — 단방향 전송
    ASK         = 0x61    # ASK dest, msg         — 질문 (요청-응답)
    DELEGATE    = 0x62    # DELEGATE dest, task   — 작업 위임
    BROADCAST   = 0x63    # BROADCAST msg         — 전체 방송

    # ── 신뢰 / 보안 ──
    TRUST_CHECK = 0x70    # TRUST_CHECK agent     — 에이전트 신뢰도 확인


# ═══════════════════════════════════════════════════════════════
# 한국어 레지스터 이름
# ═══════════════════════════════════════════════════════════════

# 한자어 레지스터 이름 → 인덱스 매핑 (기본 8개)
KOREAN_REGISTER_NAMES: dict[str, int] = {
    "영": 0, "일": 1, "이": 2, "삼": 3,
    "사": 4, "오": 5, "육": 6, "칠": 7,
    "팔": 8, "구": 9,
}

# 한자어 숫자 기호
HANJA_DIGITS: dict[str, int] = {
    "영": 0, "공": 0,
    "일": 1, "이": 2, "삼": 3, "사": 4,
    "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
}

# 한자어 단위
HANJA_UNITS: dict[str, int] = {
    "십": 10,
    "백": 100,
    "천": 1000,
    "만": 10000,
    "억": 100_000_000,
}


# ═══════════════════════════════════════════════════════════════
# 한자어 숫자 파서
# ═══════════════════════════════════════════════════════════════

def parse_korean_number(text: str) -> int:
    """한자어 숫자를 정수로 변환

    한국어 한자어 숫자 체계를 파싱:
      기본: 영(0) ~ 구(9)
      단위: 십(10), 백(100), 천(1000), 만(10000), 억(100000000)

    파싱 규칙:
      1. "삼십오" → 35  (3 × 10 + 5)
      2. "이백" → 200   (2 × 100)
      3. "만" → 10000
      4. "천만" → 10,000,000
      5. 아라비아 숫자도 그대로 지원

    Args:
        text: 한자어 또는 아라비아 숫자 문자열

    Returns:
        변환된 정수값

    Raises:
        EncodeError: 해석 불가능한 숫자인 경우
    """
    text = text.strip()

    # 빈 문자열
    if not text:
        raise EncodeError("빈 숫자 표현입니다")

    # 부호 처리
    negative = False
    if text.startswith("-"):
        negative = True
        text = text[1:]
    elif text.startswith("+"):
        text = text[1:]

    # 아라비아 숫자 (가장 빠른 경로)
    if re.fullmatch(r'[0-9]+', text):
        value = int(text)
        return -value if negative else value

    # 한자어 숫자 파싱
    result = 0
    i = 0
    current = 0

    while i < len(text):
        ch = text[i]

        # 기본 숫자
        if ch in HANJA_DIGITS:
            current = HANJA_DIGITS[ch]
            i += 1
            # 뒤에 단위가 없으면 바로 더함
            if i >= len(text) or text[i] not in HANJA_UNITS:
                result += current
                current = 0

        # 단위
        elif ch in HANJA_UNITS:
            unit_val = HANJA_UNITS[ch]
            if current == 0:
                current = 1  # "십" = 10 (암묵적 1)

            # 만 이상의 대단위는 result에 누적된 값을 곱함
            if unit_val >= 10000:
                result = (result + current) * unit_val
                current = 0
            else:
                result += current * unit_val
                current = 0

            i += 1

        else:
            raise EncodeError(f"알 수 없는 숫자 문자: '{ch}' (전체: '{text}')")

    # 남은 current 처리
    result += current

    if negative:
        result = -result

    return result


def is_korean_number(text: str) -> bool:
    """텍스트가 한자어 또는 아라비아 숫자인지 확인"""
    text = text.strip()
    if not text:
        return False
    if re.fullmatch(r'-?[0-9]+', text):
        return True
    # 모든 문자가 한자어 숫자 문자인지 확인
    valid_chars = set(HANJA_DIGITS.keys()) | set(HANJA_UNITS.keys())
    return all(ch in valid_chars for ch in text)


# ═══════════════════════════════════════════════════════════════
# 레지스터 파서
# ═══════════════════════════════════════════════════════════════

def parse_register(text: str) -> int:
    """레지스터 이름을 인덱스로 변환

    지원 형식:
      - 한국어: 영, 일, 이, 삼, 사, 오, 육, 칠, 팔, 구
      - 영문: R0, R1, ..., R63

    Args:
        text: 레지스터 이름 문자열

    Returns:
        레지스터 인덱스 (0~63)

    Raises:
        EncodeError: 잘못된 레지스터 이름인 경우
    """
    text = text.strip()

    # 한국어 레지스터 이름
    if text in KOREAN_REGISTER_NAMES:
        return KOREAN_REGISTER_NAMES[text]

    # 영문 R 접두사 형식
    m = re.fullmatch(r'[Rr]([0-9]+)', text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx <= 63:
            return idx
        raise EncodeError(f"레지스터 범위 초과: R{idx} (R0~R63만 지원)")

    # 한자어 숫자로 해석 시도
    if is_korean_number(text):
        val = parse_korean_number(text)
        if 0 <= val <= 63:
            return val

    raise EncodeError(f"레지스터를 해석할 수 없습니다: '{text}'")


# ═══════════════════════════════════════════════════════════════
# 옵코드 → 니모닉 매핑 (한국어 + 영문)
# ═══════════════════════════════════════════════════════════════

# 한국어 니모닉 → 옵코드
KOREAN_MNEMONICS: dict[str, FluxOpcode] = {
    "무":        FluxOpcode.NOP,
    "건너뛰기":  FluxOpcode.JMP,
    "영이면":    FluxOpcode.JZ,
    "영이아니면": FluxOpcode.JNZ,
    "호출":      FluxOpcode.CALL,
    "반환":      FluxOpcode.RET,
    "정지":      FluxOpcode.HALT,
    "더하기":    FluxOpcode.IADD,
    "빼기":      FluxOpcode.ISUB,
    "곱하기":    FluxOpcode.IMUL,
    "나누기":    FluxOpcode.IDIV,
    "나머지":    FluxOpcode.IMOD,
    "부정":      FluxOpcode.INEG,
    "증가":      FluxOpcode.INC,
    "감소":      FluxOpcode.DEC,
    "이동":      FluxOpcode.MOV,
    "불러오기":  FluxOpcode.LOAD,
    "저장":      FluxOpcode.STORE,
    "대입":      FluxOpcode.MOVI,
    "밀어넣기":  FluxOpcode.PUSH,
    "꺼내기":    FluxOpcode.POP,
    "비교":      FluxOpcode.CMP,
    "같으면":    FluxOpcode.JE,
    "다르면":    FluxOpcode.JNE,
    "출력":      FluxOpcode.PRINT,
    "권한요구":  FluxOpcode.CAP_REQUIRE,
    "전달":      FluxOpcode.TELL,
    "질문":      FluxOpcode.ASK,
    "위임":      FluxOpcode.DELEGATE,
    "방송":      FluxOpcode.BROADCAST,
    "신뢰확인":  FluxOpcode.TRUST_CHECK,
}

# 영문 니모닉 → 옵코드
ENGLISH_MNEMONICS: dict[str, FluxOpcode] = {
    "NOP":         FluxOpcode.NOP,
    "JMP":         FluxOpcode.JMP,
    "JZ":          FluxOpcode.JZ,
    "JNZ":         FluxOpcode.JNZ,
    "CALL":        FluxOpcode.CALL,
    "RET":         FluxOpcode.RET,
    "HALT":        FluxOpcode.HALT,
    "IADD":        FluxOpcode.IADD,
    "ISUB":        FluxOpcode.ISUB,
    "IMUL":        FluxOpcode.IMUL,
    "IDIV":        FluxOpcode.IDIV,
    "IMOD":        FluxOpcode.IMOD,
    "INEG":        FluxOpcode.INEG,
    "INC":         FluxOpcode.INC,
    "DEC":         FluxOpcode.DEC,
    "MOV":         FluxOpcode.MOV,
    "LOAD":        FluxOpcode.LOAD,
    "STORE":       FluxOpcode.STORE,
    "MOVI":        FluxOpcode.MOVI,
    "PUSH":        FluxOpcode.PUSH,
    "POP":         FluxOpcode.POP,
    "CMP":         FluxOpcode.CMP,
    "JE":          FluxOpcode.JE,
    "JNE":         FluxOpcode.JNE,
    "PRINT":       FluxOpcode.PRINT,
    "CAP_REQUIRE": FluxOpcode.CAP_REQUIRE,
    "TELL":        FluxOpcode.TELL,
    "ASK":         FluxOpcode.ASK,
    "DELEGATE":    FluxOpcode.DELEGATE,
    "BROADCAST":   FluxOpcode.BROADCAST,
    "TRUST_CHECK": FluxOpcode.TRUST_CHECK,
}

# 통합 니모닉 테이블 (영문이 우선)
ALL_MNEMONICS: dict[str, FluxOpcode] = {**KOREAN_MNEMONICS, **ENGLISH_MNEMONICS}

# 옵코드 → 표준 영문 니모닉 (역방향, 디스어셈블용)
OPCODE_TO_MNEMONIC: dict[FluxOpcode, str] = {
    op: name for name, op in ENGLISH_MNEMONICS.items()
}

# 옵코드 → 한국어 니모닉 (디스어셈블용)
OPCODE_TO_KOREAN: dict[FluxOpcode, str] = {
    op: name for name, op in KOREAN_MNEMONICS.items()
}


# ═══════════════════════════════════════════════════════════════
# 경어 수준
# ═══════════════════════════════════════════════════════════════

class EncodeHonorificLevel(IntEnum):
    """인코더용 경어 수준 — 바이트코드에 삽입할 CAP_REQUIRE 값"""
    해라체 = 1      # 평어 → system/internal
    해체 = 2        # 반말 → peer
    해요체 = 3      # 존댓말 → standard user
    하십시오체 = 4  # 최경어 → admin

HONORIFIC_PREFIX: dict[EncodeHonorificLevel, str] = {
    EncodeHonorificLevel.해라체:     "#경어:해라체",
    EncodeHonorificLevel.해체:       "#경어:해체",
    EncodeHonorificLevel.해요체:     "#경어:해요체",
    EncodeHonorificLevel.하십시오체: "#경어:하십시오체",
}


# ═══════════════════════════════════════════════════════════════
# 옵코드 정의 (피연산자 수)
# ═══════════════════════════════════════════════════════════════

# 옵코드별 필요 피연산자 수 (인코딩 검증용)
OPCODE_ARITY: dict[FluxOpcode, int] = {
    FluxOpcode.NOP:         0,
    FluxOpcode.JMP:         1,    # 주소
    FluxOpcode.JZ:          1,    # 주소
    FluxOpcode.JNZ:         1,    # 주소
    FluxOpcode.CALL:        1,    # 주소
    FluxOpcode.RET:         0,
    FluxOpcode.HALT:        0,
    FluxOpcode.IADD:        3,    # dst, a, b
    FluxOpcode.ISUB:        3,    # dst, a, b
    FluxOpcode.IMUL:        3,    # dst, a, b
    FluxOpcode.IDIV:        3,    # dst, a, b
    FluxOpcode.IMOD:        3,    # dst, a, b
    FluxOpcode.INEG:        2,    # dst, a
    FluxOpcode.INC:         1,    # dst
    FluxOpcode.DEC:         1,    # dst
    FluxOpcode.MOV:         2,    # dst, src
    FluxOpcode.LOAD:        2,    # dst, addr
    FluxOpcode.STORE:       2,    # addr, src
    FluxOpcode.MOVI:        2,    # dst, imm
    FluxOpcode.PUSH:        1,    # src
    FluxOpcode.POP:         1,    # dst
    FluxOpcode.CMP:         2,    # a, b
    FluxOpcode.JE:          1,    # 주소
    FluxOpcode.JNE:         1,    # 주소
    FluxOpcode.PRINT:       1,    # val
    FluxOpcode.CAP_REQUIRE: 1,    # level
    FluxOpcode.TELL:        2,    # dest, msg
    FluxOpcode.ASK:         2,    # dest, msg
    FluxOpcode.DELEGATE:    2,    # dest, task
    FluxOpcode.BROADCAST:   1,    # msg
    FluxOpcode.TRUST_CHECK: 1,    # agent
}


# ═══════════════════════════════════════════════════════════════
# 인코딩 오류
# ═══════════════════════════════════════════════════════════════

class EncodeError(Exception):
    """인코딩 오류"""
    pass


# ═══════════════════════════════════════════════════════════════
# 레이블
# ═══════════════════════════════════════════════════════════════

@dataclass
class Label:
    """어셈블리 레이블 — 점프 대상 위치

    Attributes:
        name: 레이블 이름
        address: 바이트코드 내 주소 (인코딩 후 결정)
    """
    name: str
    address: int = -1


# ═══════════════════════════════════════════════════════════════
# 명령어 토큰
# ═══════════════════════════════════════════════════════════════

@dataclass
class Instruction:
    """파싱된 명령어 — 옵코드와 피연산자

    Attributes:
        opcode: FLUX 옵코드
        operands: 피연산자 목록 (문자열, 나중에 타입 변환)
        line_num: 소스 줄 번호
        raw: 원시 소스 텍스트
    """
    opcode: FluxOpcode
    operands: list[str]
    line_num: int
    raw: str


# ═══════════════════════════════════════════════════════════════
# FLUX 어셈블리 인코더
# ═══════════════════════════════════════════════════════════════

class FluxEncoder:
    """FLUX 어셈블리 → 바이트코드 인코더

    한국어 니모닉, 한자어 레지스터, 한자어 숫자, 경어 수준 지시자를
    지원하는 2-pass 인코더.

    Pass 1: 레이블 수집 및 주소 계산
    Pass 2: 레이블 참조 해결 및 바이트코드 생성

    사용 예시::

        encoder = FluxEncoder()
        encoder.add_honorific_level(EncodeHonorificLevel.하십시오체)
        bytecode = encoder.encode(\"\"\"
            #경어:하십시오체
            대입 영 42          ; MOVI R0, 42
            대입 일 58          ; MOVI R1, 58
            더하기 이 영 일     ; IADD R2, R0, R1
            출력 이             ; PRINT R2
            정지
        \"\"\")
    """

    def __init__(self) -> None:
        self._instructions: list[Instruction] = []
        self._labels: dict[str, Label] = {}
        self._current_honorific: EncodeHonorificLevel = EncodeHonorificLevel.해라체
        self._errors: list[str] = []

    # ── 경어 수준 관리 ──

    def set_honorific_level(self, level: EncodeHonorificLevel) -> None:
        """경어 수준 설정

        이후에 추가되는 모든 명령어 앞에 CAP_REQUIRE 접두사가 삽입됨.

        Args:
            level: 경어 수준
        """
        self._current_honorific = level

    def get_honorific_level(self) -> EncodeHonorificLevel:
        """현재 경어 수준 반환"""
        return self._current_honorific

    # ── 소스 파싱 ──

    def _parse_line(self, line: str, line_num: int) -> Optional[Instruction | Label]:
        """단일 어셈블리 줄을 파싱

        Args:
            line: 어셈블리 소스 줄
            line_num: 줄 번호

        Returns:
            Instruction 또는 Label (주석/빈 줄은 None)
        """
        # 주석 제거 (; 또는 #)
        stripped = self._strip_comment(line).strip()
        if not stripped:
            return None

        # 경어 수준 지시자
        if stripped.startswith("#경어:"):
            level_str = stripped[4:].strip()
            level_map = {
                "해라체": EncodeHonorificLevel.해라체,
                "해체": EncodeHonorificLevel.해체,
                "해요체": EncodeHonorificLevel.해요체,
                "하십시오체": EncodeHonorificLevel.하십시오체,
            }
            if level_str in level_map:
                self._current_honorific = level_map[level_str]
            return None

        # 레이블 정의 (이름 뒤에 콜론)
        if stripped.endswith(":"):
            label_name = stripped[:-1].strip()
            if not label_name:
                self._errors.append(f"줄 {line_num}: 빈 레이블 이름")
                return None
            return Label(name=label_name)

        # 명령어 파싱
        parts = stripped.split(None, 1)  # 니모닉과 나머지 분리
        mnemonic = parts[0].upper() if re.fullmatch(r'[A-Za-z_]+', parts[0]) else parts[0]

        # 옵코드 조회
        opcode = ALL_MNEMONICS.get(mnemonic)
        if opcode is None:
            # 대소문자 무시 재시도
            opcode = ALL_MNEMONICS.get(mnemonic.upper())
        if opcode is None:
            self._errors.append(
                f"줄 {line_num}: 알 수 없는 니모닉 '{mnemonic}'"
            )
            return None

        # 피연산자 분리
        operands: list[str] = []
        if len(parts) > 1:
            operand_str = parts[1].strip()
            # 쉼표 또는 공백으로 분리
            operands = [op.strip() for op in re.split(r'[,，\s]+', operand_str) if op.strip()]

        return Instruction(
            opcode=opcode,
            operands=operands,
            line_num=line_num,
            raw=stripped,
        )

    @staticmethod
    def _strip_comment(line: str) -> str:
        """주석 제거 — 세미콜론(;) 또는 해시(#) 이후 텍스트 제거

        단, 경어 지시자(#경어:)는 보존.
        """
        # #경어: 지시자 보존
        if line.strip().startswith("#경어:"):
            return line

        # 세미콜론 주석
        for i, ch in enumerate(line):
            if ch == ';':
                return line[:i]

        # 해시 주석 (줄 시작이 아닌 경우만)
        for i, ch in enumerate(line):
            if ch == '#' and i > 0 and line[i-1] != ' ':
                return line[:i]
            elif ch == '#' and i == 0:
                return ""

        return line

    # ── 명령어에 CAP_REQUIRE 자동 삽입 ──

    def _insert_cap_require(self, instruction: Instruction) -> list[Instruction]:
        """명령어에 경어 수준 CAP_REQUIRE를 접두사로 삽입

        이미 CAP_REQUIRE인 경우 삽입하지 않음.
        NOP, HALT 등에는 삽입하지 않음.

        Args:
            instruction: 원본 명령어

        Returns:
            명령어 목록 (CAP_REQUIRE + 원본)
        """
        # 특정 옵코드에는 삽입하지 않음
        skip_opcodes = {
            FluxOpcode.NOP,
            FluxOpcode.HALT,
            FluxOpcode.CAP_REQUIRE,
            FluxOpcode.RET,
        }

        if instruction.opcode in skip_opcodes:
            return [instruction]

        cap = Instruction(
            opcode=FluxOpcode.CAP_REQUIRE,
            operands=[str(self._current_honorific.value)],
            line_num=instruction.line_num,
            raw=f"#자동:{HONORIFIC_PREFIX[self._current_honorific]}",
        )

        return [cap, instruction]

    # ── 피연산자 변환 ──

    def _resolve_operand(self, operand: str, expect_register: bool = False) -> Any:
        """피연산자 문자열을 적절한 타입으로 변환

        Args:
            operand: 피연산자 문자열
            expect_register: 레지스터를 기대하는지 여부

        Returns:
            정수 (레지스터 인덱스 또는 상수) 또는 문자열 (레이블/메시지)
        """
        operand = operand.strip()

        # 레이블 참조 (문자열)
        if not operand.lstrip('-').isdigit() and not is_korean_number(operand.replace("-", "")):
            if re.fullmatch(r'[A-Za-z_가-힣][A-Za-z0-9_가-힣]*', operand):
                # 레이블로 등록되어 있으면 주소 반환
                if operand in self._labels:
                    return self._labels[operand].address
                return operand  # 나중에 해결

        # 레지스터 파싱
        if expect_register or operand.startswith(("R", "r")):
            try:
                return parse_register(operand)
            except EncodeError:
                pass

        # 한자어 숫자 또는 아라비아 숫자
        if is_korean_number(operand):
            return parse_korean_number(operand)

        # 문자열 리터럴 (따옴표)
        if operand.startswith('"') and operand.endswith('"'):
            return operand[1:-1]

        # 그대로 반환 (문자열)
        return operand

    # ── 인코딩 ──

    def encode(self, source: str, auto_cap: bool = True) -> list[tuple]:
        """어셈블리 소스를 바이트코드로 인코딩

        2-pass 인코딩:
          Pass 1: 레이블 수집 및 주소 계산
          Pass 2: 레이블 참조 해결 및 바이트코드 생성

        Args:
            source: 어셈블리 소스 코드
            auto_cap: 경어 수준 CAP_REQUIRE 자동 삽입 여부

        Returns:
            바이트코드 명령어 목록 (tuple)

        Raises:
            EncodeError: 인코딩 오류 발생 시
        """
        self._errors.clear()
        self._labels.clear()
        self._instructions.clear()

        lines = source.strip().split("\n")

        # ── Pass 1: 파싱 및 레이블 수집 ──
        parsed_items: list[Instruction | Label] = []
        for i, line in enumerate(lines, start=1):
            item = self._parse_line(line, i)
            if item is not None:
                parsed_items.append(item)

        # 명령어 수집 (레이블 제외) + CAP_REQUIRE 삽입
        for item in parsed_items:
            if isinstance(item, Label):
                continue
            if auto_cap:
                expanded = self._insert_cap_require(item)
                self._instructions.extend(expanded)
            else:
                self._instructions.append(item)

        # 레이블 주소 계산
        addr = 0
        for item in parsed_items:
            if isinstance(item, Label):
                item.address = addr
                self._labels[item.name] = item
            else:
                addr += 1  # 각 명령어 = 1 바이트코드 슬롯

        # CAP_REQUIRE 삽입으로 인한 주소 재계산
        addr = 0
        for instr in self._instructions:
            if instr.opcode == FluxOpcode.CAP_REQUIRE and instr.raw.startswith("#자동:"):
                # 자동 삽입된 CAP은 이미 주소에 포함됨
                pass
            addr += 1

        # 레이블 주소 재계산 (CAP 포함)
        addr = 0
        for item in parsed_items:
            if isinstance(item, Label):
                item.address = addr
                self._labels[item.name] = item
            else:
                if auto_cap:
                    addr += 2  # 명령어 + CAP_REQUIRE
                else:
                    addr += 1

        # 오류 확인
        if self._errors:
            raise EncodeError("\n".join(self._errors))

        # ── Pass 2: 바이트코드 생성 ──
        bytecode: list[tuple] = []

        for instr in self._instructions:
            opcode = instr.opcode
            arity = OPCODE_ARITY.get(opcode, 0)

            # 피연산자 수 검증
            if len(instr.operands) < arity:
                self._errors.append(
                    f"줄 {instr.line_num}: {OPCODE_TO_MNEMONIC.get(opcode, '?')} "
                    f"피연산자 부족 (필요={arity}, 실제={len(instr.operands)})"
                )
                continue

            # 피연산자 변환
            resolved: list[Any] = []
            for j, op_str in enumerate(instr.operands):
                expect_reg = j < arity and opcode not in {
                    FluxOpcode.PRINT,
                    FluxOpcode.TELL,
                    FluxOpcode.ASK,
                    FluxOpcode.DELEGATE,
                    FluxOpcode.BROADCAST,
                    FluxOpcode.TRUST_CHECK,
                    FluxOpcode.CAP_REQUIRE,
                    FluxOpcode.JMP,
                    FluxOpcode.JZ,
                    FluxOpcode.JNZ,
                    FluxOpcode.JE,
                    FluxOpcode.JNE,
                    FluxOpcode.CALL,
                }
                resolved.append(self._resolve_operand(op_str, expect_register=expect_reg))

            # 바이트코드 튜플 생성
            bytecode_tuple = tuple([opcode] + resolved[:arity])
            bytecode.append(bytecode_tuple)

        if self._errors:
            raise EncodeError("\n".join(self._errors))

        return bytecode

    # ── 디스어셈블 ──

    def disassemble(self, bytecode: list[tuple], korean: bool = False) -> str:
        """바이트코드를 어셈블리 형식으로 디스어셈블

        Args:
            bytecode: 바이트코드 목록
            korean: 한국어 니모닉 사용 여부

        Returns:
            디스어셈블된 텍스트
        """
        lines: list[str] = []
        mnemonic_map = OPCODE_TO_KOREAN if korean else OPCODE_TO_MNEMONIC

        for i, instr in enumerate(bytecode):
            opcode = instr[0]
            args = instr[1:]
            mnemonic = mnemonic_map.get(opcode, f"UNKNOWN({opcode})")

            if args:
                formatted_args = []
                for arg in args:
                    if isinstance(arg, int) and opcode in {
                        FluxOpcode.CAP_REQUIRE,
                    }:
                        level_map = {
                            1: "해라체", 2: "해체",
                            3: "해요체", 4: "하십시오체",
                        }
                        formatted_args.append(level_map.get(arg, str(arg)))
                    else:
                        formatted_args.append(str(arg))

                args_str = " ".join(formatted_args)
                lines.append(f"  [{i:04d}] {mnemonic} {args_str}")
            else:
                lines.append(f"  [{i:04d}] {mnemonic}")

        return "\n".join(lines)

    # ── 단일 명령어 인코딩 ──

    def encode_instruction(
        self,
        mnemonic: str,
        *operands: Any,
        honorific: Optional[EncodeHonorificLevel] = None,
    ) -> list[tuple]:
        """단일 명령어를 바이트코드로 인코딩

        간편 메서드 — 어셈블리 소스 없이 직접 명령어 생성.

        Args:
            mnemonic: 니모닉 (한국어 또는 영문)
            *operands: 피연산자
            honorific: 경어 수준 (지정 시 CAP_REQUIRE 접두사 삽입)

        Returns:
            바이트코드 목록

        Raises:
            EncodeError: 알 수 없는 니모닉인 경우
        """
        opcode = ALL_MNEMONICS.get(mnemonic)
        if opcode is None:
            opcode = ALL_MNEMONICS.get(mnemonic.upper())
        if opcode is None:
            raise EncodeError(f"알 수 없는 니모닉: '{mnemonic}'")

        bytecode: list[tuple] = []

        # 경어 CAP 삽입
        if honorific is not None:
            bytecode.append((FluxOpcode.CAP_REQUIRE, honorific.value))

        # 명령어 생성
        resolved_operands = []
        for op in operands:
            if isinstance(op, str):
                if is_korean_number(op):
                    resolved_operands.append(parse_korean_number(op))
                else:
                    resolved_operands.append(op)
            else:
                resolved_operands.append(op)

        bytecode.append(tuple([opcode] + resolved_operands))
        return bytecode

    # ── 유틸리티 ──

    @staticmethod
    def format_bytecode_korean(bytecode: list[tuple]) -> str:
        """바이트코드를 한국어 형식으로 포맷

        각 옵코드를 한국어 니모닉으로 표시.
        """
        lines: list[str] = ["바이트코드 (한국어):", "─" * 50]

        for i, instr in enumerate(bytecode):
            opcode = instr[0]
            args = instr[1:]
            kr_name = OPCODE_TO_KOREAN.get(opcode, f"알수없음({opcode})")

            if args:
                formatted = []
                for arg in args:
                    if isinstance(arg, int) and opcode == FluxOpcode.CAP_REQUIRE:
                        level_names = {
                            1: "해라체", 2: "해체",
                            3: "해요체", 4: "하십시오체",
                        }
                        formatted.append(f"권한:{level_names.get(arg, str(arg))}")
                    elif isinstance(arg, int):
                        # 레지스터 인덱스를 한자어로 변환
                        reg_names = {
                            0: "영", 1: "일", 2: "이", 3: "삼",
                            4: "사", 5: "오", 6: "육", 7: "칠",
                            8: "팔", 9: "구",
                        }
                        if arg in reg_names:
                            formatted.append(f"R{arg}({reg_names[arg]})")
                        else:
                            formatted.append(str(arg))
                    else:
                        formatted.append(str(arg))

                args_str = " ".join(formatted)
                lines.append(f"  [{i:04d}] {kr_name} {args_str}")
            else:
                lines.append(f"  [{i:04d}] {kr_name}")

        return "\n".join(lines)
