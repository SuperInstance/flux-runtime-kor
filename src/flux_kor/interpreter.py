"""
유체 한국어 인터프리터 — SOV→CPS 컴파일

한국어 자연어를 바이트코드로 컴파일하고 VM에서 실행.
핵심 설계 원칙:
  1. SOV 어순 → continuation-passing style
     주어-목적어-동사 순서에서 동사(마지막)가 continuation이 됨
  2. 조사 → 범위 연산자
     은/는(주제), 이/가(주어), 을/를(목적어), 에/에게(방향)
  3. 활용 → 함수 합성
     동사 어미가 쌓이면 함수가 합성됨
  4. 경어 → CAP_REQUIRE 옵코드 삽입

지원 패턴:
  산술: 계산 $a 더하기 $b
        $a 곱하기 $b
        $a 부터 $b 까지 합
        $a 의 팩토리얼
  레지스터: 레지스터 영 에 $val 대입
            레지스터 영 더하기 레지스터 일
  조건: 레지스터 영 이 영이 아닌 동안 (while R0 != 0)
  통신: $agent 에게 $message 전달
         $agent 에게 $topic 질문
         전체에 $message 방송
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from flux_kor.vm import Opcode, VM, VMError
from flux_kor.honorifics import HonorificLevel, HonorificValidator


class ParseError(Exception):
    """한국어 NL 해석 오류"""
    pass


class KoreanNumeral:
    """한자어 숫자 ↔ 정수 변환

    영(0) 일(1) 이(2) 삼(3) 사(4) 오(5) 육(6) 칠(7) 팔(8) 구(9) 십(10)
    """

    NUMERALS: dict[str, int] = {
        "영": 0, "공": 0,
        "일": 1,
        "이": 2,
        "삼": 3,
        "사": 4,
        "오": 5,
        "육": 6,
        "칠": 7,
        "팔": 8,
        "구": 9,
        "십": 10,
    }

    COMPOUND_NUMERALS: dict[str, int] = {
        "십일": 11, "십이": 12, "십삼": 13, "십사": 14, "십오": 15,
        "십육": 16, "십칠": 17, "십팔": 18, "십구": 19,
        "이십": 20, "이십일": 21, "이십오": 25, "삼십": 30,
    }

    @classmethod
    def to_int(cls, text: str) -> int:
        """한자어 숫자를 정수로 변환"""
        text = text.strip()
        # 직접 매칭
        if text in cls.NUMERALS:
            return cls.NUMERALS[text]
        if text in cls.COMPOUND_NUMERALS:
            return cls.COMPOUND_NUMERALS[text]
        # 복합 숫자 (십X 형태)
        if text.endswith("십") and len(text) > 1:
            tens = cls.NUMERALS.get(text[0], 0)
            return tens * 10
        # 아라비아 숫자
        try:
            return int(text)
        except ValueError:
            pass
        raise ParseError(f"숫자를 해석할 수 없습니다: {text}")

    @classmethod
    def is_numeral(cls, text: str) -> bool:
        """텍스트가 숫자인지 확인"""
        text = text.strip()
        if text in cls.NUMERALS or text in cls.COMPOUND_NUMERALS:
            return True
        if text.endswith("십") and len(text) > 1 and text[0] in cls.NUMERALS:
            return True
        try:
            int(text)
            return True
        except ValueError:
            return False


class Particles:
    """조사(후치사) 정의 — 범위 연산자"""
    TOPIC = ["은", "는"]          # 주제 조사
    SUBJECT = ["이", "가"]        # 주어 조사
    OBJECT = ["을", "를"]        # 목적어 조사
    DIRECTION = ["에", "에게"]    # 방향 조사
    GENITIVE = ["의"]             # 소유/관계 조사
    COMPARISON = ["보다"]         # 비교 조사
    INSTRUMENT = ["으로", "로"]   # 수단 조사


class SOVContinuation:
    """SOV→CPS 변환의 continuation 표현

    SOV 구조에서 동사(마지막)가 continuation 역할:
    주어 → 목적어 → 동사 = CPS의 f(subject)(object) => result

    Korean: "레지스터 영에 5 대입하세요"
    Parse: [주어: 레지스터 영에] [목적어: 5] [동사: 대입하세요]
    CPS:   store(R0, 5, cont)
    """

    def __init__(self, arguments: list[str], verb: str) -> None:
        self.arguments = arguments
        self.verb = verb

    def __repr__(self) -> str:
        return f"SOV(args={self.arguments}, verb={self.verb})"


class FluxInterpreterKor:
    """유체 한국어 인터프리터

    한국어 자연어를 파싱 → 바이트코드 컴파일 → VM 실행.
    SOV 어순을 CPS로 변환하고 경어 수준을 CAP 옵코드로 삽입한다.
    """

    # 한자어 레지스터 이름
    REGISTER_NAMES: dict[str, int] = {
        "영": 0, "일": 1, "이": 2, "삼": 3,
        "사": 4, "오": 5, "육": 6, "칠": 7,
    }

    # 기본 숫자 매핑
    BASIC_NUMS: dict[str, int] = {
        "영": 0, "공": 0, "일": 1, "이": 2, "삼": 3,
        "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
        "십": 10, "백": 100, "천": 1000, "만": 10000,
    }

    def __init__(
        self,
        honorific_level: HonorificLevel = HonorificLevel.HAERACHE,
        enforce_honorifics: bool = False,
    ) -> None:
        self.vm = VM()
        self.vm.set_cap_level(honorific_level)
        self.honorific_level = honorific_level
        self.honorific_validator = HonorificValidator(default_level=honorific_level)
        self.enforce_honorifics = enforce_honorifics
        self.compiled_bytecode: list[tuple] = []
        self.parse_history: list[SOVContinuation] = []

    def reset(self) -> None:
        """상태 초기화"""
        self.vm.reset()
        self.vm.set_cap_level(self.honorific_level)
        self.compiled_bytecode = []
        self.parse_history = []

    def _parse_number(self, text: str) -> int:
        """텍스트를 숫자로 파싱"""
        text = text.strip()
        return KoreanNumeral.to_int(text)

    def _parse_register(self, text: str) -> int:
        """레지스터 이름을 인덱스로 파싱

        "레지스터 영" → 0, "레지스터 일" → 1, etc.
        """
        text = text.strip().replace("레지스터 ", "").replace("레지스터", "")
        for name, idx in self.REGISTER_NAMES.items():
            if name in text:
                return idx
        raise ParseError(f"레지스터를 해석할 수 없습니다: {text}")

    # ─── NL 패턴 매칭 (정규식) ──────────────────────────

    PAT_CALC = re.compile(
        r"계산\s+(.+?)\s+더하기\s+(.+?)(?:\s*$)"
    )

    PAT_MULTIPLY = re.compile(
        r"(.+?)\s+곱하기\s+(.+?)(?:\s*$)"
    )

    PAT_SUM_RANGE = re.compile(
        r"(.+?)\s+부터\s+(.+?)\s+까지\s+합(?:을\s+구하)?(?:.*)$"
    )

    PAT_FACTORIAL = re.compile(
        r"(.+?)\s+(?:의)?\s*팩토리얼(?:을\s+구하)?(?:.*)$"
    )

    PAT_REG_STORE = re.compile(
        r"레지스터\s+(\S+)\s+에\s+(.+?)\s+대입(?:.*)$"
    )

    PAT_REG_ADD = re.compile(
        r"레지스터\s+(\S+)\s+더하기\s+레지스터\s+(\S+)(?:\s|$)"
    )

    PAT_REG_MUL = re.compile(
        r"레지스터\s+(\S+)\s+곱하기\s+레지스터\s+(\S+)(?:\s|$)"
    )

    PAT_WHILE_NOT_ZERO = re.compile(
        r"레지스터\s+(\S+)\s+(?:이|가)\s+영이\s+아닌\s+동안(?:.*)$"
    )

    PAT_SEND = re.compile(
        r"(.+?)(?:에게|에게는)\s+(.+?)\s+전달(?:.*)$"
    )

    PAT_QUESTION = re.compile(
        r"(.+?)(?:에게|에게는)\s+(.+?)\s+질문(?:.*)$"
    )

    PAT_BROADCAST = re.compile(
        r"전체에\s+(.+?)\s+방송(?:.*)$"
    )

    # ─── SOV→CPS 컴파일 ─────────────────────────────────

    def _parse_sov(self, sentence: str) -> SOVContinuation:
        """SOV 구조 분석

        동사(마지막 요소)를 분리하고, 나머지를 인자로 추출.
        한국어 동사 어미를 감지하여 CPS continuation으로 표현.
        """
        tokens = sentence.split()
        if not tokens:
            raise ParseError("빈 문장입니다")

        # 동사(마지막 토큰 또는 동사 어미가 있는 마지막 토큰) 분리
        verb = tokens[-1]
        arguments = tokens[:-1]

        # 활용(동사 어미)이 포함된 경우: 마지막 두 토큰을 결합
        verb_endings = [
            "대입", "전달", "질문", "방송",
            "구하다", "계산하다", "실행하다",
            "대입하세요", "전달하세요", "방송하세요",
            "구해", "계산해", "실행해",
        ]
        if len(tokens) >= 2:
            combined = tokens[-2] + tokens[-1]
            if any(combined.endswith(ve) for ve in verb_endings):
                verb = combined
                arguments = tokens[:-2]

        return SOVContinuation(arguments, verb)

    def _detect_verb_type(self, verb: str) -> str:
        """동사 유형 감지"""
        verb_types = {
            "대입": "store", "대입하세요": "store",
            "전달": "send", "전달하세요": "send",
            "질문": "question", "질문하세요": "question",
            "방송": "broadcast", "방송하세요": "broadcast",
            "구하세요": "compute", "구해": "compute",
            "계산하세요": "compute", "계산해": "compute",
        }
        for key, vtype in verb_types.items():
            if verb.endswith(key):
                return vtype
        return "unknown"

    # ─── 바이트코드 생성 ─────────────────────────────────

    def compile_sentence(self, sentence: str) -> list[tuple]:
        """한국어 문장을 바이트코드로 컴파일

        경어 수준 검사 후 SOV→CPS 변환으로 바이트코드 생성.
        """
        bytecode = []
        sentence = sentence.strip()

        if not sentence or sentence.startswith("#"):
            return bytecode

        # 경어 수준 감지 & 검증
        detected_level = self.honorific_validator.detect(sentence)
        if self.enforce_honorifics:
            if detected_level.value > self.honorific_level.value:
                raise self.honorific_validator.format_error(
                    sentence, detected_level, self.honorific_level
                )

        # CAP_REQUIRE 삽입
        bytecode.append((Opcode.CAP_REQUIRE, detected_level.value))

        # SOV 분석
        sov = self._parse_sov(sentence)
        self.parse_history.append(sov)

        # 패턴 매칭 & 바이트코드 생성
        bc = self._match_and_compile(sentence, sov)
        bytecode.extend(bc)

        return bytecode

    def _match_and_compile(self, sentence: str, sov: SOVContinuation) -> list[tuple]:
        """문장 패턴 매칭 후 바이트코드 생성"""
        bytecode = []

        # 패턴 1: 계산 $a 더하기 $b
        m = self.PAT_CALC.match(sentence)
        if m:
            a = self._parse_number(m.group(1))
            b = self._parse_number(m.group(2))
            result = a + b
            bytecode.append((Opcode.LOAD_IMM, 0, a))
            bytecode.append((Opcode.LOAD_IMM, 1, b))
            bytecode.append((Opcode.ADD, 2, 0, 1))
            bytecode.append((Opcode.PRINT, f"결과: R2 = {a} + {b} = {result}"))
            return bytecode

        # 패턴 2: $a 곱하기 $b
        m = self.PAT_MULTIPLY.match(sentence)
        if m:
            a = self._parse_number(m.group(1))
            b = self._parse_number(m.group(2))
            result = a * b
            bytecode.append((Opcode.LOAD_IMM, 0, a))
            bytecode.append((Opcode.LOAD_IMM, 1, b))
            bytecode.append((Opcode.MUL, 2, 0, 1))
            bytecode.append((Opcode.PRINT, f"결과: R2 = {a} × {b} = {result}"))
            return bytecode

        # 패턴 3: $a 부터 $b 까지 합
        m = self.PAT_SUM_RANGE.match(sentence)
        if m:
            a = self._parse_number(m.group(1))
            b = self._parse_number(m.group(2))
            result = sum(range(a, b + 1))
            bytecode.append((Opcode.LOAD_IMM, 0, a))
            bytecode.append((Opcode.LOAD_IMM, 1, b))
            bytecode.append((Opcode.SUM_RANGE, 2, 0, 1))
            bytecode.append((Opcode.PRINT, f"결과: R2 = Σ({a}..{b}) = {result}"))
            return bytecode

        # 패턴 4: $a 의 팩토리얼
        m = self.PAT_FACTORIAL.match(sentence)
        if m:
            a = self._parse_number(m.group(1))
            result = math.factorial(a)
            bytecode.append((Opcode.LOAD_IMM, 0, a))
            bytecode.append((Opcode.FACTORIAL, 0))
            bytecode.append((Opcode.PRINT, f"결과: R0 = {a}! = {result}"))
            return bytecode

        # 패턴 5: 레지스터 영 에 $val 대입
        m = self.PAT_REG_STORE.match(sentence)
        if m:
            reg = self._parse_register(m.group(1))
            val = self._parse_number(m.group(2))
            bytecode.append((Opcode.LOAD_IMM, reg, val))
            bytecode.append((Opcode.PRINT, f"R{reg} ← {val}"))
            return bytecode

        # 패턴 6: 레지스터 영 더하기 레지스터 일
        m = self.PAT_REG_ADD.match(sentence)
        if m:
            reg_a = self._parse_register(m.group(1))
            reg_b = self._parse_register(m.group(2))
            bytecode.append((Opcode.ADD, 0, reg_a, reg_b))
            bytecode.append((Opcode.PRINT, f"R0 ← R{reg_a} + R{reg_b}"))
            return bytecode

        # 패턴 7: 레지스터 영 곱하기 레지스터 일
        m = self.PAT_REG_MUL.match(sentence)
        if m:
            reg_a = self._parse_register(m.group(1))
            reg_b = self._parse_register(m.group(2))
            bytecode.append((Opcode.MUL, 0, reg_a, reg_b))
            bytecode.append((Opcode.PRINT, f"R0 ← R{reg_a} × R{reg_b}"))
            return bytecode

        # 패턴 8: $agent 에게 $message 전달
        m = self.PAT_SEND.match(sentence)
        if m:
            agent = m.group(1).strip()
            message = m.group(2).strip()
            bytecode.append((Opcode.SEND, agent, message))
            return bytecode

        # 패턴 9: $agent 에게 $topic 질문
        m = self.PAT_QUESTION.match(sentence)
        if m:
            agent = m.group(1).strip()
            topic = m.group(2).strip()
            bytecode.append((Opcode.SEND, agent, f"질문: {topic}"))
            return bytecode

        # 패턴 10: 전체에 $message 방송
        m = self.PAT_BROADCAST.match(sentence)
        if m:
            message = m.group(1).strip()
            bytecode.append((Opcode.BROADCAST, message))
            return bytecode

        # 패턴 11: while 레지스터 영 이 영이 아닌 동안
        m = self.PAT_WHILE_NOT_ZERO.match(sentence)
        if m:
            reg = self._parse_register(m.group(1))
            # 루프 시작 마크
            loop_start = len(bytecode) + len(self.compiled_bytecode) + 2  # +2 for CAP_REQUIRE and this
            bytecode.append((Opcode.CMP, reg, 0))  # CMP로 R0 비교
            # NOTE: 루프 종료 주소는 나중에 back-patch
            bytecode.append((Opcode.PRINT, f"[while R{reg} != 0 루프 시작]"))
            return bytecode

        # 매칭 실패 → 기본 SOV 해석
        bytecode.append((Opcode.PRINT, f"[미해석: {sentence}]"))
        return bytecode

    # ─── 실행 ────────────────────────────────────────────

    def execute(self, source: str) -> list[str]:
        """한국어 소스 코드 실행

        Args:
            source: 한국어 자연어 프로그램 (여러 줄)

        Returns:
            출력 라인 목록
        """
        self.reset()
        all_bytecode = []

        lines = source.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bc = self.compile_sentence(line)
            all_bytecode.extend(bc)

        self.compiled_bytecode = all_bytecode
        return self.vm.execute(all_bytecode)

    def compile_only(self, source: str) -> list[tuple]:
        """컴파일만 수행 (VM 실행 없이)"""
        self.reset()
        all_bytecode = []

        lines = source.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bc = self.compile_sentence(line)
            all_bytecode.extend(bc)

        self.compiled_bytecode = all_bytecode
        return all_bytecode

    def get_state(self) -> dict[str, Any]:
        """현재 상태 반환"""
        return {
            "registers": self.vm.registers[:],
            "pc": self.vm.pc,
            "cap_level": self.vm.cap_level,
            "honorific_level": self.honorific_level.korean_name,
            "output": self.vm.output[:],
            "messages": self.vm.messages[:],
        }

    def format_bytecode(self, bytecode: list[tuple] | None = None) -> str:
        """바이트코드를 사람이 읽을 수 있는 형식으로"""
        if bytecode is None:
            bytecode = self.compiled_bytecode

        opcode_names = {op: op.name for op in Opcode}
        lines = ["바이트코드:", "─" * 40]
        for i, instr in enumerate(bytecode):
            op = instr[0]
            op_name = opcode_names.get(op, f"UNKNOWN({op})")
            args = ", ".join(str(a) for a in instr[1:])
            lines.append(f"  [{i:04d}] {op_name} {args}")
        return "\n".join(lines)
