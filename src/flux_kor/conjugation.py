"""
활용(活用) 시스템 — 한국어 동사 활용을 함수 합성으로

한국어 동사의 활용(어간에 어미가 결합하여 형태가 변하는 현상)을
고차 함수 합성(higher-order function composition)으로 모델링.

핵심 설계:
  1. 동사 어간(하, 되, 받, 주, 보, ...) = 기본 연산 (base function)
  2. 활용 어미(시, 아/어, 하, 세, ...) = 함수 변환자 (transform)
  3. 종결 어미(다, 요, 십시오, ...) = 실행 전략 (execution strategy)
  4. 활용 = compose(base, transform₁, transform₂, ..., strategy)

격식 등급에 따라 생성되는 바이트코드가 다름:
  해라체(평어): 최소 바이트코드 — 빠른 실행, 낮은 검증
  해요체(예사): 표준 바이트코드 — 기본 검증 포함
  하십시오체(격식): 완전 바이트코드 — 전체 검증, 로깅, 권한 확인

정규 동사 vs 불규칙 동사:
  정규 동사: 규칙적인 어미 결합 (예: 먹다 → 먹어, 먹었, 먹습니다)
  불규칙 동사: 특수한 형태 변화 (예: 듣다 → 들어, 돕다 → 도와)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════
# 활용 등급 (격식 수준)
# ═══════════════════════════════════════════════════════════════

class FormalityLevel(IntEnum):
    """격식 등급 — 활용 어미가 생성하는 바이트코드의 복잡도를 결정

    높은 격식 = 더 많은 검증/로그/보안 단계
    낮은 격식 = 최소 실행 경로
    """
    해라체 = 1       # 평어 (Plain) — 최소 바이트코드
    해체 = 2         # 반말 (Intimate) — 기본 + 동료 검증
    해요체 = 3       # 예사 (Polite) — 표준 + 사용자 검증
    하십시오체 = 4   # 격식 (Formal) — 전체 + 관리자 검증 + 로깅

    @property
    def korean_name(self) -> str:
        return {
            self.해라체: "해라체 (평어)",
            self.해체: "해체 (반말)",
            self.해요체: "해요체 (존댓말)",
            self.하십시오체: "하십시오체 (최경어)",
        }[self]

    @property
    def bytecode_overhead(self) -> int:
        """이 격식 등급이 추가하는 바이트코드 명령어 수"""
        return {
            self.해라체: 0,      # 오버헤드 없음
            self.해체: 1,        # 동료 확인 1개
            self.해요체: 3,      # 사용자 검증 3개
            self.하십시오체: 6,  # 전체 검증 + 로깅 6개
        }[self]

    @property
    def requires_verification(self) -> bool:
        """입력값 검증이 필요한지"""
        return self.value >= FormalityLevel.해요체

    @property
    def requires_logging(self) -> bool:
        """실행 로깅이 필요한지"""
        return self.value >= FormalityLevel.하십시오체


# ═══════════════════════════════════════════════════════════════
# 불규칙 동사 종류
# ═══════════════════════════════════════════════════════════════

class IrregularType(IntEnum):
    """불규칙 활용 종류"""
    규칙 = auto()       # 규칙 동사
    ㄹ탈락 = auto()     # ㄹ 탈락 (예: 살다 → 사, 살아)
    ㄷ변경 = auto()     # ㄷ → ㄹ 변경 (예: 듣다 → 들어)
    ㅂ변경 = auto()     # ㅂ → 오/우 변경 (예: 돕다 → 도와, 춥다 → 추워)
    ㅅ탈락 = auto()     # ㅅ 탈락 (예: 짓다 → 지어)
    르변경 = auto()     # 르 → ㄹ러/ㄹ라 (예: 모르다 → 몰라, 빠르다 → 빨라)
    ㅎ탈락 = auto()     # ㅎ 탈락 (예: 하다 → 해, 좋다 → 좋아)
    으탈락 = auto()     # 으 탈락 (예: 쓰다 → 써, 크다 → 커)


# ═══════════════════════════════════════════════════════════════
# 동사 어간 정의
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VerbStem:
    """동사 어간 정의

    Attributes:
        stem:      어간 (예: "하", "되", "받")
        irregular: 불규칙 종류
        bytecode_op: 이 동사가 매핑되는 기본 바이트코드 연산
        description: 한국어 설명
        category:  동사 범주 (동작, 상태, 존칭)
    """
    stem: str
    irregular: IrregularType = IrregularType.규칙
    bytecode_op: str = "NOP"
    description: str = ""
    category: str = "동작"


# 내장 동사 어간 사전
BUILTIN_STEMS: dict[str, VerbStem] = {
    # ── 기본 연산 동사 ──
    "하": VerbStem("하", IrregularType.ㅎ탈락, "EXEC",
                   "실행 동사 — 모든 동작의 기본 어간", "동작"),
    "되": VerbStem("되", IrregularType.규칙, "STATUS",
                   "상태 전환 동사 — 되다, 이루어지다", "상태"),
    "받": VerbStem("받", IrregularType.규칙, "RECV",
                   "수신 동사 — 데이터/메시지 수신", "동작"),
    "주": VerbStem("주", IrregularType.규칙, "GIVE",
                   "송신 동사 — 데이터/메시지 전송", "동작"),
    "보": VerbStem("보", IrregularType.규칙, "PRINT",
                   "출력 동사 — 화면/로그 출력", "동작"),

    # ── 산술 동사 ──
    "더하": VerbStem("더하", IrregularType.ㅎ탈락, "IADD",
                     "덧셈 동사 — 두 값을 더함", "동작"),
    "빼": VerbStem("빼", IrregularType.규칙, "ISUB",
                   "뺄셈 동사 — 두 값을 뺌", "동작"),
    "곱하": VerbStem("곱하", IrregularType.ㅎ탈락, "IMUL",
                     "곱셈 동사 — 두 값을 곱함", "동작"),
    "나누": VerbStem("나누", IrregularType.규칙, "IDIV",
                     "나눗셈 동사 — 두 값을 나눔", "동작"),

    # ── 이동/저장 동사 ──
    "대입": VerbStem("대입", IrregularType.규칙, "MOVI",
                     "대입 동사 — 값을 레지스터에 저장", "동작"),
    "이동": VerbStem("이동", IrregularType.규칙, "MOV",
                     "이동 동사 — 레지스터 간 값 복사", "동작"),
    "저장": VerbStem("저장", IrregularType.규칙, "STORE",
                     "저장 동사 — 메모리에 값 저장", "동작"),

    # ── 통신 동사 ──
    "전달": VerbStem("전달", IrregularType.규칙, "TELL",
                     "전달 동사 — 단방향 메시지 전송", "동작"),
    "질문": VerbStem("질문", IrregularType.규칙, "ASK",
                     "질문 동사 — 요청-응답 패턴", "동작"),
    "위임": VerbStem("위임", IrregularType.규칙, "DELEGATE",
                     "위임 동사 — 작업을 다른 에이전트에 위임", "동작"),
    "방송": VerbStem("방송", IrregularType.규칙, "BROADCAST",
                     "방송 동사 — 전체 에이전트에 메시지 전송", "동작"),

    # ── 제어 흐름 동사 ──
    "반복": VerbStem("반복", IrregularType.규칙, "LOOP",
                     "반복 동사 — 루프 실행", "제어"),
    "멈추": VerbStem("멈추", IrregularType.규칙, "HALT",
                     "정지 동사 — 실행 중단", "제어"),
    "건너뛰": VerbStem("건너뛰", IrregularType.규칙, "JMP",
                       "점프 동사 — 지정 위치로 이동", "제어"),

    # ── 비교 동사 ──
    "비교": VerbStem("비교", IrregularType.규칙, "CMP",
                     "비교 동사 — 두 값을 비교", "동작"),
    "같": VerbStem("같", IrregularType.규칙, "JE",
                   "동등 동사 — 같으면 점프", "상태"),
    "다르": VerbStem("다르", IrregularType.규칙, "JNE",
                     "부등 동사 — 다르면 점프", "상태"),

    # ── 상태 동사 ──
    "커지": VerbStem("커지", IrregularType.규칙, "INC",
                     "증가 동사 — 값을 1 증가", "상태"),
    "작아지": VerbStem("작아지", IrregularType.규칙, "DEC",
                       "감소 동사 — 값을 1 감소", "상태"),
    "부정": VerbStem("부정", IrregularType.규칙, "INEG",
                     "부정 동사 — 부호 반전", "상태"),
}


# ═══════════════════════════════════════════════════════════════
# 활용 어미 정의
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SuffixTransform:
    """활용 어미 변환 정의

    Attributes:
        suffix:       어미 표면형
        description:  설명
        bytecode_add: 이 어미가 추가하는 바이트코드 (해당 격식에서)
        priority:     적용 우선순위 (낮을수록 먼저 적용)
    """
    suffix: str
    description: str
    bytecode_add: list[str]
    priority: int = 0


# 활용 어미 사전
SUFFIX_TRANSFORMS: dict[str, SuffixTransform] = {
    # ── 존칭 접미사 ──
    "시": SuffixTransform("시", "존칭 접미사 — 주체를 높임",
                          ["TRUST_CHECK"], priority=1),
    "으시": SuffixTransform("으시", "존칭 접미사 (유택) — 주체를 높임",
                             ["TRUST_CHECK"], priority=1),

    # ── 시제 접미사 ──
    "았": SuffixTransform("았", "과거 시제 — 이미 완료된 동작",
                          ["VERIFY_PAST"], priority=2),
    "었": SuffixTransform("었", "과거 시제 (유택) — 이미 완료된 동작",
                          ["VERIFY_PAST"], priority=2),
    "겠": SuffixTransform("겠", "미래/추측 시제 — 예상되는 동작",
                          ["VERIFY_FUTURE"], priority=2),

    # ── 부정 접미사 ──
    "지": SuffixTransform("지", "부정 접미사 — 동작 부정 (않다 앞)",
                          ["NEGATE_FLAG"], priority=3),

    # ── 보조 동사 접미사 ──
    "내": SuffixTransform("내", "보조 동사 — 동작의 지속/완료",
                          ["VERIFY_COMPLETE"], priority=4),

    # ── 피동/사동 접미사 ──
    "이": SuffixTransform("이", "사동 접미사 — 타인에게 행위를 시킴",
                          ["CHECK_AUTHORITY"], priority=1),
    "히": SuffixTransform("히", "사동 접미사 — 타인에게 행위를 시킴",
                          ["CHECK_AUTHORITY"], priority=1),
    "리": SuffixTransform("리", "사동 접미사 — 타인에게 행위를 시킴",
                          ["CHECK_AUTHORITY"], priority=1),
    "기": SuffixTransform("기", "피동 접미사 — 행위를 받음",
                          ["CHECK_RECEIVER"], priority=1),
}


# ═══════════════════════════════════════════════════════════════
# 종결 어미 (실행 전략)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EndingStrategy:
    """종결 어미 정의 — 실행 전략을 결정

    Attributes:
        ending:          종결 어미 표면형
        formality:       격식 등급
        bytecode_prefix: 실행 전 삽입할 바이트코드
        bytecode_suffix: 실행 후 삽입할 바이트코드
    """
    ending: str
    formality: FormalityLevel
    bytecode_prefix: list[str]
    bytecode_suffix: list[str]


# 종결 어미 사전
ENDING_STRATEGIES: dict[str, EndingStrategy] = {
    # ── 해라체 (평어) ──
    "다": EndingStrategy("다", FormalityLevel.해라체, [], []),
    "ㄴ다": EndingStrategy("ㄴ다", FormalityLevel.해라체, [], []),
    "는다": EndingStrategy("는다", FormalityLevel.해라체, [], []),
    "라": EndingStrategy("라", FormalityLevel.해라체, [], []),
    "아라": EndingStrategy("아라", FormalityLevel.해라체, [], []),
    "어라": EndingStrategy("어라", FormalityLevel.해라체, [], []),
    "거라": EndingStrategy("거라", FormalityLevel.해라체, [], []),
    "너라": EndingStrategy("너라", FormalityLevel.해라체, [], []),

    # ── 해체 (반말) ──
    "아": EndingStrategy("아", FormalityLevel.해체, ["VERIFY_PEER"], []),
    "어": EndingStrategy("어", FormalityLevel.해체, ["VERIFY_PEER"], []),
    "해": EndingStrategy("해", FormalityLevel.해체, ["VERIFY_PEER"], []),

    # ── 해요체 (예사/존댓말) ──
    "아요": EndingStrategy("아요", FormalityLevel.해요체,
                           ["CAP_REQUIRE 3", "VERIFY_USER"], ["LOG_RESULT"]),
    "어요": EndingStrategy("어요", FormalityLevel.해요체,
                           ["CAP_REQUIRE 3", "VERIFY_USER"], ["LOG_RESULT"]),
    "해요": EndingStrategy("해요", FormalityLevel.해요체,
                           ["CAP_REQUIRE 3", "VERIFY_USER"], ["LOG_RESULT"]),
    "예요": EndingStrategy("예요", FormalityLevel.해요체,
                           ["CAP_REQUIRE 3", "VERIFY_USER"], ["LOG_RESULT"]),

    # ── 하십시오체 (격식/최경어) ──
    "습니다": EndingStrategy("습니다", FormalityLevel.하십시오체,
                             ["CAP_REQUIRE 4", "VERIFY_ADMIN", "LOG_START"],
                             ["LOG_END", "AUDIT_TRAIL"]),
    "ㅂ니다": EndingStrategy("ㅂ니다", FormalityLevel.하십시오체,
                             ["CAP_REQUIRE 4", "VERIFY_ADMIN", "LOG_START"],
                             ["LOG_END", "AUDIT_TRAIL"]),
    "십시오": EndingStrategy("십시오", FormalityLevel.하십시오체,
                             ["CAP_REQUIRE 4", "VERIFY_ADMIN", "LOG_START"],
                             ["LOG_END", "AUDIT_TRAIL"]),
    "으세요": EndingStrategy("으세요", FormalityLevel.하십시오체,
                             ["CAP_REQUIRE 4", "VERIFY_ADMIN", "LOG_START"],
                             ["LOG_END", "AUDIT_TRAIL"]),
    "세요": EndingStrategy("세요", FormalityLevel.하십시오체,
                           ["CAP_REQUIRE 4", "VERIFY_ADMIN", "LOG_START"],
                           ["LOG_END", "AUDIT_TRAIL"]),
    "합니다": EndingStrategy("합니다", FormalityLevel.하십시오체,
                             ["CAP_REQUIRE 4", "VERIFY_ADMIN", "LOG_START"],
                             ["LOG_END", "AUDIT_TRAIL"]),
}


# ═══════════════════════════════════════════════════════════════
# 활용 함수 (Conjugation Function)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConjugatedVerb:
    """활용된 동사 — 합성된 바이트코드 시퀀스

    동사 어간 + 활용 어미들 + 종결 어미가 합성된 결과.
    바이트코드 시퀀스는 실행 순서대로 정렬됨.

    Attributes:
        original:     원시 동사 형태 (예: "대입하십시오")
        stem:         추출된 어간
        suffixes:     적용된 활용 어미 목록
        ending:       종결 어미
        formality:    격식 등급
        bytecode_ops: 합성된 바이트코드 연산 목록
    """
    original: str
    stem: str
    suffixes: list[str]
    ending: str
    formality: FormalityLevel
    bytecode_ops: list[str] = field(default_factory=list)

    @property
    def total_bytecode_size(self) -> int:
        """총 바이트코드 크기 (기본 연산 + 활용 + 종결 오버헤드)"""
        base = 1  # 기본 연산 1개
        suffix_overhead = sum(len(SUFFIX_TRANSFORMS[s].bytecode_add)
                              for s in self.suffixes if s in SUFFIX_TRANSFORMS)
        ending_strategy = ENDING_STRATEGIES.get(self.ending)
        ending_overhead = (len(ending_strategy.bytecode_prefix) +
                           len(ending_strategy.bytecode_suffix)) if ending_strategy else 0
        return base + suffix_overhead + ending_overhead

    def __repr__(self) -> str:
        return (f"ConjugatedVerb({self.original!r}, "
                f"어간={self.stem}, 격식={self.formality.korean_name}, "
                f"명령어={len(self.bytecode_ops)}개)")


# ═══════════════════════════════════════════════════════════════
# 불규칙 활용 처리
# ═══════════════════════════════════════════════════════════════

def apply_irregular(stem: str, irregular: IrregularType, next_morpheme: str) -> str:
    """불규칙 동사 활용 적용

    불규칙 종류에 따라 어간의 형태를 변환.
    다음 형태소(어미/종결어미)의 첫 글자에 따라 변환 방식이 결정됨.

    Args:
        stem: 원시 어간
        irregular: 불규칙 종류
        next_morpheme: 다음 형태소

    Returns:
        변환된 어간
    """
    if irregular == IrregularType.규칙:
        return stem

    # 다음 형태소의 첫 글자
    next_char = next_morpheme[0] if next_morpheme else ""

    if irregular == IrregularType.ㄹ탈락:
        # ㄹ 탈락: 살다 → 살 + 아 → 살아, 살 + 었 → 살았
        # ㄹ은 모음 앞에서 탈락
        if next_char and _is_vowel(next_char):
            if stem.endswith("ㄹ") or (len(stem) >= 1 and stem[-1] == "ㄹ"):
                return stem  # 실제로는 받침 ㄹ이 탈락
            # "살" → "사" 처리
            if stem in ("살", "알", "만들", "날"):
                # ㄹ 받침 탈락 규칙 적용
                return stem
        return stem

    elif irregular == IrregularType.ㄷ변경:
        # ㄷ → ㄹ: 듣다 → 들어, 걷다 → 걸어
        if stem.endswith("ㄷ") and next_char and _is_vowel(next_char):
            # 한글 유니코드에서 ㄷ(ㄷ=3) → ㄹ(ㄹ=8) 변환
            result = stem[:-1]
            return result + "ㄹ"
        return stem

    elif irregular == IrregularType.ㅂ변경:
        # ㅂ → 오/우: 돕다 → 도와, 춥다 → 추워
        if next_char and _is_vowel(next_char):
            # "오" 또는 "우"로 변경
            if stem in ("돕", "돕"):
                return "도"
            elif stem in ("춥", "춥"):
                return "추"
        return stem

    elif irregular == IrregularType.ㅅ탈락:
        # ㅅ 탈락: 짓다 → 지어, 낫다 → 나아
        if next_char and _is_vowel(next_char):
            if stem.endswith("ㅅ"):
                return stem[:-1]
        return stem

    elif irregular == IrregularType.르변경:
        # 르 → ㄹ러/ㄹ라: 모르다 → 몰라, 빠르다 → 빨라
        if next_char and _is_vowel(next_char):
            if stem.endswith("르"):
                base = stem[:-1]
                if next_char in ("아", "아요", "아라"):
                    return base + "ㄹ라"
                else:
                    return base + "ㄹ러"
        return stem

    elif irregular == IrregularType.ㅎ탈락:
        # ㅎ 탈락: 하다 → 해, 좋다 → 좋아, 낳다 → 나아
        if next_char and _is_vowel(next_char):
            if stem.endswith("하"):
                return stem[:-1]  # "하" → "" (해로 축약됨)
        return stem

    elif irregular == IrregularType.으탈락:
        # 으 탈락: 쓰다 → 써, 크다 → 커
        if next_char and _is_vowel(next_char):
            if stem.endswith("으"):
                return stem[:-1]
        return stem

    return stem


def _is_vowel(char: str) -> bool:
    """한글 모음 판정"""
    if not char:
        return False
    code = ord(char[0])
    # 한글 모음 유니코드 범위: 3131~3163 (초성 모음) 또는 중성
    return 0x3131 <= code <= 0x318E or _is_hangul_medial(char)


def _is_hangul_medial(char: str) -> bool:
    """한글 중성(모음) 판정 — 유니코드 중성 범위"""
    if not char:
        return False
    code = ord(char[0])
    # 한글 음절에서 중성 판정
    if 0xAC00 <= code <= 0xD7A3:
        medial_index = ((code - 0xAC00) // 28) % 21
        return medial_index > 0
    # 독립 중성 유니코드
    return 0x314F <= code <= 0x3163


# ═══════════════════════════════════════════════════════════════
# 활용기 (Conjugator)
# ═══════════════════════════════════════════════════════════════

class ConjugationError(Exception):
    """활용 오류"""
    pass


class Conjugator:
    """한국어 동사 활용기 — 동사를 바이트코드 시퀀스로 합성

    동사 형태를 분석하여:
      1. 어간 추출 (stem extraction)
      2. 활용 어미 식별 (suffix identification)
      3. 종결 어미 식별 (ending identification)
      4. 불규칙 활용 적용 (irregular handling)
      5. 바이트코드 합성 (bytecode composition)

    사용 예시::

        conj = Conjugator()
        result = conj.conjugate("대입하십시오")
        # → ConjugatedVerb("대입하십시오", stem="대입", ending="하십시오", ...)
    """

    # 종결 어미를 긴 것부터 정렬 (최장 일치)
    _sorted_endings: list[str] = sorted(
        ENDING_STRATEGIES.keys(), key=len, reverse=True
    )

    # 활용 어미를 긴 것부터 정렬
    _sorted_suffixes: list[str] = sorted(
        SUFFIX_TRANSFORMS.keys(), key=len, reverse=True
    )

    def conjugate(self, verb_form: str) -> ConjugatedVerb:
        """동사 형태를 활용 분석하여 바이트코드 시퀀스 생성

        Args:
            verb_form: 활용된 동사 형태 (예: "대입하십시오", "계산해요")

        Returns:
            활용 분석 결과 (ConjugatedVerb)

        Raises:
            ConjugationError: 분석 불가능한 형태인 경우
        """
        verb_form = verb_form.strip()

        if not verb_form:
            raise ConjugationError("빈 동사 형태입니다")

        # 1. 종결 어미 식별 (최장 일치)
        ending = self._find_ending(verb_form)
        remaining = verb_form[:-len(ending)] if ending else verb_form

        # 종결 어미에서 격식 등급 결정
        strategy = ENDING_STRATEGIES.get(ending, EndingStrategy(
            "", FormalityLevel.해라체, [], []
        ))
        formality = strategy.formality

        # 2. 활용 어미 식별 (최장 일치)
        suffixes: list[str] = []
        while remaining:
            found_suffix = self._find_suffix(remaining)
            if found_suffix:
                suffixes.append(found_suffix)
                remaining = remaining[:-len(found_suffix)]
            else:
                break

        # 3. 남은 부분이 어간
        stem = remaining

        if not stem:
            raise ConjugationError(
                f"동사 어간을 추출할 수 없습니다: '{verb_form}' "
                f"(종결어미='{ending}', 접미사={suffixes})"
            )

        # 4. 어간에서 ㅎ탈락 등 불규칙 확인
        # "하십시오" → 어간 "대입" + 종결 "하십시오"
        # "하"가 종결 어미에 포함된 경우 어간은 그 앞
        if stem.endswith("하") and ending.startswith("십시오"):
            # "X하십시오" 패턴: "X"가 실제 어간
            real_stem = stem[:-1]  # "하" 제거
            if real_stem:
                stem = real_stem
                # "하"를 접미사로 처리
                suffixes.insert(0, "하")

        # 5. 사전에서 동사 어간 조회
        stem_def = self._find_stem(stem)

        # 6. 불규칙 활용 적용
        if stem_def and stem_def.irregular != IrregularType.규칙:
            next_morph = suffixes[0] if suffixes else (ending if ending else "")
            stem = apply_irregular(stem, stem_def.irregular, next_morph)

        # 7. 바이트코드 합성
        bytecode_ops = self._compose_bytecode(
            stem_def, suffixes, strategy
        )

        return ConjugatedVerb(
            original=verb_form,
            stem=stem,
            suffixes=suffixes,
            ending=ending,
            formality=formality,
            bytecode_ops=bytecode_ops,
        )

    def conjugate_with_level(
        self,
        verb_stem: str,
        formality: FormalityLevel,
    ) -> ConjugatedVerb:
        """어간과 격식 등급으로 직접 활용 생성

        종결 어미를 자동으로 선택하여 바이트코드 생성.

        Args:
            verb_stem: 동사 어간
            formality: 격식 등급

        Returns:
            활용 결과
        """
        # 격식 등급에 맞는 종결 어미 선택
        level_endings = {
            FormalityLevel.해라체: "다",
            FormalityLevel.해체: "해",
            FormalityLevel.해요체: "해요",
            FormalityLevel.하십시오체: "십시오",
        }
        ending = level_endings[formality]
        full_form = f"{verb_stem}{ending}"

        return self.conjugate(full_form)

    def _find_ending(self, text: str) -> str:
        """텍스트 끝에서 종결 어미를 찾음 (최장 일치)"""
        for ending in self._sorted_endings:
            if text.endswith(ending):
                return ending
        return ""

    def _find_suffix(self, text: str) -> str:
        """텍스트 끝에서 활용 어미를 찾음 (최장 일치)"""
        for suffix in self._sorted_suffixes:
            if text.endswith(suffix):
                return suffix
        return ""

    def _find_stem(self, stem: str) -> Optional[VerbStem]:
        """동사 어간 사전에서 조회 (부분 일치 지원)"""
        # 정확 일치
        if stem in BUILTIN_STEMS:
            return BUILTIN_STEMS[stem]

        # 부분 일치 (어간이 사전 어간으로 끝나는 경우)
        for key, vstem in BUILTIN_STEMS.items():
            if stem.endswith(key) and len(stem) <= len(key) + 2:
                return vstem

        # 기본 동사 "하" 패턴
        if stem.endswith("하"):
            return VerbStem(stem, IrregularType.ㅎ탈락, "EXEC",
                            f"복합 동사 — '{stem}' 실행", "동작")

        return None

    def _compose_bytecode(
        self,
        stem_def: Optional[VerbStem],
        suffixes: list[str],
        ending_strategy: EndingStrategy,
    ) -> list[str]:
        """활용 결과에서 바이트코드 시퀀스 합성

        합성 순서:
          1. 종결 어미 접두사 (CAP_REQUIRE, VERIFY_*, LOG_START)
          2. 활용 어미 추가 연산 (TRUST_CHECK, VERIFY_*, etc.)
          3. 기본 동사 연산 (MOVI, IADD, TELL, etc.)
          4. 종결 어미 접미사 (LOG_END, AUDIT_TRAIL)
        """
        ops: list[str] = []

        # 1. 종결 어미 접두사
        ops.extend(ending_strategy.bytecode_prefix)

        # 2. 활용 어미 추가 연산 (우선순위 순 정렬)
        suffix_defs = []
        for s in suffixes:
            if s in SUFFIX_TRANSFORMS:
                suffix_defs.append(SUFFIX_TRANSFORMS[s])
        suffix_defs.sort(key=lambda x: x.priority)
        for sdef in suffix_defs:
            ops.extend(sdef.bytecode_add)

        # 3. 기본 동사 연산
        if stem_def:
            ops.append(stem_def.bytecode_op)
        else:
            ops.append("NOP")

        # 4. 종결 어미 접미사
        ops.extend(ending_strategy.bytecode_suffix)

        return ops

    # ── 유틸리티 ──

    def detect_formality(self, verb_form: str) -> FormalityLevel:
        """동사 형태에서 격식 등급 감지

        종결 어미를 분석하여 격식 등급을 반환.
        종결 어미를 찾지 못하면 기본값(해라체) 반환.

        Args:
            verb_form: 동사 형태

        Returns:
            감지된 격식 등급
        """
        ending = self._find_ending(verb_form)
        if ending and ending in ENDING_STRATEGIES:
            return ENDING_STRATEGIES[ending].formality
        return FormalityLevel.해라체

    def list_stems(self) -> list[str]:
        """내장 동사 어간 목록 반환"""
        return list(BUILTIN_STEMS.keys())

    def list_endings(self) -> dict[str, str]:
        """종결 어미 목록 반환 (어미 → 격식 등급)"""
        return {
            ending: strategy.formality.korean_name
            for ending, strategy in ENDING_STRATEGIES.items()
        }


# ═══════════════════════════════════════════════════════════════
# 함수 합성 유틸리티
# ═══════════════════════════════════════════════════════════════

def compose_bytecode(*conjugated_verbs: ConjugatedVerb) -> list[str]:
    """여러 활용 동사를 순차 합성

    각 활용 동사의 바이트코드 시퀀스를 순서대로 연결.
    중복되는 CAP_REQUIRE는 제거하여 최적화.

    Args:
        *conjugated_verbs: 활용된 동사 목록

    Returns:
        합성된 바이트코드 연산 목록
    """
    all_ops: list[str] = []
    seen_caps: set[str] = set()

    for cv in conjugated_verbs:
        for op in cv.bytecode_ops:
            # 중복 CAP_REQUIRE 제거
            if op.startswith("CAP_REQUIRE") and op in seen_caps:
                continue
            if op.startswith("CAP_REQUIRE"):
                seen_caps.add(op)
            all_ops.append(op)

    return all_ops


def formality_to_bytecode_overhead(level: FormalityLevel) -> list[str]:
    """격식 등급에 해당하는 오버헤드 바이트코드 반환

    낮은 격식의 오버헤드는 높은 격식에 포함됨 (누적).
    """
    overhead: list[str] = []

    if level >= FormalityLevel.해체:
        overhead.append("VERIFY_PEER")
    if level >= FormalityLevel.해요체:
        overhead.append("CAP_REQUIRE 3")
        overhead.append("VERIFY_USER")
        overhead.append("LOG_RESULT")
    if level >= FormalityLevel.하십시오체:
        overhead.append("VERIFY_ADMIN")
        overhead.append("LOG_START")
        overhead.append("LOG_END")
        overhead.append("AUDIT_TRAIL")

    return overhead
