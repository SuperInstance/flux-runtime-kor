"""
조사 범위 컴파일러 (Particle Scope Compiler)
==============================================

한국어 조사(助詞)를 프로그래밍의 범위(scope) 연산자로 컴파일.
기존 particles.py 모듈의 ParticleKind를 확장하여
명시적인 ScopeCode 시스템과 조사 적층(스태킹) 분석을 추가.

조사 → 범위 매핑:
  은/는 (topic)    → SCOPE_TOPIC:     계산 스코프 정의 (Chinese R63과 동등)
  이/가 (subject)  → SCOPE_SUBJECT:   좁은 주어 범위
  을/를 (object)   → SCOPE_OBJECT:    객체 바인딩 범위
  에게/한테 (recipient) → SCOPE_DELEGATE: 위임 범위
  에서/부터 (source) → SCOPE_SOURCE:   입력 범위
  (으)로 (means)   → SCOPE_INSTRUMENT: 도구/함수 범위

조사 적층 (Particle Stacking):
  "나는 너에게 이것을 준다"
  → [SCOPE_TOPIC, SCOPE_DELEGATE, SCOPE_OBJECT]
  각 조사가 순차적으로 범위 스택을 형성.
"""

from __future__ import annotations

import re
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Optional

from flux_kor.particles import (
    ParticleKind,
    ParticleToken,
    ParticleAnalyzer,
    PARTICLE_BYTECODE_MAP,
)


# ═══════════════════════════════════════════════════════════════
# 범위 코드
# ═══════════════════════════════════════════════════════════════

class ScopeCode(IntEnum):
    """조사 기반 범위 코드 — 각 조사가 정의하는 범위 유형.

    명시적인 비트 마스크로 표현하여 범위 조합이 가능:
      topic_scope | delegate_scope = 두 범위의 결합
    """
    SCOPE_NONE        = 0x00  # 범위 없음
    SCOPE_TOPIC       = 0x01  # 은/는 — 현재 계산 스코프 정의
    SCOPE_SUBJECT     = 0x02  # 이/가 — 좁은 주어 범위
    SCOPE_OBJECT      = 0x04  # 을/를 — 객체 바인딩
    SCOPE_DELEGATE    = 0x08  # 에게/한테 — 위임/delegation
    SCOPE_SOURCE      = 0x10  # 에서/부터 — 입력/source
    SCOPE_INSTRUMENT  = 0x20  # (으)로 — 도구/함수 적용
    SCOPE_OWNERSHIP   = 0x40  # 의 — 소유/속성
    SCOPE_COMPARISON  = 0x80  # 보다 — 비교
    SCOPE_LIMIT       = 0x100 # 까지 — 상한/bound
    SCOPE_COMBINE     = 0x200 # 과/와 — 병렬 결합
    SCOPE_EXCLUSIVE   = 0x400 # 만 — 독점/한정
    SCOPE_ADDITIONAL  = 0x800 # 도 — 추가
    SCOPE_DISTRIBUTIVE = 0x1000  # 마다 — 분포/map


# ParticleKind → ScopeCode 매핑
_PARTICLE_TO_SCOPE: dict[ParticleKind, ScopeCode] = {
    ParticleKind.은는:   ScopeCode.SCOPE_TOPIC,
    ParticleKind.이가:   ScopeCode.SCOPE_SUBJECT,
    ParticleKind.을를:   ScopeCode.SCOPE_OBJECT,
    ParticleKind.에에게: ScopeCode.SCOPE_DELEGATE,
    ParticleKind.에서:   ScopeCode.SCOPE_SOURCE,
    ParticleKind.으로:   ScopeCode.SCOPE_INSTRUMENT,
    ParticleKind.의:     ScopeCode.SCOPE_OWNERSHIP,
    ParticleKind.보다:   ScopeCode.SCOPE_COMPARISON,
    ParticleKind.까지:   ScopeCode.SCOPE_LIMIT,
    ParticleKind.과와:   ScopeCode.SCOPE_COMBINE,
    ParticleKind.만:     ScopeCode.SCOPE_EXCLUSIVE,
    ParticleKind.도:     ScopeCode.SCOPE_ADDITIONAL,
    ParticleKind.마다:   ScopeCode.SCOPE_DISTRIBUTIVE,
}


# 범위 코드 → 설명
_SCOPE_DESCRIPTIONS: dict[ScopeCode, str] = {
    ScopeCode.SCOPE_TOPIC:      "은/는 (topic) — 계산 스코프 정의",
    ScopeCode.SCOPE_SUBJECT:    "이/가 (subject) — 주어 활성화",
    ScopeCode.SCOPE_OBJECT:     "을/를 (object) — 객체 바인딩",
    ScopeCode.SCOPE_DELEGATE:   "에게/한테 (delegate) — 위임/메시지 전송",
    ScopeCode.SCOPE_SOURCE:     "에서/부터 (source) — 입력/출발점",
    ScopeCode.SCOPE_INSTRUMENT: "(으)로 (instrument) — 도구/함수 적용",
    ScopeCode.SCOPE_OWNERSHIP:  "의 (ownership) — 소유/속성 접근",
    ScopeCode.SCOPE_COMPARISON: "보다 (comparison) — 비교",
    ScopeCode.SCOPE_LIMIT:      "까지 (limit) — 범위 상한",
    ScopeCode.SCOPE_COMBINE:    "과/와 (combine) — 병렬 결합",
    ScopeCode.SCOPE_EXCLUSIVE:  "만 (exclusive) — 독점 접근",
    ScopeCode.SCOPE_ADDITIONAL: "도 (additional) — 추가 연산",
    ScopeCode.SCOPE_DISTRIBUTIVE: "마다 (distributive) — 분포/map",
}


# ═══════════════════════════════════════════════════════════════
# 범위 토큰
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScopeToken:
    """범위 토큰 — 조사에서 추출된 범위 정보.

    Attributes:
        noun: 조사가 붙은 명사
        particle: 조사 표면형
        scope: 범위 코드
        depth: 중첩 깊이 (0=최상위)
        full_text: 명사 + 조사 전체
    """
    noun: str
    particle: str
    scope: ScopeCode
    depth: int = 0
    full_text: str = ""

    @property
    def description(self) -> str:
        return _SCOPE_DESCRIPTIONS.get(self.scope, "알 수 없는 범위")

    def __repr__(self) -> str:
        return (
            f"ScopeToken({self.full_text!r}, "
            f"scope=0x{self.scope.value:04X}, "
            f"depth={self.depth})"
        )


# ═══════════════════════════════════════════════════════════════
# 범위 스택
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScopeStack:
    """범위 스택 — 조사 적층으로 형성된 범위 계층 구조.

    한국어 문장에서 조사가 나타나는 순서대로 범위를 스택에 푸시.
    각 스택 레벨이 하나의 실행 범위를 나타냄.

    예시: "나는 너에게 이것을 준다"
    → push(TOPIC, "나는")
    → push(DELEGATE, "너에게")
    → push(OBJECT, "이것을")
    → scope_mask = TOPIC | DELEGATE | OBJECT
    """

    _stack: list[ScopeToken] = field(default_factory=list)

    @property
    def scope_mask(self) -> int:
        """현재 스택의 결합된 범위 마스크."""
        mask = 0
        for token in self._stack:
            mask |= token.scope.value
        return mask

    @property
    def depth(self) -> int:
        """현재 스택 깊이."""
        return len(self._stack)

    @property
    def top(self) -> Optional[ScopeToken]:
        """스택 최상단 토큰."""
        return self._stack[-1] if self._stack else None

    @property
    def has_topic(self) -> bool:
        return bool(self.scope_mask & ScopeCode.SCOPE_TOPIC)

    @property
    def has_subject(self) -> bool:
        return bool(self.scope_mask & ScopeCode.SCOPE_SUBJECT)

    @property
    def has_object(self) -> bool:
        return bool(self.scope_mask & ScopeCode.SCOPE_OBJECT)

    @property
    def has_delegate(self) -> bool:
        return bool(self.scope_mask & ScopeCode.SCOPE_DELEGATE)

    @property
    def tokens(self) -> list[ScopeToken]:
        return list(self._stack)

    def push(self, token: ScopeToken) -> None:
        """범위 토큰을 스택에 푸시."""
        token.depth = len(self._stack)
        self._stack.append(token)

    def pop(self) -> Optional[ScopeToken]:
        """스택에서 팝."""
        if self._stack:
            return self._stack.pop()
        return None

    def clear(self) -> None:
        """스택 초기화."""
        self._stack.clear()

    def to_scope_string(self) -> str:
        """범위 스택을 읽기 쉬운 문자열로 변환."""
        parts = [f"0x{t.scope.value:04X}({t.description[:15]})" for t in self._stack]
        return "[" + ", ".join(parts) + "]"

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        return f"ScopeStack(depth={self.depth}, mask=0x{self.scope_mask:04X})"


# ═══════════════════════════════════════════════════════════════
# 조사 범위 컴파일러
# ═══════════════════════════════════════════════════════════════

class ParticleScopeCompiler:
    """조사 범위 컴파일러 — 한국어 텍스트에서 조사를 분석하고 범위를 결정.

    기존 ParticleAnalyzer를 사용하여 텍스트에서 조사를 추출한 후,
    각 조사를 ScopeCode로 매핑하고 ScopeStack을 구성.

    Usage::

        compiler = ParticleScopeCompiler()
        result = compiler.compile("나는 너에게 이것을 준다")
        # result.stack = [TOPIC, DELEGATE, OBJECT]
        # result.scope_mask = 0x01 | 0x08 | 0x04 = 0x0D
    """

    def __init__(self) -> None:
        self._analyzer = ParticleAnalyzer()
        self._stack = ScopeStack()

    def compile(self, text: str) -> ScopeCompilationResult:
        """한국어 텍스트에서 조사를 분석하고 범위 스택을 구성.

        Args:
            text: 한국어 텍스트

        Returns:
            범위 컴파일 결과
        """
        self._stack.clear()

        # 기존 조사 분석기로 토큰 추출
        particle_tokens = self._analyzer.analyze(text)

        # 각 조사를 범위 토큰으로 변환
        scope_tokens: list[ScopeToken] = []
        for pt in particle_tokens:
            scope_code = _PARTICLE_TO_SCOPE.get(pt.kind, ScopeCode.SCOPE_NONE)
            st = ScopeToken(
                noun=pt.noun,
                particle=pt.surface,
                scope=scope_code,
                full_text=pt.full_text,
            )
            scope_tokens.append(st)
            self._stack.push(st)

        # 바이트코드 힌트 생성
        bytecode_hints = self._analyzer.get_bytecode_hints()

        return ScopeCompilationResult(
            scope_tokens=scope_tokens,
            stack=self._stack,
            scope_mask=self._stack.scope_mask,
            bytecode_hints=bytecode_hints,
            original_text=text,
        )

    def compile_multi(self, sentences: list[str]) -> list[ScopeCompilationResult]:
        """여러 문장을 컴파일.

        각 문장은 독립적인 범위 스택을 가짐.
        """
        return [self.compile(s) for s in sentences]

    def analyze_particle_nesting(self, text: str) -> list[ScopeToken]:
        """조사 중첩 구조를 분석.

        소유 조사(의)가 결합되는 경우 등의 중첩 패턴을 감지.
        "회사의 직원의 이름" → OWNERSHIP > OWNERSHIP (중첩 소유)
        """
        self._stack.clear()
        particle_tokens = self._analyzer.analyze(text)

        result: list[ScopeToken] = []
        for pt in particle_tokens:
            scope_code = _PARTICLE_TO_SCOPE.get(pt.kind, ScopeCode.SCOPE_NONE)
            st = ScopeToken(
                noun=pt.noun,
                particle=pt.surface,
                scope=scope_code,
                full_text=pt.full_text,
            )
            # 소유 중첩 깊이 추적
            if scope_code == ScopeCode.SCOPE_OWNERSHIP:
                ownership_depth = sum(
                    1 for t in result if t.scope == ScopeCode.SCOPE_OWNERSHIP
                )
                st.depth = ownership_depth
            self._stack.push(st)
            result.append(st)

        return result


@dataclass
class ScopeCompilationResult:
    """범위 컴파일 결과.

    Attributes:
        scope_tokens: 추출된 범위 토큰 목록
        stack: 범위 스택
        scope_mask: 결합된 범위 마스크
        bytecode_hints: 바이트코드 힌트 목록
        original_text: 원시 텍스트
    """
    scope_tokens: list[ScopeToken]
    stack: ScopeStack
    scope_mask: int
    bytecode_hints: list[str]
    original_text: str

    @property
    def num_particles(self) -> int:
        return len(self.scope_tokens)

    def describe(self) -> str:
        """결과를 읽기 쉬운 형식으로 반환."""
        lines = [f"텍스트: {self.original_text!r}"]
        lines.append(f"조사 수: {self.num_particles}")
        lines.append(f"범위 마스크: 0x{self.scope_mask:04X}")
        for i, st in enumerate(self.scope_tokens):
            lines.append(
                f"  [{i}] {st.full_text!r} → "
                f"0x{st.scope.value:04X} ({st.description})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ScopeCompilationResult(particles={self.num_particles}, "
            f"mask=0x{self.scope_mask:04X})"
        )
