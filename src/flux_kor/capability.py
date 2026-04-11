"""
경어 능력 체계 (Honorific Capability System)
=============================================

한국어 경어 체계의 5등급을 RBAC 권한 비트 마스크로 매핑.
사회적 관계가 접근 제어가 되는 순간 — 한국어는 프로그래밍 언어보다
먼저 A2A 프로그래밍을 내장하고 있었다.

경어 등급 (낮은 것 → 높은 것):
  1. 해라체 (Haera-che)      → CAP_ANON  (0x40)  미신뢰, 읽기 전용
  2. 해체 (Hae-che)          → CAP_LOCAL (0x80)  샌드박스/로컬 연산
  3. 해요체 (Haeyo-che)      → CAP_USER  (0xC0)  표준 사용자 연산
  4. 하십시오체 (Hashipsio-che)→ CAP_ADMIN (0xF0)  관리 연산
  5. 합쇼체 (Hasipsio-che)   → CAP_ROOT  (0xFF)  전체 시스템 접근

핵심 원리:
  경어 등급 = 사회적 거리가 결정하는 권한.
  높은 경어(존댓말) = 상대에게 더 많은 권한을 부여.
  낮은 경어(반말) = 적은 권한, 제한된 접근.
"""

from __future__ import annotations

import re
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 경어 능력 등급
# ═══════════════════════════════════════════════════════════════

class CapabilityLevel(IntEnum):
    """경어 기반 능력 등급 — 사회적 관계를 접근 제어로.

    높은 비트 = 더 높은 권한. AND 마스크로 하위 비트를 검사.
    """
    CAP_ANON  = 0x40   # 해라체 (평어) — 미신뢰, 읽기 전용
    CAP_LOCAL = 0x80   # 해체 (반말) — 샌드박스/로컬 연산만
    CAP_USER  = 0xC0   # 해요체 (존댓말) — 표준 사용자 연산
    CAP_ADMIN = 0xF0   # 하십시오체 (격식체) — 관리 연산
    CAP_ROOT  = 0xFF   # 합쇼체 (최경어) — 전체 시스템 접근


# 경어 등급별 상세 정보
_CAPABILITY_INFO: dict[CapabilityLevel, dict[str, str]] = {
    CapabilityLevel.CAP_ANON: {
        "korean": "해라체 (평어)",
        "description": "미신뢰 코드 — 읽기 전용, 쓰기 불가",
        "role": "anonymous/untrusted",
        "sanskrit_equivalent": "śūnya-pramāṇam (무권한)",
        "vibhakti_scope": "pañcamī (ablative/source only)",
    },
    CapabilityLevel.CAP_LOCAL: {
        "korean": "해체 (반말)",
        "description": "샌드박스 내 연산 — 로컬 범위만",
        "role": "local/sandboxed",
        "sanskrit_equivalent": "saptamī-pradeśaḥ (장소적 범위)",
        "vibhakti_scope": "saptamī (locative/context only)",
    },
    CapabilityLevel.CAP_USER: {
        "korean": "해요체 (존댓말)",
        "description": "표준 사용자 — 일반 연산 허용",
        "role": "standard-user",
        "sanskrit_equivalent": "prathamā-dhāraṇam (공용 접근)",
        "vibhakti_scope": "prathamā (nominative/public)",
    },
    CapabilityLevel.CAP_ADMIN: {
        "korean": "하십시오체 (격식체)",
        "description": "관리자 — 설정 변경, 사용자 관리 허용",
        "role": "admin",
        "sanskrit_equivalent": "caturthī-pradānam (권한 부여)",
        "vibhakti_scope": "caturthī (dative/capability-granting)",
    },
    CapabilityLevel.CAP_ROOT: {
        "korean": "합쇼체 (최경어)",
        "description": "시스템 최고 권한 — 모든 연산 허용",
        "role": "root/system",
        "sanskrit_equivalent": "sarva-śaktiḥ (전체 권한)",
        "vibhakti_scope": "sambodhana (vocative/agent-invocation)",
    },
}


# ═══════════════════════════════════════════════════════════════
# 동사 어미 패턴 → 능력 등급 매핑
# ═══════════════════════════════════════════════════════════════

class HonorificCapabilityResolver:
    """경어 능력 해석기 — 동사 어미에서 능력 등급을 감지.

    소스 텍스트의 말뭉치 종결 어미를 분석하여
    해당 문장/함수의 실행 능력을 결정.

    능력 전파 (Capability Propagation):
      함수는 호출자(caller)의 경어 등급을 상속받음.
      합쇼체 함수가 해요체 함수를 호출하면,
      내부 함수는 해요체 능력으로 실행됨.

    Usage::

        resolver = HonorificCapabilityResolver()
        level = resolver.resolve("데이터를 전송하십시오")
        # → CAP_ADMIN (하십시오체)

        # 능력 검사
        can_exec = resolver.check(level, CapabilityLevel.CAP_USER)
    """

    # 동사 어미 → 능력 등급 (정규식 패턴, 우선순위 높은 것부터)
    _LEVEL_PATTERNS: list[tuple[re.Pattern, CapabilityLevel]] = [
        # 합쇼체: -나이다, -사옵니다, -옵소서 (최고 경어, 의례 표현)
        (re.compile(r"(나이다|사옵니다|옵소서|나이다)$"), CapabilityLevel.CAP_ROOT),
        # 하십시오체: -ㅂ니다/습니다, -십시오, -(으)세요, -합니다
        (re.compile(r"(습니다|ㅂ니다|십시오|으세요|하세요|합니다)$"),
         CapabilityLevel.CAP_ADMIN),
        # 해요체: -아요/어요/해요, -(이)에요
        (re.compile(r"(아요|어요|해요|이에요|예요)$"), CapabilityLevel.CAP_USER),
        # 해체: -아/어/해 (명령형 종결 어미가 아닌 경우)
        (re.compile(r"(?<![가-힣])해(?![요라서])[^\s]*$"), CapabilityLevel.CAP_LOCAL),
        (re.compile(r"(해라|거라|너라)$"), CapabilityLevel.CAP_LOCAL),
        # 해라체: -다/ㄴ다/는다, -(으)라, -아라/어라
        (re.compile(r"(한다|이다|아라|어라|우라|다)$"), CapabilityLevel.CAP_ANON),
    ]

    # 한자어/의존 명사 + 경어 조합 패턴
    _HONORIFIC_MARKER_PATTERNS: list[tuple[re.Pattern, CapabilityLevel]] = [
        # 존칭 접미사 + 하십시오체 → 합쇼체 승격
        (re.compile(r"(귀하|님께서|대평하).*(하십시오|옵소서)"), CapabilityLevel.CAP_ROOT),
        # 존칭 접미사 + 하십시오체 → CAP_ADMIN 유지
        (re.compile(r"(님께|귀하).*(세요|으세요)"), CapabilityLevel.CAP_ADMIN),
    ]

    def __init__(self, default_level: CapabilityLevel = CapabilityLevel.CAP_ANON):
        self.default_level = default_level
        self._call_stack: list[CapabilityLevel] = []  # 능력 전파 추적
        self._function_caps: dict[str, CapabilityLevel] = {}  # 함수별 능력 캐시

    def resolve(self, text: str) -> CapabilityLevel:
        """텍스트에서 경어 능력 등급을 감지.

        Args:
            text: 한국어 텍스트 (문장 또는 동사 구)

        Returns:
            감지된 능력 등급
        """
        text = text.strip()

        # 1. 존칭 + 경어 조합 패턴 먼저 검사
        for pattern, level in self._HONORIFIC_MARKER_PATTERNS:
            if pattern.search(text):
                return level

        # 2. 동사 어미 패턴 검사
        for pattern, level in self._LEVEL_PATTERNS:
            if pattern.search(text):
                return level

        return self.default_level

    def resolve_function(self, func_name: str, func_body: str) -> CapabilityLevel:
        """함수의 경어 능력을 감지하고 캐시.

        Args:
            func_name: 함수 이름
            func_body: 함수 본문 텍스트

        Returns:
            감지된 능력 등급
        """
        if func_name in self._function_caps:
            return self._function_caps[func_name]

        level = self.resolve(func_body)
        self._function_caps[func_name] = level
        return level

    def check(self, current: CapabilityLevel, required: CapabilityLevel) -> bool:
        """능력 검사 — 현재 등급이 필요한 등급을 만족하는지.

        비트 마스크 AND 연산: current의 상위 비트가 required를 포함하면 OK.

        Args:
            current: 현재 실행 컨텍스트의 능력
            required: 연산에 필요한 능력

        Returns:
            접근 허용 여부
        """
        # 상위 4비트 비교 (0xF0 마스크)
        return (current & 0xF0) >= (required & 0xF0)

    def check_exact(self, current: CapabilityLevel, required: CapabilityLevel) -> bool:
        """정확한 등급 일치 검사."""
        return current == required

    def enter_call(self, caller_level: CapabilityLevel) -> None:
        """함수 호출 진입 — 호출자 능력을 스택에 푸시.

        Args:
            caller_level: 호출자의 경어 능력 등급
        """
        self._call_stack.append(caller_level)

    def exit_call(self) -> CapabilityLevel | None:
        """함수 호출 종료 — 능력 스택에서 팝.

        Returns:
            이전 능력 등급 (없으면 None)
        """
        if self._call_stack:
            return self._call_stack.pop()
        return None

    @property
    def call_depth(self) -> int:
        """현재 호출 깊이."""
        return len(self._call_stack)

    def propagate(self, caller_level: CapabilityLevel, callee_body: str) -> CapabilityLevel:
        """능력 전파 — 호출자 능력과 피호출자 본문을 결합.

        규칙:
          1. 피호출자 본문의 경어가 더 높으면 → 피호출자 등급 사용
          2. 피호출자 본문의 경어가 낮거나 같으면 → 호출자 등급 사용

        Args:
            caller_level: 호출자의 능력 등급
            callee_body: 피호출자의 소스 텍스트

        Returns:
            전파된 최종 능력 등급
        """
        callee_level = self.resolve(callee_body)
        return max(caller_level, callee_level)

    def format_table(self) -> str:
        """경어 능력 등급 테이블 반환."""
        lines = [
            "╔═══════════════════════════════════════════════════════════════╗",
            "║  경어 능력 체계 — Honorific Capability Ladder                 ║",
            "║  한국어 경어 → RBAC 권한 비트 마스크                          ║",
            "╠══════════╦════════════════════════════╦═══════╦══════════════╣",
            "║  비트    ║ 경어 등급                  ║ 역할  ║ 설명          ║",
            "╠══════════╬════════════════════════════╬═══════╬══════════════╣",
        ]
        for level in CapabilityLevel:
            info = _CAPABILITY_INFO[level]
            lines.append(
                f"║  0x{level.value:02X}   ║ {info['korean']:<24s} "
                f"║ {info['role']:<13s} ║ {info['description'][:12]:12s} ║"
            )
        lines.append("╚══════════╩════════════════════════════╩═══════╩══════════════╝")
        return "\n".join(lines)


@dataclass
class CapabilityToken:
    """능력 토큰 — 소스 텍스트에서 추출된 경어 능력 정보.

    Attributes:
        text: 원시 텍스트
        level: 감지된 능력 등급
        position: 텍스트 내 시작 위치
        pattern_name: 매칭된 패턴 설명
    """
    text: str
    level: CapabilityLevel
    position: int = 0
    pattern_name: str = ""

    @property
    def info(self) -> dict[str, str]:
        return _CAPABILITY_INFO[self.level]

    def __repr__(self) -> str:
        return (
            f"CapabilityToken({self.text!r}, "
            f"level=0x{self.level.value:02X}, "
            f"role={self.info['role']})"
        )


class CapabilityError(Exception):
    """능력 부족 오류."""
    pass
