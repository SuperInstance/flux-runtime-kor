"""
경어 시스템 — 한국어 높임말을 RBAC 권한 관리로

한국어의 4가지 경어 체계(높임말 등급)를 FLUX VM의 CAP_REQUIRE 옵코드에 매핑.
동사 어미의 형태로 경어 수준을 감지하고 검증한다.

경어 등급:
  1. 하십시오체 (Hasipsioche) — 격식체/존댓말 (admin capability)
  2. 해요체 (Haeyoche)       — 예사체/존댓말 (standard user)
  3. 해체 (Haeche)          — 반말 (peer)
  4. 해라체 (Haerache)      — 평어/반말 (internal/system)

매핑:
  하십시오체 → CAP level 4 (admin)
  해요체     → CAP level 3 (standard user)
  해체       → CAP level 2 (peer)
  해라체     → CAP level 1 (system/internal)
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import Optional


class HonorificLevel(IntEnum):
    """경어 수준 열거형 — 높은 수준일수록 높은 권한"""
    HAERACHE = 1       # 해라체 (평어) → internal/system
    HAE = 2            # 해체 (반말) → peer
    HAEOYO = 3         # 해요체 → standard user
    HASIPSIOCHE = 4    # 하십시오체 → admin

    @classmethod
    def from_name(cls, name: str) -> Optional[HonorificLevel]:
        """한국어 이름으로 경어 수준 찾기"""
        mapping = {
            "하십시오체": cls.HASIPSIOCHE,
            "해요체": cls.HAEOYO,
            "해체": cls.HAE,
            "해라체": cls.HAERACHE,
            "하십시오": cls.HASIPSIOCHE,
            "해요": cls.HAEOYO,
            "해": cls.HAE,
            "해라": cls.HAERACHE,
            # English aliases
            "formal": cls.HASIPSIOCHE,
            "polite": cls.HAEOYO,
            "intimate": cls.HAE,
            "plain": cls.HAERACHE,
        }
        return mapping.get(name.strip())

    @property
    def korean_name(self) -> str:
        return {
            self.HAERACHE: "해라체 (평어)",
            self.HAE: "해체 (반말)",
            self.HAEOYO: "해요체 (존댓말)",
            self.HASIPSIOCHE: "하십시오체 (최경어)",
        }[self]

    @property
    def capability_bit(self) -> int:
        """RBAC 권한 비트 마스크"""
        return 1 << (self.value - 1)

    @property
    def role_name(self) -> str:
        return {
            self.HAERACHE: "system/internal",
            self.HAE: "peer",
            self.HAEOYO: "standard-user",
            self.HASIPSIOCHE: "admin",
        }[self]

    def can_access(self, required: HonorificLevel) -> bool:
        """이 경어 수준이 필요한 수준에 접근할 수 있는지"""
        return self.capability_bit >= required.capability_bit


class HonorificError(Exception):
    """경어 수준 오류"""
    pass


class HonorificValidator:
    """경어 수준 검증기

    한국어 동사 어미를 분석하여 경어 수준을 감지하고,
    문장 간 경어 수준의 일관성을 검증한다.
    """

    # 동사 어미 → 경어 수준 매핑 (정규식)
    ENDING_PATTERNS: list[tuple[re.Pattern, HonorificLevel]] = [
        # 하십시오체: -ㅂ니다/습니다, -십시오, -(으)세요, -합니다
        (re.compile(r"(습니다|ㅂ니다|십시오|세요|으세요|합니다)$"), HonorificLevel.HASIPSIOCHE),
        # 해요체: -아요/어요/해요, -(이)에요
        (re.compile(r"(아요|어요|해요|이에요|예요)$"), HonorificLevel.HAEOYO),
        # 해체: -아/어/해 (명령형 종결 어미가 아닌 경우)
        (re.compile(r"(해|해라|거라|너라)$"), HonorificLevel.HAE),
        # 해라체: -다/ㄴ다/는다, -(으)라, -아라/어라
        (re.compile(r"(한다|이다|아라|어라|우라)$"), HonorificLevel.HAERACHE),
    ]

    # 동사 활용형 감지 (문장 중간에서)
    # 주의: 단어 중간의 '해'(예: 항해사)는 동사 어미가 아님
    CONJUGATION_PATTERNS: list[tuple[re.Pattern, HonorificLevel]] = [
        (re.compile(r"십시오|하십시오"), HonorificLevel.HASIPSIOCHE),
        (re.compile(r"해요|주세요|감사합니다"), HonorificLevel.HAEOYO),
        # '해'가 동사 어미로 쓰이는 패턴: 단어/공백 뒤에 오거나 문장 끝
        (re.compile(r"(?<![가-힣])해(?!요|라|서)[^가-힣]*$|반말"), HonorificLevel.HAE),
        (re.compile(r"(?<![가-힣])한다(?![가-힣])|(?<![가-힣])이다(?![가-힣])"), HonorificLevel.HAERACHE),
    ]

    # 종결 어미 목록 (문장 끝 감지)
    SENTENCE_ENDINGS = [
        "습니다", "ㅂ니다", "십시오", "세요",
        "아요", "어요", "해요", "이에요", "예요",
        "해", "해라", "거라", "너라",
        "한다", "이다", "아라", "어라",
    ]

    def __init__(self, default_level: HonorificLevel = HonorificLevel.HAERACHE) -> None:
        self.default_level = default_level
        self.detected_level: Optional[HonorificLevel] = None

    def detect_from_ending(self, sentence: str) -> Optional[HonorificLevel]:
        """문장의 동사 어미에서 경어 수준 감지"""
        stripped = sentence.rstrip(".,!? ")
        for pattern, level in self.ENDING_PATTERNS:
            if pattern.search(stripped):
                return level
        return None

    def detect_from_conjugation(self, sentence: str) -> Optional[HonorificLevel]:
        """문장 내 활용형에서 경어 수준 감지"""
        for pattern, level in self.CONJUGATION_PATTERNS:
            if pattern.search(sentence):
                return level
        return None

    def detect(self, sentence: str) -> HonorificLevel:
        """문장에서 경어 수준 감지 (우선순위: 어미 > 활용형 > 기본값)"""
        level = self.detect_from_ending(sentence)
        if level is not None:
            self.detected_level = level
            return level
        level = self.detect_from_conjugation(sentence)
        if level is not None:
            self.detected_level = level
            return level
        return self.default_level

    def validate_consistency(self, sentences: list[str]) -> tuple[bool, list[str]]:
        """여러 문장 간 경어 수준 일관성 검증

        Returns:
            (일관성 여부, 오류 메시지 목록)
        """
        levels = []
        errors = []
        for i, sentence in enumerate(sentences):
            level = self.detect(sentence)
            levels.append(level)
            if level != levels[0]:
                errors.append(
                    f"경어 불일치: 문장 {i + 1} \"{sentence[:20]}...\" "
                    f"({level.korean_name}) ≠ 문장 1 ({levels[0].korean_name})"
                )
        return len(errors) == 0, errors

    def get_cap_require_opcode(self, sentence: str) -> tuple:
        """문장의 경어 수준에 맞는 CAP_REQUIRE 옵코드 생성"""
        from flux_kor.vm import Opcode

        level = self.detect(sentence)
        return (Opcode.CAP_REQUIRE, level.value)

    def format_error(self, sentence: str, required: HonorificLevel, actual: HonorificLevel) -> str:
        """경어 오류 메시지를 한국어로 포맷"""
        return (
            f"경어 권한 오류: \"{sentence[:30]}...\" "
            f"요구 수준={required.korean_name}({required.role_name}), "
            f"현재 수준={actual.korean_name}({actual.role_name})"
        )
