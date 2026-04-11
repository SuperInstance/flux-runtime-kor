"""
조사(助詞) 시스템 — 한국어 후치사를 범위 연산자로

한국어 조사는 명사에 붙어 문법적 역할을 결정하는 후치사다.
이 모듈에서 조사를 프로그래밍의 범위(scope) 연산자로 모델링하여,
각 조사가 FLUX 바이트코드 시퀀스로 매핑된다.

조사 분류:
  은/는 (주제 조사)   → 현재 계산 스코프 정의
  이/가 (주어 조사)   → 활성 레지스터 (Actor)
  을/를 (목적어 조사) → 대상 레지스터 (Target)
  에/에게 (방향 조사) → 메시지 전송 목적지
  의 (소유 조사)      → 소유권 이전 (Ownership transfer)
  (으)로 (수단 조사)  → 함수 적용 (Function application)
  보다 (비교 조사)    → 비교 연산 (Comparison)

조사 적층 규칙 (Particle stacking):
  한국어에서는 문법적 제약으로 특정 조사 조합만 허용됨.
  예: "이가", "은는" 같은 모순 적층은 감지하여 오류를 발생시킴.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 조사 종류 열거형
# ═══════════════════════════════════════════════════════════════

class ParticleKind(IntEnum):
    """조사 종류 — 각 조사가 생성하는 바이트코드 시퀀스를 결정"""
    은는 = auto()    # 주제 조사 — 스코프 경계
    이가 = auto()    # 주어 조사 — 활성 레지스터
    을를 = auto()    # 목적어 조사 — 대상 레지스터
    에에게 = auto()  # 방향 조사 — 메시지 전송 목적지
    의 = auto()      # 소유 조사 — 소유권 이전
    으로 = auto()    # 수단 조사 — 함수 적용
    보다 = auto()    # 비교 조사 — 비교 연산
    에서 = auto()    # 출발 조사 — 소스 레지스터
    까지 = auto()    # 한계 조사 — 범위 상한
    과와 = auto()    # 병렬 조사 — 결합 연산
    만 = auto()      # 한정 조사 — 독점 접근
    도 = auto()      # 보조 조사 — 추가 연산
    마다 = auto()    # 분포 조사 — 반복/맵


# ═══════════════════════════════════════════════════════════════
# 조사 정의
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParticleDef:
    """조사 정의 — 단일 조사의 문법적·의미론적 정보

    Attributes:
        surface:     조사 표면형 (예: "은", "는")
        kind:        조사 종류
        description: 한국어 설명
        bytecode_hint: 이 조사가 매핑되는 바이트코드 연산 힌트
        consonant_final: 받침 있는 명사에 붙는지 여부
    """
    surface: str
    kind: ParticleKind
    description: str
    bytecode_hint: str
    consonant_final: bool  # True=받침 있음, False=받침 없음, None=양쪽 가능


# 모든 조사 정의
PARTICLE_DEFS: list[ParticleDef] = [
    # ── 주제 조사: 은/는 ──
    ParticleDef("은", ParticleKind.은는,
                "주제 조사 (받침 O) — 현재 계산 스코프의 주제를 선언",
                "SCOPE_BEGIN", consonant_final=True),
    ParticleDef("는", ParticleKind.은는,
                "주제 조사 (받침 X) — 현재 계산 스코프의 주제를 선언",
                "SCOPE_BEGIN", consonant_final=False),

    # ── 주어 조사: 이/가 ──
    ParticleDef("이", ParticleKind.이가,
                "주어 조사 (받침 O) — 연산의 주체를 활성 레지스터에 지정",
                "ACTIVATE_REG", consonant_final=True),
    ParticleDef("가", ParticleKind.이가,
                "주어 조사 (받침 X) — 연산의 주체를 활성 레지스터에 지정",
                "ACTIVATE_REG", consonant_final=False),

    # ── 목적어 조사: 을/를 ──
    ParticleDef("을", ParticleKind.을를,
                "목적어 조사 (받침 O) — 연산의 대상을 타겟 레지스터에 지정",
                "TARGET_REG", consonant_final=True),
    ParticleDef("를", ParticleKind.을를,
                "목적어 조사 (받침 X) — 연산의 대상을 타겟 레지스터에 지정",
                "TARGET_REG", consonant_final=False),

    # ── 방향 조사: 에/에게 ──
    ParticleDef("에게", ParticleKind.에에게,
                "방향 조사 (생명체) — 메시지 전송의 목적지를 지정",
                "SEND_DEST", consonant_final=None),
    ParticleDef("에", ParticleKind.에에게,
                "방향 조사 (장소/시간) — 데이터 저장의 목적지를 지정",
                "STORE_DEST", consonant_final=None),

    # ── 소유 조사: 의 ──
    ParticleDef("의", ParticleKind.의,
                "소유 조사 — 소유권 이전, 속성 접근, 필드 참조",
                "OWNERSHIP_TRANSFER", consonant_final=None),

    # ── 수단 조사: (으)로 ──
    ParticleDef("으로", ParticleKind.으로,
                "수단 조사 (받침 O/ㄹ) — 함수를 인자에 적용",
                "APPLY_FUNC", consonant_final=True),
    ParticleDef("로", ParticleKind.으로,
                "수단 조사 (받침 X) — 함수를 인자에 적용",
                "APPLY_FUNC", consonant_final=False),

    # ── 비교 조사: 보다 ──
    ParticleDef("보다", ParticleKind.보다,
                "비교 조사 — 두 값을 비교하여 플래그 설정",
                "CMP", consonant_final=None),

    # ── 출발 조사: 에서 ──
    ParticleDef("에서", ParticleKind.에서,
                "출발 조사 — 데이터의 소스를 지정",
                "LOAD_SRC", consonant_final=None),

    # ── 한계 조사: 까지 ──
    ParticleDef("까지", ParticleKind.까지,
                "한계 조사 — 범위의 상한을 지정 (반복/슬라이스)",
                "RANGE_UPPER", consonant_final=None),

    # ── 병렬 조사: 과/와 ──
    ParticleDef("과", ParticleKind.과와,
                "병렬 조사 (받침 O) — 두 값을 결합",
                "CONCAT", consonant_final=True),
    ParticleDef("와", ParticleKind.과와,
                "병렬 조사 (받침 X) — 두 값을 결합",
                "CONCAT", consonant_final=False),

    # ── 한정 조사: 만 ──
    ParticleDef("만", ParticleKind.만,
                "한정 조사 — 독점 접근, 다른 참조를 차단",
                "EXCLUSIVE_LOCK", consonant_final=None),

    # ── 보조 조사: 도 ──
    ParticleDef("도", ParticleKind.도,
                "보조 조사 — 기존 연산에 추가로 수행",
                "ALSO", consonant_final=None),

    # ── 분포 조사: 마다 ──
    ParticleDef("마다", ParticleKind.마다,
                "분포 조사 — 각 요소에 반복 적용 (map)",
                "MAP_EACH", consonant_final=None),
]

# 표면형 → 정의 매핑 (빠른 검색용)
_SURFACE_TO_DEF: dict[str, ParticleDef] = {
    p.surface: p for p in PARTICLE_DEFS
}

# 조사 종류별 바이트코드 시퀀스 매핑
PARTICLE_BYTECODE_MAP: dict[ParticleKind, list[str]] = {
    ParticleKind.은는:  ["SCOPE_PUSH"],
    ParticleKind.이가:  ["MOV", "ACTIVATE"],
    ParticleKind.을를:  ["MOV", "TARGET"],
    ParticleKind.에에게: ["PUSH", "SEND_PREP"],
    ParticleKind.의:    ["DEREF", "OWNERSHIP"],
    ParticleKind.으로:  ["PUSH_ARG", "CALL"],
    ParticleKind.보다:  ["CMP", "SET_FLAG"],
    ParticleKind.에서:  ["LOAD", "PUSH"],
    ParticleKind.까지:  ["SET_BOUND"],
    ParticleKind.과와:  ["CONCAT", "PUSH"],
    ParticleKind.만:    ["LOCK", "GUARD"],
    ParticleKind.도:    ["CLONE", "ALSO"],
    ParticleKind.마다:  ["ITER_BEGIN", "APPLY"],
}


# ═══════════════════════════════════════════════════════════════
# 조사 적층 규칙
# ═══════════════════════════════════════════════════════════════

# 같은 종류의 조사는 중복 적층 불가
_FORBIDDEN_STACK_SAME_KIND: set[ParticleKind] = {
    ParticleKind.은는,
    ParticleKind.이가,
    ParticleKind.을를,
    ParticleKind.에에게,
    ParticleKind.의,
    ParticleKind.으로,
    ParticleKind.보다,
}

# 허용되는 조사 적층 순서 (앞 조사 → 뒤 조사)
_ALLOWED_STACKING: list[tuple[ParticleKind, ParticleKind]] = [
    (ParticleKind.은는, ParticleKind.이가),
    (ParticleKind.은는, ParticleKind.을를),
    (ParticleKind.은는, ParticleKind.에에게),
    (ParticleKind.은는, ParticleKind.보다),
    (ParticleKind.은는, ParticleKind.에서),
    (ParticleKind.은는, ParticleKind.까지),
    (ParticleKind.이가, ParticleKind.을를),
    (ParticleKind.이가, ParticleKind.에에게),
    (ParticleKind.이가, ParticleKind.보다),
    (ParticleKind.이가, ParticleKind.으로),
    (ParticleKind.이가, ParticleKind.까지),
    (ParticleKind.이가, ParticleKind.도),
    (ParticleKind.이가, ParticleKind.만),
    (ParticleKind.을를, ParticleKind.에에게),
    (ParticleKind.을를, ParticleKind.보다),
    (ParticleKind.을를, ParticleKind.에서),
    (ParticleKind.을를, ParticleKind.까지),
    (ParticleKind.을를, ParticleKind.도),
    (ParticleKind.의, ParticleKind.이가),
    (ParticleKind.의, ParticleKind.을를),
    (ParticleKind.에서, ParticleKind.까지),
    (ParticleKind.에서, ParticleKind.까지),
    (ParticleKind.과와, ParticleKind.과와),
]


# ═══════════════════════════════════════════════════════════════
# 한글 받침 판정
# ═══════════════════════════════════════════════════════════════

def _has_final_consonant(syllable: str) -> bool | None:
    """음절의 받침 유무 판정

    Unicode 한글 음절에서 받침 여부를 확인.
    초성+중성+종성 구조에서 종성 위치(인덱스 2)의 값으로 판정.

    Args:
        syllable: 단일 한글 음절

    Returns:
        True (받침 있음), False (받침 없음), None (한글이 아닌 경우)
    """
    if not syllable:
        return None

    ch = syllable[-1]  # 마지막 글자 확인

    # 한글 유니코드 범위: AC00(가) ~ D7A3(힣)
    if not (0xAC00 <= ord(ch) <= 0xD7A3):
        return None

    # 종성 인덱스 계산: (코드 - 0xAC00) % 28
    # 0이면 받침 없음, 1~27이면 받침 있음
    jongseong_index = (ord(ch) - 0xAC00) % 28
    return jongseong_index > 0


# ═══════════════════════════════════════════════════════════════
# 조사 첨부 (명사 + 조사 → 조사 붙은 형태)
# ═══════════════════════════════════════════════════════════════

def attach_particle(noun: str, particle_surface: str) -> str:
    """명사에 알맞은 조사를 첨부

    받침 유무에 따라 은/는, 이/가 등을 자동 선택.
    이미 올바른 조사가 붙어있으면 그대로 반환.

    Args:
        noun: 명사 (한글 또는 영문)
        particle_surface: 조사 표면형 (예: "은/는", "이/가")

    Returns:
        조사가 붙은 형태의 문자열
    """
    # 슬래시 구분 처리: "은/는" → 받침에 따라 선택
    if "/" in particle_surface:
        variants = particle_surface.split("/")
        if len(variants) == 2:
            has_fin = _has_final_consonant(noun)
            if has_fin is True:
                chosen = variants[0]  # 받침 O 형태
            elif has_fin is False:
                chosen = variants[1]  # 받침 X 형태
            else:
                # 한글이 아닌 경우 (영문 등): 기본형 사용
                chosen = variants[0]
            return f"{noun}{chosen}"
        return f"{noun}{particle_surface}"

    return f"{noun}{particle_surface}"


# ═══════════════════════════════════════════════════════════════
# 조사 토큰
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParticleToken:
    """조사 토큰 — 텍스트에서 추출된 조사 정보

    Attributes:
        surface:   조사 표면형
        kind:      조사 종류
        noun:      조사가 붙은 명사 (조사 제외)
        position:  텍스트 내 시작 위치
        full_text: 명사 + 조사 전체 텍스트
    """
    surface: str
    kind: ParticleKind
    noun: str
    position: int
    full_text: str

    def __repr__(self) -> str:
        return f"ParticleToken({self.full_text!r} → {self.kind.name})"


# ═══════════════════════════════════════════════════════════════
# 조사 분석기
# ═══════════════════════════════════════════════════════════════

class ParticleAnalyzer:
    """조사 분석기 — 한국어 텍스트에서 조사를 추출하고 분석

    긴 조사(에게, 에서, 까지, 마다, 으로)를 우선 매칭하여
    짧은 조사(에, 서, 다, 로)의 오인식을 방지.
    """

    # 매칭 순서가 중요: 긴 조사를 먼저 검사
    _SORTED_PARTICLES: list[str] = sorted(
        _SURFACE_TO_DEF.keys(), key=len, reverse=True
    )

    def __init__(self) -> None:
        self._tokens: list[ParticleToken] = []

    def analyze(self, text: str) -> list[ParticleToken]:
        """텍스트에서 조사를 추출

        긴 조사 우선 매칭 (최장 일치 원칙).
        이미 매칭된 위치는 건너뜀.

        Args:
            text: 한국어 텍스트

        Returns:
            추출된 조사 토큰 목록 (텍스트 순서)
        """
        self._tokens = []
        matched_positions: set[int] = set()

        for i in range(len(text)):
            if i in matched_positions:
                continue

            # 각 위치에서 가장 긴 조사를 찾음
            best_match: Optional[tuple[str, ParticleDef]] = None
            best_end = i

            for particle_surface in self._SORTED_PARTICLES:
                p_len = len(particle_surface)
                end = i + p_len

                if end > len(text):
                    continue

                # 모든 위치가 이미 매칭된 건지 확인
                if any(pos in matched_positions for pos in range(i, end)):
                    continue

                if text[i:end] == particle_surface:
                    pdef = _SURFACE_TO_DEF[particle_surface]
                    best_match = (particle_surface, pdef)
                    best_end = end
                    break  # 가장 긴 것을 찾으면 바로 종료

            if best_match is not None:
                surface, pdef = best_match
                noun = text[:i].rstrip()
                # 명사가 비어있으면 빈 문자열
                if noun and self._tokens:
                    pass  # 이전 토큰의 뒤에 이어지는 명사
                full_text = noun + surface if noun else surface

                token = ParticleToken(
                    surface=surface,
                    kind=pdef.kind,
                    noun=noun,
                    position=i,
                    full_text=full_text,
                )
                self._tokens.append(token)

                for pos in range(i, best_end):
                    matched_positions.add(pos)

        return self._tokens

    def get_bytecode_hints(self) -> list[str]:
        """추출된 조사의 바이트코드 힌트 목록 반환"""
        hints: list[str] = []
        for token in self._tokens:
            bc_seq = PARTICLE_BYTECODE_MAP.get(token.kind, [])
            hints.extend(bc_seq)
        return hints

    def get_particle_kinds(self) -> list[ParticleKind]:
        """추출된 조사 종류 목록 반환"""
        return [t.kind for t in self._tokens]


# ═══════════════════════════════════════════════════════════════
# 조사 적층 검증기
# ═══════════════════════════════════════════════════════════════

class ParticleStackError(Exception):
    """조사 적층 오류 — 문법적으로 허용되지 않는 조사 조합"""
    pass


class ParticleStack:
    """조사 스택 — 조사 적층 순서를 검증하고 관리

    한국어의 조사 적층 규칙을 적용하여,
    문법적으로 올바른 조사 조합인지 검증.

    예시 (올바름):
      "값이"        → [이가]
      "값을"        → [을를]
      "값이 범위까지" → [이가, 까지]

    예시 (오류):
      "값이가"       → 같은 종류 중복
      "값을에서"     → 허용되지 않는 순서
    """

    def __init__(self) -> None:
        self._stack: list[tuple[str, ParticleKind]] = []

    @property
    def current(self) -> Optional[ParticleKind]:
        """스택 최상단 조사 종류"""
        if self._stack:
            return self._stack[-1][1]
        return None

    @property
    def kinds(self) -> list[ParticleKind]:
        """스택 내 모든 조사 종류"""
        return [kind for _, kind in self._stack]

    def push(self, surface: str, kind: ParticleKind) -> None:
        """조사를 스택에 푸시

        Args:
            surface: 조사 표면형
            kind: 조사 종류

        Raises:
            ParticleStackError: 적층 규칙 위반 시
        """
        # 1. 같은 종류 중복 적층 금지
        if kind in _FORBIDDEN_STACK_SAME_KIND:
            for _, existing_kind in self._stack:
                if existing_kind == kind:
                    raise ParticleStackError(
                        f"조사 중복 적층: '{surface}'({kind.name})은/는 "
                        f"이미 스택에 있는 '{existing_kind.name}'과 같은 종류입니다"
                    )

        # 2. 적층 순서 검증
        if self.current is not None:
            allowed = False
            for prev_kind, next_kind in _ALLOWED_STACKING:
                if prev_kind == self.current and next_kind == kind:
                    allowed = True
                    break

            if not allowed:
                raise ParticleStackError(
                    f"허용되지 않는 조사 적층 순서: "
                    f"'{self.current.name}' 뒤에 '{kind.name}'을/를 올릴 수 없습니다"
                )

        self._stack.append((surface, kind))

    def pop(self) -> tuple[str, ParticleKind]:
        """스택에서 조사를 팝"""
        if not self._stack:
            raise ParticleStackError("조사 스택이 비어 있습니다")
        return self._stack.pop()

    def clear(self) -> None:
        """스택 비우기"""
        self._stack.clear()

    def to_bytecode_sequence(self) -> list[str]:
        """스택의 조사들을 바이트코드 시퀀스로 변환"""
        sequence: list[str] = []
        for surface, kind in self._stack:
            bc_ops = PARTICLE_BYTECODE_MAP.get(kind, [])
            sequence.extend(bc_ops)
        return sequence

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        parts = [f"{s}({k.name})" for s, k in self._stack]
        return f"ParticleStack[{', '.join(parts)}]"


# ═══════════════════════════════════════════════════════════════
# 조사 → 레지스터 매핑
# ═══════════════════════════════════════════════════════════════

class ParticleRegisterMapper:
    """조사를 레지스터 할당에 매핑

    한국어 문장에서 조사가 붙은 명사를 레지스터에 할당.
    SOV 구조에서 조사가 명사의 역할을 결정하므로,
    조사 종류에 따라 레지스터 할당 전략이 달라짐.

    매핑 규칙:
      이/가 (주어)  → 다음 사용 가능한 소스 레지스터
      을/를 (목적어) → 다음 사용 가능한 타겟 레지스터
      에/에게 (방향) → 목적지 레지스터 (고정)
    """

    # 조사 종류별 레지스터 할당 범위
    SUBJECT_REG_RANGE = range(0, 4)     # R0 ~ R3 (소스)
    OBJECT_REG_RANGE = range(4, 8)      # R4 ~ R7 (타겟)
    DEST_REG = 7                         # R7 (목적지, 고정)

    def __init__(self) -> None:
        self._subject_idx = 0
        self._object_idx = 0
        self._assignments: dict[str, int] = {}

    def assign(self, noun: str, kind: ParticleKind) -> int:
        """명사를 조사 종류에 따라 레지스터에 할당

        Args:
            noun: 명사 식별자
            kind: 조사 종류

        Returns:
            할당된 레지스터 인덱스
        """
        if noun in self._assignments:
            return self._assignments[noun]

        if kind == ParticleKind.이가:
            reg = self.SUBJECT_REG_RANGE[self._subject_idx % len(self.SUBJECT_REG_RANGE)]
            self._subject_idx += 1
        elif kind == ParticleKind.을를:
            reg = self.OBJECT_REG_RANGE[self._object_idx % len(self.OBJECT_REG_RANGE)]
            self._object_idx += 1
        elif kind == ParticleKind.에에게:
            reg = self.DEST_REG
        else:
            # 기본: 순차 할당
            reg = (self._subject_idx + self._object_idx) % 8
            self._subject_idx += 1

        self._assignments[noun] = reg
        return reg

    def get_assignment(self, noun: str) -> Optional[int]:
        """이미 할당된 레지스터 조회"""
        return self._assignments.get(noun)

    def clear(self) -> None:
        """할당 상태 초기화"""
        self._subject_idx = 0
        self._object_idx = 0
        self._assignments.clear()

    @property
    def assignments(self) -> dict[str, int]:
        """현재 할당 맵 (읽기 전용)"""
        return dict(self._assignments)
