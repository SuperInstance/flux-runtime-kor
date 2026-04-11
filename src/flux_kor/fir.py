"""
FIR (Flux Intermediate Representation) SSA 빌더 — 한국어 NL 소스용

한국어 자연어 소스를 SSA(Static Single Assignment) 형태의 FIR로 변환.
한국어 문법 구조를 IR 설계에 직접 반영:

  1. SOV → 연속 트리(Continuation Tree)
     주어-목적어 순서를 데이터 흐름 그래프의 위쪽 노드로,
     동사를 종결 노드(continuation)로 매핑.

  2. 조사 → 변수 스코핑
     은/는(주제) → 스코프 경계, 이/가(주어) → 정의,
     을/를(목적어) → 사용, 의(소유) → SSA phi 노드.

  3. 경어 → 타입 수준
     하십시오체 → admin_t, 해요체 → user_t, 해체 → peer_t, 해라체 → sys_t.
     경어 수준이 변수의 타입 안전성을 결정.

  4. 기본 블록(Basic Block) → 한국어 레이블
     블록의 terminator를 한국어 동사 레이블로 표현:
     "종료", "건너뛰기", "반복", "분기".

SSA 형식의 핵심:
  모든 변수는 정확히 한 번만 대입(assignment)된다.
  제어 흐름이 합류하는 지점에서 phi(φ) 노드로 값을 병합.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
# FIR 타입 시스템
# ═══════════════════════════════════════════════════════════════

class FirType(IntEnum):
    """FIR 타입 — 경어 수준에 기반한 타입 시스템

    한국어의 경어 체계를 타입의 신뢰 수준으로 모델링.
    높은 경어 수준의 변수는 더 강한 타입 검증을 받음.
    """
    정수 = auto()       # 정수형 — 기본 수치 타입
    실수 = auto()       # 실수형
    문자열 = auto()     # 문자열형
    논리 = auto()       # 논리형
    함수 = auto()       # 함수형
    메시지 = auto()     # 메시지형 — 에이전트 간 통신
    에이전트 = auto()   # 에이전트 참조형

    # 경어 기반 타입 수준
    시스템 = auto()     # sys_t (해라체) — 내부 시스템 데이터
    동료 = auto()       # peer_t (해체) — 동료 간 공유 데이터
    사용자 = auto()     # user_t (해요체) — 사용자 데이터
    관리자 = auto()     # admin_t (하십시오체) — 관리자 데이터
    알수없음 = auto()   # 미정 타입


# 경어 수준 → FIR 타입 매핑
HONORIFIC_TO_FIR_TYPE: dict[int, FirType] = {
    1: FirType.시스템,
    2: FirType.동료,
    3: FirType.사용자,
    4: FirType.관리자,
}

# FIR 타입 → 신뢰 수준 (높을수록 더 신뢰됨)
TYPE_TRUST_LEVEL: dict[FirType, int] = {
    FirType.시스템: 1,
    FirType.동료: 2,
    FirType.사용자: 3,
    FirType.관리자: 4,
    FirType.정수: 2,
    FirType.실수: 2,
    FirType.문자열: 2,
    FirType.논리: 2,
    FirType.함수: 3,
    FirType.메시지: 3,
    FirType.에이전트: 4,
    FirType.알수없음: 0,
}


# ═══════════════════════════════════════════════════════════════
# FIR 옵코드
# ═══════════════════════════════════════════════════════════════

class FirOp(IntEnum):
    """FIR 연산 코드 — SSA 형식의 중간 표현용"""
    # 상수/변수
    상수 = auto()       # const — 즉시값
    정의 = auto()       # def — SSA 변수 정의
    # 산술
    덧셈 = auto()       # add
    뺄셈 = auto()       # sub
    곱셈 = auto()       # mul
    나눗셈 = auto()     # div
    나머지 = auto()     # mod
    부정 = auto()       # neg
    # 비교
    비교 = auto()       # cmp — 결과를 플래그에 저장
    # 데이터
    이동 = auto()       # mov — 값 복사
    로드 = auto()       # load — 메모리 읽기
    저장 = auto()       # store — 메모리 쓰기
    # 제어 흐름
    점프 = auto()       # jmp — 무조건 점프
    조건점프 = auto()   # jcc — 조건부 점프
    호출 = auto()       # call — 함수 호출
    반환 = auto()       # ret — 함수 반환
    # SSA 특수
    파이 = auto()       # phi — SSA phi 노드
    # 경어/권한
    권한요구 = auto()   # cap_require
    # 통신
    전달 = auto()       # tell
    질문 = auto()       # ask
    위임 = auto()       # delegate
    방송 = auto()       # broadcast
    # 기타
    출력 = auto()       # print
    종료 = auto()       # halt


# 옵코드 → 한국어 이름
FIR_OP_NAMES: dict[FirOp, str] = {
    FirOp.상수: "상수",
    FirOp.정의: "정의",
    FirOp.덧셈: "더하기",
    FirOp.뺄셈: "빼기",
    FirOp.곱셈: "곱하기",
    FirOp.나눗셈: "나누기",
    FirOp.나머지: "나머지",
    FirOp.부정: "부정",
    FirOp.비교: "비교",
    FirOp.이동: "이동",
    FirOp.로드: "불러오기",
    FirOp.저장: "저장",
    FirOp.점프: "건너뛰기",
    FirOp.조건점프: "조건분기",
    FirOp.호출: "호출",
    FirOp.반환: "반환",
    FirOp.파이: "φ(파이)",
    FirOp.권한요구: "권한요구",
    FirOp.전달: "전달",
    FirOp.질문: "질문",
    FirOp.위임: "위임",
    FirOp.방송: "방송",
    FirOp.출력: "출력",
    FirOp.종료: "종료",
}


# ═══════════════════════════════════════════════════════════════
# SSA 변수
# ═══════════════════════════════════════════════════════════════

@dataclass
class FirValue:
    """SSA 값 — 단일 대입 변수

    SSA에서 모든 변수는 정확히 한 번만 정의됨.
    버전 번호를 통해 동일 이름의 여러 정의를 구분.

    Attributes:
        name: 변수 기본 이름 (예: "x", "값", "R영")
        version: SSA 버전 번호 (0, 1, 2, ...)
        fir_type: FIR 타입
        honorific_level: 정의 시점의 경어 수준 (타입 추론에 사용)
    """
    name: str
    version: int
    fir_type: FirType = FirType.알수없음
    honorific_level: int = 1

    @property
    def ssa_name(self) -> str:
        """SSA 형식 이름 — name.version"""
        return f"{self.name}.{self.version}"

    def __repr__(self) -> str:
        type_str = self.fir_type.name
        return f"{self.ssa_name}:{type_str}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FirValue):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        return hash((self.name, self.version))


# ═══════════════════════════════════════════════════════════════
# FIR 명령어
# ═══════════════════════════════════════════════════════════════

@dataclass
class FirInstr:
    """FIR 명령어 — SSA 형태의 중간 명령어

    Attributes:
        op: 연산 코드
        result: 결과값 (SSA 변수), 없으면 None
        operands: 피연산자 목록 (SSA 변수 또는 상수)
        label: 옵션 레이블 (주석/디버그용)
        source_line: 원시 소스 줄
    """
    op: FirOp
    result: Optional[FirValue] = None
    operands: list[Any] = field(default_factory=list)
    label: str = ""
    source_line: str = ""

    @property
    def is_terminator(self) -> bool:
        """기본 블록 종결 명령어인지"""
        return self.op in {
            FirOp.점프,
            FirOp.조건점프,
            FirOp.반환,
            FirOp.종료,
        }

    @property
    def is_phi(self) -> bool:
        """phi 노드인지"""
        return self.op == FirOp.파이

    def __repr__(self) -> str:
        op_name = FIR_OP_NAMES.get(self.op, str(self.op))
        if self.result:
            operands_str = ", ".join(str(o) for o in self.operands)
            return f"  {self.result.ssa_name} = {op_name} {operands_str}"
        else:
            operands_str = ", ".join(str(o) for o in self.operands)
            return f"  {op_name} {operands_str}"


# ═══════════════════════════════════════════════════════════════
# 기본 블록 (Basic Block)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BasicBlock:
    """FIR 기본 블록 — 한국어 레이블이 지정된 명령어 시퀀스

    Attributes:
        label: 블록 레이블 (한국어)
        instructions: 블록 내 명령어 목록
        predecessors: 선행 블록 레이블 목록
        successors: 후속 블록 레이블 목록
    """
    label: str
    instructions: list[FirInstr] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)

    @property
    def terminator(self) -> Optional[FirInstr]:
        """블록의 종결 명령어"""
        for instr in reversed(self.instructions):
            if instr.is_terminator:
                return instr
        return None

    @property
    def phi_nodes(self) -> list[FirInstr]:
        """블록의 phi 노드 목록"""
        return [i for i in self.instructions if i.is_phi]

    def add_instruction(self, instr: FirInstr) -> None:
        """명령어 추가"""
        self.instructions.append(instr)

    def add_predecessor(self, label: str) -> None:
        """선행 블록 추가"""
        if label not in self.predecessors:
            self.predecessors.append(label)

    def add_successor(self, label: str) -> None:
        """후속 블록 추가"""
        if label not in self.successors:
            self.successors.append(label)

    def __repr__(self) -> str:
        return f"기본블록({self.label}, 명령어={len(self.instructions)}개)"


# ═══════════════════════════════════════════════════════════════
# 연속 트리 (Continuation Tree)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContinuationNode:
    """연속 트리 노드 — SOV 구조를 트리로 표현

    한국어 SOV(주어-목적어-동사) 구조를 트리 형태로 변환.
    동사가 continuation(종결 노드)이고, 주어/목적어가 데이터 노드.

    Attributes:
        text: 노드 텍스트
        role: 문법적 역할 (주어, 목적어, 동사, 부사, 조사)
        children: 자식 노드 목록
        particle: 이 노드에 붙은 조사
        honorific_level: 경어 수준
    """
    text: str
    role: str = ""
    children: list[ContinuationNode] = field(default_factory=list)
    particle: str = ""
    honorific_level: int = 1

    @property
    def is_continuation(self) -> bool:
        """이 노드가 continuation(동사)인지"""
        return self.role == "동사"

    @property
    def is_data(self) -> bool:
        """이 노드가 데이터 노드(주어/목적어)인지"""
        return self.role in ("주어", "목적어", "주제")

    def flatten_data_nodes(self) -> list[ContinuationNode]:
        """데이터 노드를 순서대로 평탄화 (SOV 순서 유지)"""
        result: list[ContinuationNode] = []
        for child in self.children:
            if child.is_data:
                result.append(child)
            result.extend(child.flatten_data_nodes())
        return result

    def __repr__(self) -> str:
        particle_str = f"[{self.particle}]" if self.particle else ""
        return f"{self.role}({self.text}{particle_str})"


# ═══════════════════════════════════════════════════════════════
# FIR 빌더
# ═══════════════════════════════════════════════════════════════

class FirBuildError(Exception):
    """FIR 빌드 오류"""
    pass


class FirBuilder:
    """FIR SSA 빌더 — 한국어 NL 소스에서 SSA 형태의 FIR을 구성

    빌드 과정:
      1. 소스 토큰화 (한국어 형태소 단위 분리)
      2. SOV → 연속 트리 변환
      3. 기본 블록 구성
      4. SSA 변수 관리 및 phi 노드 삽입
      5. 경어 기반 타입 추론

    사용 예시::

        builder = FirBuilder()
        fir = builder.build("값이 삼입니다\\n값이 오입니다\\n값을 더하십시오")
        print(fir.format())
    """

    # ── 한자어 숫자 맵 ──
    _HANJA_NUMS: dict[str, int] = {
        "영": 0, "공": 0, "일": 1, "이": 2, "삼": 3,
        "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
        "십": 10, "백": 100, "천": 1000,
    }

    # ── 조사 패턴 ──
    _PARTICLE_PATTERNS: list[tuple[str, str]] = [
        ("에게", "에게"), ("에서", "에서"), ("까지", "까지"),
        ("으로", "으로"), ("보다", "보다"), ("마다", "마다"),
        ("에는", "은"), ("는은", "는"),
        ("이가", "이가"), ("을를", "을를"),
        ("의의", "의"),
        ("은", "은"), ("는", "는"),
        ("이", "이"), ("가", "가"),
        ("을", "을"), ("를", "를"),
        ("에", "에"), ("의", "의"),
        ("로", "로"), ("도", "도"), ("만", "만"),
    ]

    # ── 동사 패턴 ──
    _VERB_ENDINGS: list[str] = [
        "하십시오", "합니다", "습니다", "ㅂ니다", "세요", "으세요",
        "해요", "아요", "어요", "예요", "이에요",
        "한다", "이다", "해라", "거라", "너라",
        "해", "아", "어", "다", "고", "서",
    ]

    def __init__(self) -> None:
        # SSA 변수 관리
        self._ssa_counters: dict[str, int] = {}
        self._ssa_values: dict[str, list[FirValue]] = {}

        # 기본 블록
        self._blocks: dict[str, BasicBlock] = {}
        self._block_order: list[str] = []
        self._current_block: Optional[str] = None

        # 빌드 상태
        self._continuation_tree: Optional[ContinuationNode] = None
        self._honorific_level: int = 1
        self._errors: list[str] = []

    # ── SSA 변수 관리 ──

    def _new_ssa_value(
        self,
        name: str,
        fir_type: FirType = FirType.알수없음,
    ) -> FirValue:
        """새 SSA 변수 생성 (버전 증가)

        Args:
            name: 변수 기본 이름
            fir_type: FIR 타입

        Returns:
            새 SSA 값
        """
        version = self._ssa_counters.get(name, 0)
        self._ssa_counters[name] = version + 1

        value = FirValue(
            name=name,
            version=version,
            fir_type=fir_type,
            honorific_level=self._honorific_level,
        )

        if name not in self._ssa_values:
            self._ssa_values[name] = []
        self._ssa_values[name].append(value)

        return value

    def _get_latest_value(self, name: str) -> Optional[FirValue]:
        """변수의 가장 최근 SSA 값을 반환"""
        values = self._ssa_values.get(name, [])
        return values[-1] if values else None

    def _get_all_versions(self, name: str) -> list[FirValue]:
        """변수의 모든 SSA 버전 반환"""
        return self._ssa_values.get(name, [])

    # ── 기본 블록 관리 ──

    def _create_block(self, label: str) -> BasicBlock:
        """새 기본 블록 생성

        Args:
            label: 블록 레이블 (한국어)

        Returns:
            생성된 기본 블록
        """
        if label in self._blocks:
            return self._blocks[label]

        block = BasicBlock(label=label)
        self._blocks[label] = block

        if label not in self._block_order:
            self._block_order.append(label)

        return block

    def _current(self) -> BasicBlock:
        """현재 블록 반환"""
        if self._current_block is None:
            self._current_block = "시작"
            return self._create_block("시작")
        return self._blocks[self._current_block]

    def _switch_block(self, label: str) -> BasicBlock:
        """현재 블록 전환"""
        self._current_block = label
        return self._create_block(label)

    def _emit(self, instr: FirInstr) -> None:
        """현재 블록에 명령어 추가"""
        self._current().add_instruction(instr)

    # ── 토큰화 ──

    def _tokenize(self, text: str) -> list[str]:
        """한국어 소스를 토큰 단위로 분리

        조사, 동사 어미, 명사를 분리하여 토큰 목록 생성.

        Args:
            text: 한국어 소스 텍스트

        Returns:
            토큰 목록
        """
        tokens: list[str] = []

        # 줄 단위 처리
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 조사 분리 (긴 조사 우선)
            remaining = line
            while remaining:
                matched = False
                for particle, role in self._PARTICLE_PATTERNS:
                    if remaining.endswith(particle):
                        # 조사 앞의 명사 추출
                        noun_part = remaining[:-len(particle)].strip()
                        if noun_part:
                            tokens.append(noun_part)
                        tokens.append(particle)
                        remaining = ""
                        matched = True
                        break

                if matched:
                    break

                # 동사 어미 매칭
                verb_found = False
                for ending in self._VERB_ENDINGS:
                    if remaining.endswith(ending):
                        base = remaining[:-len(ending)].strip()
                        if base:
                            tokens.append(base)
                        tokens.append(ending)
                        remaining = ""
                        verb_found = True
                        break

                if verb_found:
                    break

                # 공백 분리
                parts = remaining.split(None, 1)
                if parts:
                    tokens.append(parts[0])
                    remaining = parts[1] if len(parts) > 1 else ""
                else:
                    break

        return tokens

    # ── SOV → 연속 트리 변환 ──

    def _build_continuation_tree(self, tokens: list[str]) -> ContinuationNode:
        """토큰 목록에서 SOV 연속 트리 구성

        한국어 SOV 구조를 분석하여 트리 형태로 변환.
        동사(마지막 요소)가 continuation 루트.

        Args:
            tokens: 토큰 목록

        Returns:
            연속 트리 루트 노드
        """
        if not tokens:
            return ContinuationNode(text="", role="빈")

        # 동사 식별 (마지막 토큰이 동사 어미이면 동사)
        last_token = tokens[-1]
        is_verb = any(last_token.endswith(e) for e in self._VERB_ENDINGS)

        if is_verb:
            verb_node = ContinuationNode(
                text=last_token,
                role="동사",
                honorific_level=self._honorific_level,
            )
            # 나머지 토큰을 데이터 노드로 추가
            data_tokens = tokens[:-1]
            i = 0
            while i < len(data_tokens):
                token = data_tokens[i]
                role = self._classify_token(token)
                particle = ""

                # 다음 토큰이 조사인지 확인
                if i + 1 < len(data_tokens):
                    next_token = data_tokens[i + 1]
                    if self._is_particle(next_token):
                        particle = next_token
                        i += 1  # 조사 토큰 스킵

                child = ContinuationNode(
                    text=token,
                    role=role,
                    particle=particle,
                    honorific_level=self._honorific_level,
                )
                verb_node.children.append(child)
                i += 1

            return verb_node
        else:
            # 동사를 찾지 못한 경우 — 전체를 단일 노드로
            return ContinuationNode(
                text=" ".join(tokens),
                role="미분류",
                honorific_level=self._honorific_level,
            )

    def _classify_token(self, token: str) -> str:
        """토큰의 문법적 역할 분류

        Args:
            token: 토큰 문자열

        Returns:
            역할 이름 (주어, 목적어, 주제, 부사, 수식어)
        """
        # 한자어 숫자인지 확인
        if token in self._HANJA_NUMS:
            return "수식어"

        # 아라비아 숫자인지 확인
        if re.fullmatch(r'-?[0-9]+', token):
            return "수식어"

        # 레지스터 참조인지 확인
        if token.startswith("R") or token in ("영", "일", "이", "삼",
                                              "사", "오", "육", "칠", "팔", "구"):
            return "수식어"

        # 기본값: 목적어 (SOV에서 주어/목적어 구분은 조사에 의함)
        return "목적어"

    def _is_particle(self, token: str) -> bool:
        """토큰이 조사인지 확인"""
        particles = set(p for p, _ in self._PARTICLE_PATTERNS)
        return token in particles

    # ── 경어 감지 ──

    def _detect_honorific(self, text: str) -> int:
        """텍스트에서 경어 수준 감지

        Args:
            text: 한국어 텍스트

        Returns:
            경어 수준 (1~4)
        """
        # 하십시오체 패턴
        formal_patterns = [
            r"습니다$", r"ㅂ니다$", r"십시오$", r"하세요$",
            r"으세요$", r"합니다$",
        ]
        for pat in formal_patterns:
            if re.search(pat, text):
                self._honorific_level = 4
                return 4

        # 해요체 패턴
        polite_patterns = [r"해요$", r"아요$", r"어요$", r"예요$", r"이에요$"]
        for pat in polite_patterns:
            if re.search(pat, text):
                self._honorific_level = 3
                return 3

        # 해체 패턴
        intimate_patterns = [r"해$"]
        for pat in intimate_patterns:
            if re.search(pat, text):
                self._honorific_level = 2
                return 2

        # 해라체 (기본)
        self._honorific_level = 1
        return 1

    # ── 타입 추론 ──

    def _infer_type(self, token: str, honorific_level: int = 1) -> FirType:
        """토큰에서 FIR 타입 추론

        경어 수준에 따라 타입 신뢰도를 결정.
        한자어 숫자 → 정수, "참/거짓" → 논리, 기타 → 경어 타입.

        Args:
            token: 토큰 문자열
            honorific_level: 경어 수준

        Returns:
            추론된 FIR 타입
        """
        # 한자어/아라비아 숫자
        if token in self._HANJA_NUMS or re.fullmatch(r'-?[0-9]+', token):
            return FirType.정수

        # 논리값
        if token in ("참", "거짓", "true", "false", "True", "False"):
            return FirType.논리

        # 문자열 리터럴
        if token.startswith('"') or token.startswith("'"):
            return FirType.문자열

        # 경어 기반 타입
        honorific_type = HONORIFIC_TO_FIR_TYPE.get(honorific_level)
        if honorific_type:
            return honorific_type

        return FirType.알수없음

    # ── phi 노드 생성 ──

    def _create_phi(
        self,
        name: str,
        block: BasicBlock,
        sources: list[tuple[str, str]],
    ) -> FirInstr:
        """phi 노드 생성

        여러 제어 흐름이 합류하는 지점에서 변수 값을 병합.
        조사 기반 스코핑: 각 합류 경로의 조사 컨텍스트를 보존.

        Args:
            name: 변수 이름
            block: phi가 속한 블록
            sources: (블록_레이블, SSA_값_이름) 쌍 목록

        Returns:
            phi 명령어
        """
        # 새 SSA 값 생성
        phi_value = self._new_ssa_value(name)

        # 피연산자 구성
        operands: list[Any] = []
        for src_block, src_value in sources:
            operands.append((src_block, src_value))

        phi_instr = FirInstr(
            op=FirOp.파이,
            result=phi_value,
            operands=operands,
            label=f"φ({name})",
        )

        # phi 노드는 블록의 맨 앞에 삽입
        block.instructions.insert(0, phi_instr)

        return phi_instr

    # ── 빌드 ──

    def build(self, source: str) -> FirModule:
        """한국어 NL 소스에서 FIR 모듈 구성

        Args:
            source: 한국어 자연어 소스 코드

        Returns:
            구성된 FIR 모듈

        Raises:
            FirBuildError: 빌드 오류 발생 시
        """
        self._errors.clear()
        self._ssa_counters.clear()
        self._ssa_values.clear()
        self._blocks.clear()
        self._block_order.clear()
        self._current_block = None
        self._honorific_level = 1

        lines = source.strip().split("\n")

        # 시작 블록
        self._switch_block("시작")

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 경어 수준 감지
            honorific = self._detect_honorific(line)
            self._honorific_level = honorific

            # 토큰화
            tokens = self._tokenize(line)

            # 연속 트리 구성
            tree = self._build_continuation_tree(tokens)
            self._continuation_tree = tree

            # FIR 명령어 생성
            self._tree_to_fir(tree, line, line_num)

        # 종료 명령어 (없으면 자동 추가)
        current = self._current()
        if not current.terminator:
            self._emit(FirInstr(
                op=FirOp.종료,
                label="자동종료",
            ))

        # 모듈 구성
        module = FirModule(
            blocks=list(self._blocks.values()),
            block_order=self._block_order,
            ssa_values=self._ssa_values,
        )

        if self._errors:
            raise FirBuildError("\n".join(self._errors))

        return module

    def _tree_to_fir(
        self,
        node: ContinuationNode,
        source_line: str,
        line_num: int,
    ) -> None:
        """연속 트리 노드에서 FIR 명령어 생성

        Args:
            node: 연속 트리 노드
            source_line: 원시 소스 줄
            line_num: 줄 번호
        """
        # 데이터 노드를 먼저 처리 (SOV 순서)
        data_nodes = node.flatten_data_nodes()

        # 경어 수준에 따른 권한 삽입
        if self._honorific_level > 1:
            cap_value = self._new_ssa_value("cap", FirType.시스템)
            self._emit(FirInstr(
                op=FirOp.권한요구,
                operands=[self._honorific_level],
                label=f"경어_L{self._honorific_level}",
                source_line=source_line,
            ))

        # 데이터 노드 처리 → SSA 변수 정의
        defined_values: list[FirValue] = []
        for data_node in data_nodes:
            value = self._process_data_node(data_node, source_line, line_num)
            if value:
                defined_values.append(value)

        # continuation (동사) 노드 처리
        if node.is_continuation:
            self._process_verb_node(node, defined_values, source_line, line_num)

    def _process_data_node(
        self,
        node: ContinuationNode,
        source_line: str,
        line_num: int,
    ) -> Optional[FirValue]:
        """데이터 노드를 FIR 상수/정의 명령어로 변환

        Args:
            node: 데이터 노드
            source_line: 원시 소스
            line_num: 줄 번호

        Returns:
            생성된 SSA 값 (없으면 None)
        """
        text = node.text
        particle = node.particle

        # 변수 이름 결정
        var_name = text
        if particle in ("이", "가", "은", "는"):
            var_name = text
        elif particle in ("을", "를"):
            var_name = text
        else:
            var_name = text

        # 타입 추론
        fir_type = self._infer_type(text, self._honorific_level)

        # 상수인지 변수인지 판별
        if text in self._HANJA_NUMS:
            # 한자어 숫자 상수
            value = self._new_ssa_value(f"_상수_{text}", FirType.정수)
            self._emit(FirInstr(
                op=FirOp.상수,
                result=value,
                operands=[self._HANJA_NUMS[text]],
                label=f"한자어:{text}={self._HANJA_NUMS[text]}",
                source_line=source_line,
            ))
            return value
        elif re.fullmatch(r'-?[0-9]+', text):
            # 아라비아 숫자 상수
            int_val = int(text)
            value = self._new_ssa_value(f"_상수_{text}", FirType.정수)
            self._emit(FirInstr(
                op=FirOp.상수,
                result=value,
                operands=[int_val],
                label=f"숫자:{text}",
                source_line=source_line,
            ))
            return value
        else:
            # 변수 정의
            value = self._new_ssa_value(var_name, fir_type)
            self._emit(FirInstr(
                op=FirOp.정의,
                result=value,
                operands=[],
                label=f"{var_name} 정의 (조사:{particle})",
                source_line=source_line,
            ))
            return value

    def _process_verb_node(
        self,
        node: ContinuationNode,
        data_values: list[FirValue],
        source_line: str,
        line_num: int,
    ) -> None:
        """동사(continuation) 노드를 FIR 명령어로 변환

        Args:
            node: 동사 노드
            data_values: 앞서 정의된 데이터 값들
            source_line: 원시 소스
            line_num: 줄 번호
        """
        verb_text = node.text

        # 동사 → 연산 매핑
        verb_op_map: dict[str, FirOp] = {
            "더하기": FirOp.덧셈,
            "빼기": FirOp.뺄셈,
            "곱하기": FirOp.곱셈,
            "나누기": FirOp.나눗셈,
            "대입": FirOp.이동,
            "저장": FirOp.저장,
            "전달": FirOp.전달,
            "질문": FirOp.질문,
            "위임": FirOp.위임,
            "방송": FirOp.방송,
            "출력": FirOp.출력,
            "비교": FirOp.비교,
            "반환": FirOp.반환,
        }

        # 동사 어미 포함 매핑
        for verb_key, op in verb_op_map.items():
            if verb_text.endswith(verb_key) or verb_key in verb_text:
                result = self._new_ssa_value("_결과")
                self._emit(FirInstr(
                    op=op,
                    result=result,
                    operands=data_values if data_values else [],
                    label=f"동사:{verb_text}",
                    source_line=source_line,
                ))
                return

        # "-하다" 계열 동사
        for ending in self._VERB_ENDINGS:
            if verb_text.endswith(ending):
                stem = verb_text[:-len(ending)]
                # 어간에서 연산 유추
                if stem in ("대입", "하"):
                    result = self._new_ssa_value("_결과")
                    self._emit(FirInstr(
                        op=FirOp.이동,
                        result=result,
                        operands=data_values,
                        label=f"동사:{verb_text} (대입계열)",
                        source_line=source_line,
                    ))
                elif stem in ("더하기", "더하", "합"):
                    result = self._new_ssa_value("_결과")
                    self._emit(FirInstr(
                        op=FirOp.덧셈,
                        result=result,
                        operands=data_values,
                        label=f"동사:{verb_text} (합계계열)",
                        source_line=source_line,
                    ))
                else:
                    # 일반 실행
                    result = self._new_ssa_value("_결과")
                    self._emit(FirInstr(
                        op=FirOp.정의,
                        result=result,
                        operands=data_values,
                        label=f"동사:{verb_text} ({stem}계열)",
                        source_line=source_line,
                    ))
                return

        # 매칭 실패 — 기본 명령어
        self._emit(FirInstr(
            op=FirOp.출력,
            operands=[verb_text],
            label=f"미분류 동사:{verb_text}",
            source_line=source_line,
        ))

    # ── phi 노드 수동 삽입 ──

    def insert_phi(
        self,
        var_name: str,
        merge_block_label: str,
        sources: list[tuple[str, str]],
    ) -> FirInstr:
        """phi 노드를 수동으로 삽입

        제어 흐름 합류 지점에서 변수 병합.
        조사 기반 스코핑: 각 경로의 조사 컨텍스트를 phi 피연산자에 기록.

        Args:
            var_name: 변수 이름
            merge_block_label: 합류 블록 레이블
            sources: (블록_레이블, SSA_값) 쌍 목록

        Returns:
            생성된 phi 명령어
        """
        block = self._create_block(merge_block_label)
        return self._create_phi(var_name, block, sources)

    # ── 유틸리티 ──

    def get_ssa_value(self, name: str, version: Optional[int] = None) -> Optional[FirValue]:
        """SSA 값 조회"""
        if version is not None:
            values = self._ssa_values.get(name, [])
            for v in values:
                if v.version == version:
                    return v
            return None
        return self._get_latest_value(name)

    @property
    def blocks(self) -> dict[str, BasicBlock]:
        """현재 빌드된 모든 블록"""
        return dict(self._blocks)


# ═══════════════════════════════════════════════════════════════
# FIR 모듈
# ═══════════════════════════════════════════════════════════════

@dataclass
class FirModule:
    """FIR 모듈 — 완성된 SSA 형태의 중간 표현

    Attributes:
        blocks: 기본 블록 목록
        block_order: 블록 실행 순서
        ssa_values: SSA 변수 버전 관리 맵
    """
    blocks: list[BasicBlock]
    block_order: list[str]
    ssa_values: dict[str, list[FirValue]]

    def format(self) -> str:
        """FIR 모듈을 한국어 형식의 텍스트로 출력

        Returns:
            포맷된 FIR 텍스트
        """
        lines: list[str] = []
        lines.append("═══ FIR (유체 중간 표현) ═══")
        lines.append(f"기본 블록: {len(self.blocks)}개")
        lines.append("")

        for block_label in self.block_order:
            block = self._find_block(block_label)
            if block is None:
                continue

            lines.append(f"┌─ [{block_label}] ─────────────────")
            lines.append(f"│  선행: {block.predecessors}")
            lines.append(f"│  후속: {block.successors}")

            if block.phi_nodes:
                lines.append("│  ── φ(파이) 노드 ──")
                for phi in block.phi_nodes:
                    lines.append(f"│  {phi}")

            lines.append("│  ── 명령어 ──")
            for instr in block.instructions:
                if not instr.is_phi:
                    lines.append(f"│  {instr}")

            if block.terminator:
                lines.append(f"│  ── 종결: {FIR_OP_NAMES.get(block.terminator.op, '?')} ──")

            lines.append(f"└{'─' * 40}")
            lines.append("")

        return "\n".join(lines)

    def _find_block(self, label: str) -> Optional[BasicBlock]:
        """레이블로 블록 검색"""
        for block in self.blocks:
            if block.label == label:
                return block
        return None

    @property
    def total_instructions(self) -> int:
        """전체 명령어 수"""
        return sum(len(b.instructions) for b in self.blocks)

    @property
    def total_phi_nodes(self) -> int:
        """전체 phi 노드 수"""
        return sum(len(b.phi_nodes) for b in self.blocks)

    def get_variable_versions(self, name: str) -> list[FirValue]:
        """변수의 모든 SSA 버전 반환"""
        return self.ssa_values.get(name, [])
