"""
CPS 연속 변환기 (CPS Continuation Builder)
============================================

한국어 SOV 어순을 Continuation-Passing Style로 변환.
동사(마지막 요소)가 continuation이 되고, 주어/목적어가
데이터 노드로 중첩되는 구조를 생성.

SOV → CPS 변환 예시:
  "나 사과 먹어" (나 사과 먹다)
  → CPS: λk. k(나, λna. k_result(apple, λapple. eat(k_final)))

  "나는 너에게 이것을 준다"
  → CPS: k(TOPIC:나, λna. k(DELEGATE:너, λneo. k(OBJECT:이것, λit. give(k_final))))

핵심 설계:
  1. 한국어 문장의 SOV 구조에서 동사를 마지막 continuation으로 추출
  2. 주어/목적어를 왼쪽에서 오른쪽으로 순차적으로 중첩
  3. 각 중첩이 다음 continuation을 받는 람다로 표현
  4. 조사가 범위/바인딩 결정
  5. 경어 수준이 실행 능력을 결정

내장 SOV → CPS IR 노드:
  CPSNode — 단일 CPS 노드 (함수 응용 또는 값)
  CPSChain — continuation 체인 (데이터 → continuation)
  CPSLambda — 람다/함수 정의
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
# CPS IR 노드 유형
# ═══════════════════════════════════════════════════════════════

class CPSNodeType(Enum):
    """CPS IR 노드 유형."""
    VALUE = auto()       # 원시 값 (상수, 변수 참조)
    APPLY = auto()       # 함수 응용: f(arg, k)
    LAMBDA = auto()      # 람다: λx. body
    CHAIN = auto()       # 연속 체인: node₁ → node₂
    LET = auto()         # let 바인딩: let x = node in body
    CONDITIONAL = auto() # 조건부: if test then t else f
    SEND = auto()        # 메시지 전송: send(target, msg, k)
    HALT = auto()        # 종료


# ═══════════════════════════════════════════════════════════════
# CPS IR 노드
# ═══════════════════════════════════════════════════════════════

@dataclass
class CPSNode:
    """CPS IR 기본 노드.

    Attributes:
        node_type: 노드 유형
        value: 값 (문자열, 숫자, 또는 다른 노드)
        children: 자식 노드 목록
        particle: 이 노드에 연결된 조사 (범위 정보)
        honorific_level: 경어 수준
        source_text: 원시 소스 텍스트
    """
    node_type: CPSNodeType
    value: Any = None
    children: list[CPSNode] = field(default_factory=list)
    particle: str = ""
    honorific_level: int = 1
    source_text: str = ""

    def is_continuation(self) -> bool:
        """이 노드가 continuation (동사)인지."""
        return self.node_type == CPSNodeType.APPLY

    def is_data(self) -> bool:
        """이 노드가 데이터 노드 (주어/목적어)인지."""
        return self.node_type == CPSNodeType.VALUE

    def __repr__(self) -> str:
        p = f"[{self.particle}]" if self.particle else ""
        return f"CPSNode({self.node_type.name}, {self.value!r}{p})"


@dataclass
class CPSContinuation:
    """CPS continuation — 함수 응용 표현.

    func에 arg를 전달하고, 결과가 cont로 전달됨.
    CPS: cont(func(arg))

    Attributes:
        func: 함수 노드
        arg: 인자 노드
        cont: 다음 continuation (없으면 최종)
        label: 디버그 레이블
    """
    func: Any = None
    arg: Any = None
    cont: Optional[CPSContinuation] = None
    label: str = ""

    def chain(self, next_cont: CPSContinuation) -> CPSContinuation:
        """이 continuation 뒤에 다음 continuation을 연결."""
        if self.cont is None:
            self.cont = next_cont
        else:
            self.cont = self.cont.chain(next_cont)
        return self

    def flatten(self) -> list[CPSContinuation]:
        """연속 체인을 평탄화."""
        result = [self]
        current = self.cont
        while current is not None:
            result.append(current)
            current = current.cont
        return result

    def depth(self) -> int:
        """연속 체인 깊이."""
        return len(self.flatten())

    def __repr__(self) -> str:
        if self.cont is None:
            return f"CPS({self.label or '...'}.({self.func}, {self.arg}))"
        return f"CPS({self.label or '...'}.({self.func}, {self.arg}) → {self.cont.label or '...'})"


# ═══════════════════════════════════════════════════════════════
# SOV → CPS 변환기
# ═══════════════════════════════════════════════════════════════

# 동사 어미 패턴 (간단한 버전)
_VERB_ENDINGS = [
    "하십시오", "합니다", "습니다", "ㅂ니다", "세요", "으세요",
    "해요", "아요", "어요", "예요", "이에요",
    "한다", "이다", "해라", "거라", "너라",
    "해", "아", "어", "다",
    "대입", "전달", "질문", "방송", "계산", "실행",
    "대입하세요", "전달하세요", "방송하세요", "계산하세요",
    "구해", "계산해", "실행해", "먹어", "준다", "본다",
]

# 조사 패턴 (긴 것부터 우선)
_PARTICLE_ENDINGS = [
    "에게는", "에게", "한테", "에서", "부터", "까지",
    "으로", "보다", "마다",
    "은", "는", "이", "가", "을", "를",
    "에", "의", "도", "만", "로",
]


class CPSBuilder:
    """SOV → CPS 변환기 — 한국어 SOV 문장을 CPS IR로 변환.

    핵심 알고리즘:
      1. 텍스트를 토큰화 (명사 + 조사, 동사 분리)
      2. 마지막 토큰이 동사인지 확인
      3. 동사 앞의 모든 토큰을 데이터 노드로 변환
      4. 데이터 노드들을 순차적으로 중첩 (왼쪽 → 오른쪽)
      5. 가장 안쪽에 동사 continuation 삽입

    Usage::

        builder = CPSBuilder()
        result = builder.build("나 사과 먹어")
        # → CPS chain: k(나, λna. k(apple, λapple. eat(k_final)))

        result = builder.build("나는 너에게 이것을 준다")
        # → CPS chain with particle scoping
    """

    def __init__(self) -> None:
        self._counter = 0

    def _fresh_var(self, prefix: str = "k") -> str:
        """새 continuation 변수명 생성."""
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def build(self, sentence: str) -> CPSBuildResult:
        """한국어 SOV 문장을 CPS IR로 변환.

        Args:
            sentence: 한국어 자연어 문장

        Returns:
            CPS 빌드 결과
        """
        self._counter = 0
        tokens = self._tokenize(sentence)

        if not tokens:
            return CPSBuildResult(
                nodes=[],
                continuation=None,
                cps_ir="",
                depth=0,
                original=sentence,
            )

        # 동사(마지막 토큰) 분리
        verb_token = tokens[-1]
        data_tokens = tokens[:-1]

        # 동사 continuation 생성
        final_k = self._fresh_var("k_final")
        verb_cont = CPSContinuation(
            func=verb_token["text"],
            arg=final_k,
            label=f"verb:{verb_token['text']}",
        )

        # 데이터 노드를 역순으로 중첩 (오른쪽 → 왼쪽)
        # 마지막 데이터 노드가 가장 안쪽
        current_cont = verb_cont

        data_nodes: list[CPSNode] = []
        for dt in reversed(data_tokens):
            k_var = self._fresh_var("k")
            node = CPSNode(
                node_type=CPSNodeType.VALUE,
                value=dt["text"],
                particle=dt.get("particle", ""),
                source_text=sentence,
            )
            data_nodes.append(node)

            cont = CPSContinuation(
                func=k_var,
                arg=node.value,
                cont=current_cont,
                label=f"data:{dt['text']}",
            )
            current_cont = cont

        # 노드 목록을 원래 순서로 정렬
        data_nodes.reverse()

        # CPS IR 텍스트 생성
        cps_ir = self._format_cps(data_nodes, verb_token, final_k)

        return CPSBuildResult(
            nodes=data_nodes + [CPSNode(
                node_type=CPSNodeType.APPLY,
                value=verb_token["text"],
                source_text=sentence,
            )],
            continuation=current_cont,
            cps_ir=cps_ir,
            depth=current_cont.depth(),
            original=sentence,
        )

    def build_nested(self, sentences: list[str]) -> CPSBuildResult:
        """여러 문장을 중첩된 CPS로 변환.

        각 문장의 CPS를 이전 문장의 continuation 안에 중첩.
        """
        if not sentences:
            return CPSBuildResult(
                nodes=[], continuation=None, cps_ir="", depth=0, original=""
            )

        # 첫 문장으로 기본 CPS 생성
        result = self.build(sentences[0])
        all_nodes = result.nodes[:]
        base_cont = result.continuation

        # 이후 문장들을 순차적으로 중첩
        for sentence in sentences[1:]:
            sub_result = self.build(sentence)
            all_nodes.extend(sub_result.nodes)

            # sub_result의 continuation을 base_cont 안에 중첩
            if sub_result.continuation and base_cont:
                # 마지막 continuation을 덮어씀
                innermost = base_cont
                while innermost.cont is not None:
                    innermost = innermost.cont
                innermost.cont = sub_result.continuation

        return CPSBuildResult(
            nodes=all_nodes,
            continuation=base_cont,
            cps_ir=self._format_nested(sentences),
            depth=base_cont.depth() if base_cont else 0,
            original="\n".join(sentences),
        )

    def _tokenize(self, sentence: str) -> list[dict[str, str]]:
        """한국어 문장을 토큰화.

        조사를 분리하여 명사/동사/조사를 식별.
        """
        tokens: list[dict[str, str]] = []
        remaining = sentence.strip()

        while remaining:
            matched = False

            # 조사 매칭 (긴 것 우선)
            for particle in _PARTICLE_ENDINGS:
                if remaining.endswith(particle):
                    noun = remaining[:-len(particle)].strip()
                    if noun:
                        tokens.append({
                            "text": noun,
                            "type": "data",
                            "particle": particle,
                        })
                    tokens.append({
                        "text": particle,
                        "type": "particle",
                        "particle": particle,
                    })
                    remaining = ""
                    matched = True
                    break

            if matched:
                continue

            # 동사 어미 매칭
            for ending in _VERB_ENDINGS:
                if remaining.endswith(ending) and len(remaining) > len(ending):
                    base = remaining[:-len(ending)].strip()
                    if base:
                        # 앞에 공백이 있으면 공백으로 분리
                        tokens.append({
                            "text": ending,
                            "type": "verb",
                            "particle": "",
                        })
                        remaining = base
                        matched = True
                        break

            if matched:
                continue

            # 공백으로 분리
            parts = remaining.rsplit(None, 1)
            if len(parts) == 2:
                tokens.append({
                    "text": parts[1],
                    "type": "data",
                    "particle": "",
                })
                remaining = parts[0]
            else:
                tokens.append({
                    "text": parts[0],
                    "type": "unknown",
                    "particle": "",
                })
                remaining = ""

        # 역순으로 수집했으므로 정렬
        tokens.reverse()

        # particle 타입을 data에 병합
        merged: list[dict[str, str]] = []
        for tok in tokens:
            if tok["type"] == "particle":
                continue
            merged.append(tok)

        return merged

    def _format_cps(
        self,
        data_nodes: list[CPSNode],
        verb_token: dict[str, str],
        final_k: str,
    ) -> str:
        """CPS IR을 읽기 쉬운 형식으로 변환."""
        if not data_nodes:
            return f"{verb_token['text']}({final_k})"

        parts = []
        for i, node in enumerate(data_nodes):
            k = self._fresh_var("k") if i < len(data_nodes) else final_k
            p = f"[{node.particle}]" if node.particle else ""
            parts.append(f"{node.value}{p}")

        verb = verb_token["text"]
        # 중첩 형식: k(na, k1 => k1(apple, k2 => k2(eat, k_final)))
        inner = f"{verb}({final_k})"
        for node in reversed(data_nodes):
            k_var = self._fresh_var("k")
            p = f"[{node.particle}]" if node.particle else ""
            inner = f"{k_var}({node.value}{p}, λ{k_var}. {inner})"

        return inner

    def _format_nested(self, sentences: list[str]) -> str:
        """중첩된 CPS를 형식화."""
        parts = []
        for i, s in enumerate(sentences):
            result = self.build(s)
            parts.append(f"  [{i}] {result.cps_ir}")
        return "NESTED_CPS:\n" + "\n".join(parts)


@dataclass
class CPSBuildResult:
    """CPS 빌드 결과.

    Attributes:
        nodes: 생성된 CPS 노드 목록
        continuation: 최상위 continuation 체인
        cps_ir: CPS IR 텍스트 표현
        depth: continuation 중첩 깊이
        original: 원시 소스 텍스트
    """
    nodes: list[CPSNode]
    continuation: Optional[CPSContinuation]
    cps_ir: str
    depth: int
    original: str

    def describe(self) -> str:
        """결과를 설명 형식으로 반환."""
        lines = [
            f"원시: {self.original!r}",
            f"깊이: {self.depth}",
            f"노드 수: {len(self.nodes)}",
            f"CPS IR: {self.cps_ir}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CPSBuildResult(depth={self.depth}, nodes={len(self.nodes)})"
