"""
FLUX-KOR Bridge Adapter — 경어체/CPS 브릿지 어댑터

Exposes the Korean honorific system (경어체) and CPS (Continuation-Passing
Style) transform to the A2A type-safe cross-language bridge.

한국어의 핵심 타입 시스템:
  경어 수준 (Honorific Level): RBAC 권한 관리
  화법 (Speech Form): 실행 전략
  CPS 변환 (CPS Transform): SOV → Continuation 구조

Interface:
    adapter = KorBridgeAdapter()
    types = adapter.export_types()
    local = adapter.import_type(universal)
    cost = adapter.bridge_cost("deu")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from flux_kor.honorifics import HonorificLevel
from flux_kor.cps import (
    CPSNode,
    CPSContinuation,
    CPSBuildResult,
    CPSNodeType,
    CPSBuilder,
)


# ══════════════════════════════════════════════════════════════════════
# Common bridge types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BridgeCost:
    numeric_cost: float
    information_loss: list[str] = field(default_factory=list)
    ambiguity_warnings: list[str] = field(default_factory=list)


@dataclass
class UniversalType:
    paradigm: str
    category: str
    constraints: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class BridgeAdapter(ABC):
    @abstractmethod
    def export_types(self) -> list[UniversalType]: ...

    @abstractmethod
    def import_type(self, universal: UniversalType) -> Any: ...

    @abstractmethod
    def bridge_cost(self, target_lang: str) -> BridgeCost: ...


# ══════════════════════════════════════════════════════════════════════
# KorTypeSignature — Korean type representation
# ══════════════════════════════════════════════════════════════════════

@dataclass
class KorTypeSignature:
    """Represents a Korean type for bridge export/import.

    Korean's type system operates through:
      - honorific_level: 경어 수준 (4 levels: 하십시오체, 해요체, 해체, 해라체)
      - speech_form: 화법 (declarative, interrogative, imperative, propositive)
      - cps_transform: CPS chain depth (how deeply nested the continuation is)

    Attributes:
        honorific_level: Korean honorific level enum value
        speech_form: grammatical speech form / sentence ending type
        cps_transform: CPS continuation depth (0 = no transform)
        confidence: mapping confidence
    """
    honorific_level: HonorificLevel
    speech_form: str = "declarative"
    cps_transform: int = 0
    confidence: float = 1.0

    @property
    def korean_name(self) -> str:
        return self.honorific_level.korean_name

    @property
    def role_name(self) -> str:
        return self.honorific_level.role_name

    @property
    def cap_bit(self) -> int:
        return self.honorific_level.capability_bit


# ══════════════════════════════════════════════════════════════════════
# Honorific Level → Universal Type Mapping
# ══════════════════════════════════════════════════════════════════════

_HONORIFIC_TO_UNIVERSAL: dict[HonorificLevel, tuple[str, str, float]] = {
    HonorificLevel.HASIPSIOCHE: ("Formal", "Admin-level execution — 최경어/합쇼체", 0.95),
    HonorificLevel.HAEOYO:     ("Polite", "Standard user execution — 존댓말/해요체", 0.95),
    HonorificLevel.HAE:        ("Intimate", "Peer-level execution — 반말/해체", 0.95),
    HonorificLevel.HAERACHE:   ("Plain", "System/internal execution — 평어/해라체", 0.95),
}

# Speech form → execution mode
_SPEECH_FORM_TO_MODE: dict[str, str] = {
    "declarative":  "Direct",
    "interrogative": "Query",
    "imperative":   "Command",
    "propositive":  "Suggest",
}

# CPS depth → execution complexity
_CPS_DEPTH_TO_COMPLEXITY: dict[str, str] = {
    "flat":     "Sequential",
    "shallow":  "Composable",
    "deep":     "Recursive",
    "nested":   "HigherOrder",
}

# Reverse map: universal → honorific
_UNIVERSAL_TO_HONORIFIC: dict[str, HonorificLevel] = {
    "Formal":    HonorificLevel.HASIPSIOCHE,
    "Polite":    HonorificLevel.HAEOYO,
    "Intimate":  HonorificLevel.HAE,
    "Plain":     HonorificLevel.HAERACHE,
}


# ══════════════════════════════════════════════════════════════════════
# Language affinity
# ══════════════════════════════════════════════════════════════════════

_LANG_AFFINITY: dict[str, dict[str, Any]] = {
    "kor": {"cost": 0.0, "loss": [], "ambiguity": []},
    "zho": {"cost": 0.35, "loss": ["Honorific formality levels",
            "CPS continuation structure"],
            "ambiguity": ["Chinese politeness system differs from Korean 경어"]},
    "deu": {"cost": 0.40, "loss": ["Honorific levels (German has Sie/du only)",
            "CPS transform depth"],
            "ambiguity": ["German formality maps to 2 levels, Korean has 4"]},
    "san": {"cost": 0.45, "loss": ["Honorific system entirely",
            "CPS SOV structure"],
            "ambiguity": ["Sanskrit has no honorific system — no equivalent"]},
    "lat": {"cost": 0.45, "loss": ["Honorific levels", "SOV word order"],
            "ambiguity": ["Latin formality through mood/voice, not honorifics"]},
    "wen": {"cost": 0.50, "loss": ["Honorific system", "Verb-final structure"],
            "ambiguity": ["Classical Chinese formality through register, not conjugation"]},
}


# ══════════════════════════════════════════════════════════════════════
# KorBridgeAdapter
# ══════════════════════════════════════════════════════════════════════

class KorBridgeAdapter(BridgeAdapter):
    """Bridge adapter for the Korean (한국어) honorific/CPS type system.

    Exports Korean's four honorific levels and CPS execution modes as
    UniversalType instances for cross-language type-safe bridging.

    Usage:
        adapter = KorBridgeAdapter()
        types = adapter.export_types()
        cost = adapter.bridge_cost("deu")
    """

    PARADIGM = "kor"

    def export_types(self) -> list[UniversalType]:
        """Export all Korean honorific levels and CPS modes.

        Returns:
            List of UniversalType covering:
            - 4 honorific levels (Formal, Polite, Intimate, Plain)
            - 4 speech forms (Direct, Query, Command, Suggest)
            - CPS execution complexity levels
        """
        exported: list[UniversalType] = []

        # Export honorific levels
        for level, (cat, desc, conf) in _HONORIFIC_TO_UNIVERSAL.items():
            exported.append(UniversalType(
                paradigm=self.PARADIGM,
                category=cat,
                constraints={
                    "honorific_level": level.name,
                    "korean_name": level.korean_name,
                    "role_name": level.role_name,
                    "capability_bit": level.capability_bit,
                    "description": desc,
                    "type_kind": "honorific_level",
                },
                confidence=conf,
            ))

        # Export speech forms as execution strategies
        for form, mode in _SPEECH_FORM_TO_MODE.items():
            exported.append(UniversalType(
                paradigm=self.PARADIGM,
                category=mode,
                constraints={
                    "speech_form": form,
                    "description": f"Korean {form} speech form → {mode} execution",
                    "type_kind": "speech_form",
                },
                confidence=0.85,
            ))

        # Export CPS complexity modes
        for depth_key, complexity in _CPS_DEPTH_TO_COMPLEXITY.items():
            exported.append(UniversalType(
                paradigm=self.PARADIGM,
                category=complexity,
                constraints={
                    "cps_depth": depth_key,
                    "description": f"CPS {depth_key} → {complexity} execution",
                    "type_kind": "cps_mode",
                },
                confidence=0.8,
            ))

        return exported

    def import_type(self, universal: UniversalType) -> KorTypeSignature:
        """Import a universal type into the Korean honorific/CPS system.

        Args:
            universal: A UniversalType from another runtime

        Returns:
            KorTypeSignature with best-matching honorific level
        """
        category = universal.category
        constraints = universal.constraints

        # Resolve honorific level from category
        level = _UNIVERSAL_TO_HONORIFIC.get(category)

        # Check constraints for explicit honorific
        if level is None and "honorific_level" in constraints:
            for hl in HonorificLevel:
                if hl.name == constraints["honorific_level"]:
                    level = hl
                    break

        # Fallback to plain (lowest level, safest default)
        if level is None:
            level = HonorificLevel.HAERACHE

        # Determine speech form
        speech_form = "declarative"
        if "speech_form" in constraints:
            sf = constraints["speech_form"]
            if sf in _SPEECH_FORM_TO_MODE:
                speech_form = sf

        # Determine CPS depth from category
        cps_depth = 0
        cat_lower = category.lower()
        if "sequential" in cat_lower or "flat" in cat_lower:
            cps_depth = 0
        elif "composable" in cat_lower or "shallow" in cat_lower:
            cps_depth = 1
        elif "recursive" in cat_lower or "deep" in cat_lower:
            cps_depth = 2
        elif "higher" in cat_lower or "nested" in cat_lower:
            cps_depth = 3

        return KorTypeSignature(
            honorific_level=level,
            speech_form=speech_form,
            cps_transform=cps_depth,
            confidence=universal.confidence * 0.85,
        )

    def bridge_cost(self, target_lang: str) -> BridgeCost:
        """Estimate bridge cost to another runtime.

        Args:
            target_lang: Target language code

        Returns:
            BridgeCost with estimated difficulty
        """
        target = target_lang.lower().strip()

        if target == self.PARADIGM:
            return BridgeCost(numeric_cost=0.0)

        affinity = _LANG_AFFINITY.get(target, {
            "cost": 0.6,
            "loss": ["Honorific system", "CPS structure"],
            "ambiguity": ["Unknown target language"],
        })

        return BridgeCost(
            numeric_cost=affinity["cost"],
            information_loss=list(affinity["loss"]),
            ambiguity_warnings=list(affinity["ambiguity"]),
        )

    def detect_honorific(self, sentence: str) -> KorTypeSignature:
        """Detect honorific level from a Korean sentence.

        Args:
            sentence: Korean sentence text

        Returns:
            KorTypeSignature with detected honorific level
        """
        from flux_kor.honorifics import HonorificValidator

        validator = HonorificValidator()
        level = validator.detect(sentence)

        # Build CPS info
        builder = CPSBuilder()
        cps_result = builder.build(sentence)

        return KorTypeSignature(
            honorific_level=level,
            speech_form="declarative",
            cps_transform=cps_result.depth,
            confidence=0.9,
        )
