"""
FLUX-kor — 유체 · 流體言語通用執行

한국어 우선 자연어 런타임. 한국어 문법이 아키텍처를 근본적으로 형성:
  경어 체계 → RBAC 권한 관리
  SOV 어순 → CPS 컴파일
  조사 → 범위 연산자
  활용 → 함수 합성
  한자어 → 공유 추상 타입명
"""

__version__ = "0.2.0"
__title__ = "FLUX-kor"

from flux_kor.vm import Opcode, VM
from flux_kor.honorifics import HonorificLevel, HonorificValidator
from flux_kor.interpreter import FluxInterpreterKor
from flux_kor.capability import HonorificCapabilityResolver, CapabilityLevel, CapabilityToken, CapabilityError
from flux_kor.particle_scope import ParticleScopeCompiler, ScopeCode, ScopeStack, ScopeToken
from flux_kor.cps import CPSBuilder, CPSBuildResult, CPSNode, CPSContinuation, CPSNodeType

__all__ = [
    # Core
    "Opcode",
    "VM",
    "HonorificLevel",
    "HonorificValidator",
    "FluxInterpreterKor",
    # R7: Capability & Scope & CPS
    "HonorificCapabilityResolver",
    "CapabilityLevel",
    "CapabilityToken",
    "CapabilityError",
    "ParticleScopeCompiler",
    "ScopeCode",
    "ScopeStack",
    "ScopeToken",
    "CPSBuilder",
    "CPSBuildResult",
    "CPSNode",
    "CPSContinuation",
    "CPSNodeType",
]
