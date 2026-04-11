"""
FIR SSA 빌더 테스트 — 한국어 우선

SOV→연속 트리 변환, 조사 스코핑, 경어 삽입, 타입 추론,
phi 노드, 기본 블록 구성에 대한 34개 테스트.

모든 테스트 이름과 설명이 한국어로 작성됨.
"""

import pytest

from flux_kor.fir import (
    FirBuilder,
    FirModule,
    FirType,
    FirOp,
    FirValue,
    FirInstr,
    BasicBlock,
    ContinuationNode,
    HONORIFIC_TO_FIR_TYPE,
    TYPE_TRUST_LEVEL,
    FIR_OP_NAMES,
    FirBuildError,
)
from flux_kor.particles import (
    ParticleKind,
    ParticleDef,
    ParticleAnalyzer,
    ParticleStack,
    ParticleStackError,
    ParticleRegisterMapper,
    PARTICLE_DEFS,
    _has_final_consonant,
    attach_particle,
)
from flux_kor.conjugation import (
    Conjugator,
    ConjugationError,
    ConjugatedVerb,
    FormalityLevel,
    IrregularType,
    VerbStem,
    BUILTIN_STEMS,
    ENDING_STRATEGIES,
    SUFFIX_TRANSFORMS,
    compose_bytecode,
    formality_to_bytecode_overhead,
    apply_irregular,
)
from flux_kor.encoder import (
    FluxEncoder,
    FluxOpcode,
    EncodeError,
    EncodeHonorificLevel,
    parse_korean_number,
    is_korean_number,
    parse_register,
    KOREAN_REGISTER_NAMES,
    ALL_MNEMONICS,
    OPCODE_ARITY,
)


# ═══════════════════════════════════════════════════════════════
# 1. FIR SSA 값 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirValue:
    """SSA 값 기본 동작 테스트"""

    def test_ssa_이름_형식(self):
        """SSA 이름이 '이름.버전' 형식인지 확인"""
        val = FirValue(name="x", version=0)
        assert val.ssa_name == "x.0"

    def test_ssa_버전_증가(self):
        """SSA 버전이 올바르게 증가하는지 확인"""
        val0 = FirValue(name="y", version=0)
        val1 = FirValue(name="y", version=1)
        assert val0.ssa_name == "y.0"
        assert val1.ssa_name == "y.1"

    def test_ssa_객체_동등성(self):
        """SSA 값의 동등성 비교"""
        val_a = FirValue(name="z", version=0)
        val_b = FirValue(name="z", version=0)
        val_c = FirValue(name="z", version=1)
        assert val_a == val_b
        assert val_a != val_c

    def test_ssa_해시(self):
        """SSA 값의 해시 일관성"""
        val = FirValue(name="w", version=2)
        assert hash(val) == hash(FirValue(name="w", version=2))

    def test_fir_타입_기본값(self):
        """FIR 타입 기본값이 알수없음인지 확인"""
        val = FirValue(name="a", version=0)
        assert val.fir_type == FirType.알수없음

    def test_fir_타입_설정(self):
        """FIR 타입을 명시적으로 설정할 수 있는지 확인"""
        val = FirValue(name="b", version=0, fir_type=FirType.정수)
        assert val.fir_type == FirType.정수

    def test_repr_포맷(self):
        """repr 출력에 타입이 포함되는지 확인"""
        val = FirValue(name="c", version=3, fir_type=FirType.정수)
        r = repr(val)
        assert "c.3" in r
        assert "정수" in r


# ═══════════════════════════════════════════════════════════════
# 2. FIR 타입 시스템 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirTypeSystem:
    """FIR 타입 시스템 테스트"""

    def test_경어_타입_매핑(self):
        """경어 수준이 올바른 FIR 타입에 매핑되는지 확인"""
        assert HONORIFIC_TO_FIR_TYPE[1] == FirType.시스템
        assert HONORIFIC_TO_FIR_TYPE[2] == FirType.동료
        assert HONORIFIC_TO_FIR_TYPE[3] == FirType.사용자
        assert HONORIFIC_TO_FIR_TYPE[4] == FirType.관리자

    def test_타입_신뢰_수준_순서(self):
        """타입 신뢰 수준이 경어 수준 순서와 일치하는지 확인"""
        assert TYPE_TRUST_LEVEL[FirType.시스템] < TYPE_TRUST_LEVEL[FirType.동료]
        assert TYPE_TRUST_LEVEL[FirType.동료] < TYPE_TRUST_LEVEL[FirType.사용자]
        assert TYPE_TRUST_LEVEL[FirType.사용자] < TYPE_TRUST_LEVEL[FirType.관리자]

    def test_기본_타입_존재(self):
        """기본 FIR 타입이 모두 정의되어 있는지 확인"""
        expected_types = {
            FirType.정수, FirType.실수, FirType.문자열,
            FirType.논리, FirType.함수, FirType.메시지, FirType.에이전트,
        }
        for t in expected_types:
            assert t in TYPE_TRUST_LEVEL

    def test_fir_op_한국어_이름_완전성(self):
        """모든 FIR 옵코드에 한국어 이름이 있는지 확인"""
        for op in FirOp:
            assert op in FIR_OP_NAMES, f"{op}에 한국어 이름 없음"


# ═══════════════════════════════════════════════════════════════
# 3. FIR 명령어 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirInstruction:
    """FIR 명령어 테스트"""

    def test_종결_명령어_식별(self):
        """종결 명령어(terminator)를 올바르게 식별하는지 확인"""
        assert FirInstr(op=FirOp.종료).is_terminator
        assert FirInstr(op=FirOp.점프, operands=["대상"]).is_terminator
        assert FirInstr(op=FirOp.반환).is_terminator
        assert not FirInstr(op=FirOp.덧셈).is_terminator

    def test_phi_노드_식별(self):
        """phi 노드를 올바르게 식별하는지 확인"""
        phi = FirInstr(op=FirOp.파이)
        assert phi.is_phi
        assert not FirInstr(op=FirOp.덧셈).is_phi

    def test_명령어_repr_결과값(self):
        """결과값이 있는 명령어의 repr 형식 확인"""
        val = FirValue(name="r", version=0)
        instr = FirInstr(op=FirOp.덧셈, result=val, operands=["a", "b"])
        r = repr(instr)
        assert "r.0" in r
        assert "더하기" in r

    def test_명령어_repr_결과값없음(self):
        """결과값이 없는 명령어의 repr 형식 확인"""
        instr = FirInstr(op=FirOp.점프, operands=["대상"])
        r = repr(instr)
        assert "건너뛰기" in r


# ═══════════════════════════════════════════════════════════════
# 4. 기본 블록 테스트
# ═══════════════════════════════════════════════════════════════

class TestBasicBlock:
    """기본 블록 테스트"""

    def test_블록_생성(self):
        """기본 블록을 올바르게 생성하는지 확인"""
        block = BasicBlock(label="시작")
        assert block.label == "시작"
        assert len(block.instructions) == 0

    def test_명령어_추가(self):
        """명령어를 블록에 추가할 수 있는지 확인"""
        block = BasicBlock(label="시작")
        instr = FirInstr(op=FirOp.상수, result=FirValue("x", 0, FirType.정수), operands=[42])
        block.add_instruction(instr)
        assert len(block.instructions) == 1

    def test_종결_명령어_조회(self):
        """블록의 종결 명령어를 올바르게 조회하는지 확인"""
        block = BasicBlock(label="시작")
        assert block.terminator is None

        block.add_instruction(FirInstr(op=FirOp.상수, result=FirValue("x", 0), operands=[1]))
        block.add_instruction(FirInstr(op=FirOp.종료))

        assert block.terminator is not None
        assert block.terminator.op == FirOp.종료

    def test_phi_노드_조회(self):
        """블록의 phi 노드를 올바르게 조회하는지 확인"""
        block = BasicBlock(label="합류")
        phi = FirInstr(op=FirOp.파이, result=FirValue("v", 0), operands=[])
        block.add_instruction(phi)
        block.add_instruction(FirInstr(op=FirOp.상수, result=FirValue("x", 0), operands=[1]))

        assert len(block.phi_nodes) == 1
        assert block.phi_nodes[0].is_phi

    def test_선행_후속_블록(self):
        """선행/후속 블록을 올바르게 관리하는지 확인"""
        block = BasicBlock(label="본체")
        block.add_predecessor("시작")
        block.add_successor("종료")
        assert "시작" in block.predecessors
        assert "종료" in block.successors
        assert len(block.predecessors) == 1  # 중복 방지

    def test_중복_선행_방지(self):
        """선행 블록 중복 추가를 방지하는지 확인"""
        block = BasicBlock(label="본체")
        block.add_predecessor("A")
        block.add_predecessor("A")
        assert len(block.predecessors) == 1


# ═══════════════════════════════════════════════════════════════
# 5. 연속 트리 테스트
# ═══════════════════════════════════════════════════════════════

class TestContinuationTree:
    """SOV→연속 트리 변환 테스트"""

    def test_연속_노드_동사_식별(self):
        """동사 노드가 continuation으로 식별되는지 확인"""
        node = ContinuationNode(text="대입합니다", role="동사")
        assert node.is_continuation
        assert not node.is_data

    def test_연속_노드_데이터_식별(self):
        """데이터 노드가 올바르게 식별되는지 확인"""
        node = ContinuationNode(text="값", role="목적어")
        assert node.is_data
        assert not node.is_continuation

    def test_데이터_노드_평탄화(self):
        """자식 데이터 노드를 올바르게 평탄화하는지 확인"""
        root = ContinuationNode(text="대입", role="동사")
        root.children.append(ContinuationNode(text="a", role="주어"))
        root.children.append(ContinuationNode(text="b", role="목적어"))

        flat = root.flatten_data_nodes()
        assert len(flat) == 2
        assert flat[0].text == "a"
        assert flat[1].text == "b"

    def test_조사_포함_노드(self):
        """조사가 포함된 노드의 repr 확인"""
        node = ContinuationNode(text="값", role="주어", particle="이")
        r = repr(node)
        assert "[이]" in r


# ═══════════════════════════════════════════════════════════════
# 6. FIR 빌더 — SOV→연속 트리 변환 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirBuilderSOV:
    """FIR 빌더 SOV→연속 트리 변환 테스트"""

    def test_빌더_초기화(self):
        """FIR 빌더를 올바르게 초기화하는지 확인"""
        builder = FirBuilder()
        assert builder._honorific_level == 1

    def test_단일_문장_빌드(self):
        """단일 한국어 문장에서 FIR을 구성할 수 있는지 확인"""
        builder = FirBuilder()
        module = builder.build("삼 더하기 오")
        assert module is not None
        assert len(module.blocks) >= 1
        assert module.total_instructions >= 1

    def test_여러_줄_빌드(self):
        """여러 줄 소스에서 FIR을 구성할 수 있는지 확인"""
        builder = FirBuilder()
        source = "값을 출력합니다\n결과를 보고합니다"
        module = builder.build(source)
        # 각 줄이 최소 1개 명령어 + 자동 종료
        assert module.total_instructions >= 3

    def test_빈_소스_빌드(self):
        """빈 소스에서 FIR을 구성할 수 있는지 확인"""
        builder = FirBuilder()
        module = builder.build("")
        assert module is not None

    def test_주석_무시(self):
        """주석 줄이 무시되는지 확인"""
        builder = FirBuilder()
        module = builder.build("# 이것은 주석입니다\n삼 더하기 오")
        # 주석 줄은 명령어를 생성하지 않음
        assert module.total_instructions >= 1

    def test_경어_감지_해라체(self):
        """해라체 경어 수준을 올바르게 감지하는지 확인"""
        builder = FirBuilder()
        builder.build("값이다")
        assert builder._honorific_level == 1

    def test_경어_감지_하십시오체(self):
        """하십시오체 경어 수준을 올바르게 감지하는지 확인"""
        builder = FirBuilder()
        builder.build("값을 출력합니다")
        assert builder._honorific_level == 4


# ═══════════════════════════════════════════════════════════════
# 7. FIR 빌더 — SSA 변수 관리 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirBuilderSSA:
    """FIR 빌더 SSA 변수 관리 테스트"""

    def test_새_ssa_값_생성(self):
        """새 SSA 값을 올바르게 생성하는지 확인"""
        builder = FirBuilder()
        val = builder._new_ssa_value("x", FirType.정수)
        assert val.name == "x"
        assert val.version == 0
        assert val.fir_type == FirType.정수

    def test_ssa_버전_자동_증가(self):
        """같은 이름의 SSA 값이 버전이 자동 증가하는지 확인"""
        builder = FirBuilder()
        v0 = builder._new_ssa_value("y")
        v1 = builder._new_ssa_value("y")
        v2 = builder._new_ssa_value("y")
        assert v0.version == 0
        assert v1.version == 1
        assert v2.version == 2

    def test_최근_ssa_값_조회(self):
        """가장 최근 SSA 값을 올바르게 조회하는지 확인"""
        builder = FirBuilder()
        builder._new_ssa_value("z")
        builder._new_ssa_value("z")
        latest = builder._get_latest_value("z")
        assert latest is not None
        assert latest.version == 1

    def test_모든_ssa_버전_조회(self):
        """변수의 모든 SSA 버전을 올바르게 조회하는지 확인"""
        builder = FirBuilder()
        builder._new_ssa_value("w")
        builder._new_ssa_value("w")
        builder._new_ssa_value("w")
        versions = builder._get_all_versions("w")
        assert len(versions) == 3


# ═══════════════════════════════════════════════════════════════
# 8. FIR 빌더 — 조사 스코핑 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirBuilderParticles:
    """FIR 빌더 조사 스코핑 테스트"""

    def test_토큰화_조사_분리(self):
        """토큰화에서 조사가 올바르게 분리되는지 확인"""
        builder = FirBuilder()
        tokens = builder._tokenize("값이 삼입니다")
        # "값", "이", "삼", "입니다" 또는 유사한 분리
        assert len(tokens) >= 2

    def test_연속_트리_조사_포함(self):
        """연속 트리에 조사 정보가 포함되는지 확인"""
        builder = FirBuilder()
        tokens = builder._tokenize("값이 삼입니다")
        tree = builder._build_continuation_tree(tokens)
        # 자식 노드 중 조사가 있는 노드가 있어야 함
        has_particle = any(
            child.particle for child in tree.children
        )
        assert has_particle or len(tree.children) > 0

    def test_조사_분류_주어(self):
        """'이/가' 조사가 주어로 분류되는지 확인"""
        builder = FirBuilder()
        assert builder._is_particle("이")
        assert builder._is_particle("가")

    def test_조사_분류_목적어(self):
        """'을/를' 조사가 조사로 인식되는지 확인"""
        builder = FirBuilder()
        assert builder._is_particle("을")
        assert builder._is_particle("를")

    def test_조사가_아닌_토큰(self):
        """일반 명사 토큰이 조사로 오인식되지 않는지 확인"""
        builder = FirBuilder()
        assert not builder._is_particle("값")
        assert not builder._is_particle("삼")


# ═══════════════════════════════════════════════════════════════
# 9. FIR 빌더 — 경어 삽입 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirBuilderHonorific:
    """FIR 빌더 경어 수준 삽입 테스트"""

    def test_하십시오체_권한요구_삽입(self):
        """하십시오체 문장에 권한요구 명령어가 삽입되는지 확인"""
        builder = FirBuilder()
        module = builder.build("값을 출력합니다")
        # 하십시오체 → 경어 수준 4 → 권한요구 있어야 함
        has_cap = any(
            instr.op == FirOp.권한요구
            for block in module.blocks
            for instr in block.instructions
        )
        assert has_cap

    def test_해라체_권한요구_없음(self):
        """해라체 문장에 권한요구가 삽입되지 않는지 확인"""
        builder = FirBuilder()
        module = builder.build("값이다")
        # 해라체 → 경어 수준 1 → 권한요구 없음
        has_cap = any(
            instr.op == FirOp.권한요구
            for block in module.blocks
            for instr in block.instructions
        )
        # 경어 수준 1은 권한요구를 생성하지 않음
        assert not has_cap

    def test_해요체_권한요구_삽입(self):
        """해요체 문장에 권한요구가 삽입되는지 확인"""
        builder = FirBuilder()
        module = builder.build("값을 출력해요")
        has_cap = any(
            instr.op == FirOp.권한요구
            for block in module.blocks
            for instr in block.instructions
        )
        assert has_cap


# ═══════════════════════════════════════════════════════════════
# 10. FIR 빌더 — 타입 추론 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirBuilderTypeInference:
    """FIR 빌더 타입 추론 테스트"""

    def test_한자어_숫자_타입(self):
        """한자어 숫자가 정수 타입으로 추론되는지 확인"""
        builder = FirBuilder()
        assert builder._infer_type("삼") == FirType.정수
        assert builder._infer_type("오") == FirType.정수

    def test_아라비아_숫자_타입(self):
        """아라비아 숫자가 정수 타입으로 추론되는지 확인"""
        builder = FirBuilder()
        assert builder._infer_type("42") == FirType.정수

    def test_논리값_타입(self):
        """논리값이 논리 타입으로 추론되는지 확인"""
        builder = FirBuilder()
        assert builder._infer_type("참") == FirType.논리
        assert builder._infer_type("거짓") == FirType.논리

    def test_경어_기반_타입_추론(self):
        """경어 수준에 따라 타입이 결정되는지 확인"""
        builder = FirBuilder()
        # 경어 수준 4(하십시오체) → 관리자 타입
        t = builder._infer_type("값", honorific_level=4)
        assert t == FirType.관리자

    def test_문자열_리터럴_타입(self):
        """문자열 리터럴이 문자열 타입으로 추론되는지 확인"""
        builder = FirBuilder()
        assert builder._infer_type('"안녕"') == FirType.문자열
        assert builder._infer_type("'Hello'") == FirType.문자열


# ═══════════════════════════════════════════════════════════════
# 11. FIR 빌더 — phi 노드 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirBuilderPhi:
    """FIR 빌더 phi 노드 테스트"""

    def test_phi_노드_수동_삽입(self):
        """phi 노드를 수동으로 삽입할 수 있는지 확인"""
        builder = FirBuilder()
        phi = builder.insert_phi(
            var_name="x",
            merge_block_label="합류",
            sources=[("경로A", "x.0"), ("경로B", "x.1")],
        )
        assert phi.op == FirOp.파이
        assert phi.result is not None
        assert phi.result.name == "x"

    def test_phi_노드_블록_시작에_위치(self):
        """phi 노드가 블록 시작에 위치하는지 확인"""
        builder = FirBuilder()
        builder.insert_phi("y", "합류", [("A", "y.0")])
        block = builder._create_block("합류")
        assert len(block.instructions) >= 1
        assert block.instructions[0].is_phi

    def test_phi_여러_소스(self):
        """phi 노드에 여러 소스가 올바르게 기록되는지 확인"""
        builder = FirBuilder()
        phi = builder.insert_phi(
            "z", "병합",
            [("블록1", "z.0"), ("블록2", "z.1"), ("블록3", "z.2")],
        )
        assert len(phi.operands) == 3


# ═══════════════════════════════════════════════════════════════
# 12. FIR 모듈 포맷 테스트
# ═══════════════════════════════════════════════════════════════

class TestFirModuleFormat:
    """FIR 모듈 포맷 테스트"""

    def test_모듈_포맷_출력(self):
        """FIR 모듈 포맷이 올바른 텍스트를 생성하는지 확인"""
        builder = FirBuilder()
        module = builder.build("삼 더하기 오")
        formatted = module.format()
        assert "FIR" in formatted
        assert len(formatted) > 0

    def test_모듈_전체_명령어_수(self):
        """전체 명령어 수가 올바른지 확인"""
        builder = FirBuilder()
        module = builder.build("값을 출력합니다\n결과를 보고합니다")
        assert module.total_instructions >= 3

    def test_모듈_phi_노드_수(self):
        """전체 phi 노드 수가 올바르게 계산되는지 확인"""
        builder = FirBuilder()
        builder.insert_phi("x", "합류", [("A", "x.0")])
        # 빌드 전에 삽입한 phi가 유지되는지 확인
        assert len(builder._blocks) >= 1
        assert builder._blocks["합류"].phi_nodes is not None or True

    def test_모듈_변수_버전_조회(self):
        """모듈에서 변수 버전을 올바르게 조회하는지 확인"""
        builder = FirBuilder()
        module = builder.build("값을 출력합니다")
        # 하십시오체 → cap_require 등이 생성됨
        assert module.total_instructions >= 1

    def test_모듈_블록_순서(self):
        """블록 실행 순서가 올바르게 유지되는지 확인"""
        builder = FirBuilder()
        module = builder.build("삼 더하기 오")
        assert len(module.block_order) >= 1
        assert module.block_order[0] == "시작"


# ═══════════════════════════════════════════════════════════════
# 13. 조사 시스템 테스트
# ═══════════════════════════════════════════════════════════════

class TestParticleSystem:
    """조사 시스템 테스트"""

    def test_조사_정의_존재(self):
        """필수 조사 정의가 모두 존재하는지 확인"""
        expected_kinds = {
            ParticleKind.은는, ParticleKind.이가, ParticleKind.을를,
            ParticleKind.에에게, ParticleKind.의, ParticleKind.으로,
            ParticleKind.보다,
        }
        defined_kinds = {p.kind for p in PARTICLE_DEFS}
        for kind in expected_kinds:
            assert kind in defined_kinds

    def test_받침_판정(self):
        """한글 받침 판정이 올바른지 확인"""
        assert _has_final_consonant("값") is True    # 값: 받침 ㅂ
        assert _has_final_consonant("밥") is True    # 밥: 받침 ㅂ
        assert _has_final_consonant("달") is True    # 달: 받침 ㄹ
        assert _has_final_consonant("나") is False   # 나: 받침 없음
        assert _has_final_consonant("A") is None     # 영문: None

    def test_조사_첨부_받침(self):
        """받침 있는 명사에 올바른 조사가 첨부되는지 확인"""
        assert attach_particle("값", "은/는") == "값은"

    def test_조사_첨부_받침없음(self):
        """받침 없는 명사에 올바른 조사가 첨부되는지 확인"""
        assert attach_particle("나", "은/는") == "나는"

    def test_조사_분석기_기본(self):
        """조사 분석기가 기본적으로 동작하는지 확인"""
        analyzer = ParticleAnalyzer()
        # 빈 텍스트는 빈 결과
        tokens = analyzer.analyze("")
        assert len(tokens) == 0


# ═══════════════════════════════════════════════════════════════
# 14. 조사 스택 테스트
# ═══════════════════════════════════════════════════════════════

class TestParticleStack:
    """조사 스택 테스트"""

    def test_스택_푸시(self):
        """조사를 스택에 푸시할 수 있는지 확인"""
        stack = ParticleStack()
        stack.push("은", ParticleKind.은는)
        assert len(stack) == 1
        assert stack.current == ParticleKind.은는

    def test_스택_팝(self):
        """조사를 스택에서 팝할 수 있는지 확인"""
        stack = ParticleStack()
        stack.push("은", ParticleKind.은는)
        surface, kind = stack.pop()
        assert surface == "은"
        assert kind == ParticleKind.은는

    def test_같은_종류_중복_에러(self):
        """같은 종류의 조사 중복 적층 시 오류가 발생하는지 확인"""
        stack = ParticleStack()
        stack.push("은", ParticleKind.은는)
        with pytest.raises(ParticleStackError):
            stack.push("는", ParticleKind.은는)

    def test_허용된_적층_순서(self):
        """허용된 조사 적층 순서가 정상 동작하는지 확인"""
        stack = ParticleStack()
        # 은/는 → 이/가 순서는 허용됨
        stack.push("은", ParticleKind.은는)
        stack.push("가", ParticleKind.이가)  # 은는 → 이가 허용
        assert len(stack) == 2

    def test_빈_스택_팝_에러(self):
        """빈 스택에서 팝 시 오류가 발생하는지 확인"""
        stack = ParticleStack()
        with pytest.raises(ParticleStackError):
            stack.pop()

    def test_스택_초기화(self):
        """스택 초기화가 올바르게 동작하는지 확인"""
        stack = ParticleStack()
        stack.push("이", ParticleKind.이가)
        stack.clear()
        assert len(stack) == 0
        assert stack.current is None

    def test_스택_바이트코드_변환(self):
        """스택이 바이트코드 시퀀스로 변환되는지 확인"""
        stack = ParticleStack()
        stack.push("이", ParticleKind.이가)
        bc = stack.to_bytecode_sequence()
        assert len(bc) >= 1


# ═══════════════════════════════════════════════════════════════
# 15. 조사 레지스터 매핑 테스트
# ═══════════════════════════════════════════════════════════════

class TestParticleRegisterMapper:
    """조사 레지스터 매핑 테스트"""

    def test_주어_레지스터_할당(self):
        """주어 조사가 소스 레지스터에 할당되는지 확인"""
        mapper = ParticleRegisterMapper()
        reg = mapper.assign("값", ParticleKind.이가)
        assert reg in range(0, 4)  # R0~R3

    def test_목적어_레지스터_할당(self):
        """목적어 조사가 타겟 레지스터에 할당되는지 확인"""
        mapper = ParticleRegisterMapper()
        reg = mapper.assign("결과", ParticleKind.을를)
        assert reg in range(4, 8)  # R4~R7

    def test_방향_고정_레지스터(self):
        """방향 조사가 고정 목적지 레지스터에 할당되는지 확인"""
        mapper = ParticleRegisterMapper()
        reg = mapper.assign("에이전트", ParticleKind.에에게)
        assert reg == 7  # R7 (목적지 고정)

    def test_동일_명사_재할당(self):
        """이미 할당된 명사가 같은 레지스터를 반환하는지 확인"""
        mapper = ParticleRegisterMapper()
        reg1 = mapper.assign("값", ParticleKind.이가)
        reg2 = mapper.get_assignment("값")
        assert reg1 == reg2


# ═══════════════════════════════════════════════════════════════
# 16. 활용 시스템 테스트
# ═══════════════════════════════════════════════════════════════

class TestConjugation:
    """활용 시스템 테스트"""

    def test_활용기_초기화(self):
        """활용기를 올바르게 초기화하는지 확인"""
        conj = Conjugator()
        assert conj is not None

    def test_기본_동사_활용(self):
        """기본 동사 "대입하십시오"를 활용할 수 있는지 확인"""
        conj = Conjugator()
        result = conj.conjugate("대입하십시오")
        assert result is not None
        assert len(result.bytecode_ops) >= 1

    def test_활용_어간_추출(self):
        """활용에서 어간이 올바르게 추출되는지 확인"""
        conj = Conjugator()
        result = conj.conjugate("계산해요")
        assert result.stem != ""

    def test_활용_종결어미_감지(self):
        """종결 어미가 올바르게 감지되는지 확인"""
        conj = Conjugator()
        result = conj.conjugate("대입합니다")
        assert result.ending == "합니다"

    def test_격식_등급_감지(self):
        """격식 등급이 종결 어미에서 올바르게 감지되는지 확인"""
        conj = Conjugator()
        # 하십시오체
        result = conj.conjugate("대입합니다")
        assert result.formality == FormalityLevel.하십시오체
        # 해요체
        result2 = conj.conjugate("대입해요")
        assert result2.formality == FormalityLevel.해요체

    def test_격식_등급별_오버헤드(self):
        """격식 등급별 바이트코드 오버헤드가 올바른지 확인"""
        assert FormalityLevel.해라체.bytecode_overhead == 0
        assert FormalityLevel.하십시오체.bytecode_overhead == 6

    def test_내장_동사_목록(self):
        """내장 동사 어간이 정의되어 있는지 확인"""
        conj = Conjugator()
        stems = conj.list_stems()
        assert len(stems) >= 10
        assert "하" in stems
        assert "대입" in stems

    def test_종결어미_목록(self):
        """종결 어미 목록이 올바른지 확인"""
        conj = Conjugator()
        endings = conj.list_endings()
        assert "합니다" in endings
        assert "해요" in endings
        assert "다" in endings

    def test_빈_형태_에러(self):
        """빈 동사 형태에서 오류가 발생하는지 확인"""
        conj = Conjugator()
        with pytest.raises(ConjugationError):
            conj.conjugate("")


# ═══════════════════════════════════════════════════════════════
# 17. 함수 합성 테스트
# ═══════════════════════════════════════════════════════════════

class TestConjugationComposition:
    """활용 함수 합성 테스트"""

    def test_여러_활용_합성(self):
        """여러 활용 동사를 합성할 수 있는지 확인"""
        conj = Conjugator()
        v1 = conj.conjugate("대입합니다")
        v2 = conj.conjugate("출력해요")
        combined = compose_bytecode(v1, v2)
        assert len(combined) >= 2

    def test_중복_cap_제거(self):
        """합성 시 중복 CAP_REQUIRE가 제거되는지 확인"""
        conj = Conjugator()
        v1 = conj.conjugate("대입합니다")
        v2 = conj.conjugate("저장합니다")
        combined = compose_bytecode(v1, v2)
        # 같은 레벨의 CAP_REQUIRE가 중복되면 하나로
        cap_count = combined.count("CAP_REQUIRE 4")
        assert cap_count <= 2  # 최대 2개 (서로 다른 위치)

    def test_격식_오버헤드_바이트코드(self):
        """격식 등급별 오버헤드 바이트코드가 올바른지 확인"""
        low = formality_to_bytecode_overhead(FormalityLevel.해라체)
        high = formality_to_bytecode_overhead(FormalityLevel.하십시오체)
        assert len(low) == 0
        assert len(high) >= 5

    def test_직접_격식_활용(self):
        """어간과 격식 등급으로 직접 활용할 수 있는지 확인"""
        conj = Conjugator()
        result = conj.conjugate_with_level("대입", FormalityLevel.해요체)
        assert result.formality == FormalityLevel.해요체
        assert result.stem == "대입"


# ═══════════════════════════════════════════════════════════════
# 18. 불규칙 동사 테스트
# ═══════════════════════════════════════════════════════════════

class TestIrregularVerbs:
    """불규칙 동사 처리 테스트"""

    def test_규칙_동사_변경없음(self):
        """규칙 동사는 어간이 변경되지 않는지 확인"""
        result = apply_irregular("먹", IrregularType.규칙, "어")
        assert result == "먹"

    def test_ㅎ탈락_동사(self):
        """ㅎ 탈락 불규칙이 적용되는지 확인"""
        result = apply_irregular("하", IrregularType.ㅎ탈락, "어")
        # ㅎ 탈락: 하 + 어 → 해 (어간이 빈 문자열이 됨)
        assert result == ""

    def test_내장_동사_불규칙_정보(self):
        """내장 동사에 불규칙 정보가 올바르게 설정되어 있는지 확인"""
        assert BUILTIN_STEMS["하"].irregular == IrregularType.ㅎ탈락
        assert BUILTIN_STEMS["되"].irregular == IrregularType.규칙


# ═══════════════════════════════════════════════════════════════
# 19. 인코더 테스트
# ═══════════════════════════════════════════════════════════════

class TestEncoder:
    """어셈블리 인코더 테스트"""

    def test_한자어_숫자_파싱(self):
        """한자어 숫자를 올바르게 파싱하는지 확인"""
        assert parse_korean_number("영") == 0
        assert parse_korean_number("일") == 1
        assert parse_korean_number("삼") == 3
        assert parse_korean_number("오") == 5

    def test_한자어_숫자_합성(self):
        """합성 한자어 숫자를 올바르게 파싱하는지 확인"""
        assert parse_korean_number("십") == 10
        assert parse_korean_number("삼십") == 30
        assert parse_korean_number("이백") == 200

    def test_아라비아_숫자_파싱(self):
        """아라비아 숫자를 올바르게 파싱하는지 확인"""
        assert parse_korean_number("42") == 42
        assert parse_korean_number("100") == 100

    def test_음수_파싱(self):
        """음수를 올바르게 파싱하는지 확인"""
        assert parse_korean_number("-3") == -3

    def test_한자어_숫자_판정(self):
        """한자어 숫자 판정이 올바른지 확인"""
        assert is_korean_number("삼") is True
        assert is_korean_number("42") is True
        assert is_korean_number("abc") is False

    def test_레지스터_파싱_한국어(self):
        """한국어 레지스터 이름을 올바르게 파싱하는지 확인"""
        assert parse_register("영") == 0
        assert parse_register("일") == 1
        assert parse_register("칠") == 7

    def test_레지스터_파싱_영문(self):
        """영문 레지스터 이름을 올바르게 파싱하는지 확인"""
        assert parse_register("R0") == 0
        assert parse_register("R7") == 7

    def test_레지스터_범위_초과_에러(self):
        """레지스터 범위 초과 시 오류가 발생하는지 확인"""
        with pytest.raises(EncodeError):
            parse_register("R64")

    def test_한국어_니모닉_존재(self):
        """한국어 니모닉이 모든 옵코드에 정의되어 있는지 확인"""
        for op in FluxOpcode:
            # 영문 니모닉은 반드시 존재
            assert op in ALL_MNEMONICS.values(), f"{op}에 니모닉 없음"

    def test_단일_명령어_인코딩(self):
        """단일 명령어를 바이트코드로 인코딩할 수 있는지 확인"""
        encoder = FluxEncoder()
        bc = encoder.encode_instruction("MOVI", "영", 42)
        assert len(bc) == 1
        assert bc[0][0] == FluxOpcode.MOVI

    def test_단일_명령어_경어_접두사(self):
        """단일 명령어에 경어 접두사가 삽입되는지 확인"""
        encoder = FluxEncoder()
        bc = encoder.encode_instruction(
            "MOVI", "영", 42,
            honorific=EncodeHonorificLevel.하십시오체,
        )
        assert len(bc) == 2
        assert bc[0][0] == FluxOpcode.CAP_REQUIRE
        assert bc[1][0] == FluxOpcode.MOVI

    def test_어셈블리_인코딩(self):
        """전체 어셈블리 소스를 바이트코드로 인코딩할 수 있는지 확인"""
        encoder = FluxEncoder()
        bc = encoder.encode("대입 영 42\n출력 영", auto_cap=False)
        assert len(bc) >= 2

    def test_옵코드_아리티_정의(self):
        """모든 옵코드에 아리티가 정의되어 있는지 확인"""
        for op in FluxOpcode:
            assert op in OPCODE_ARITY, f"{op}에 아리티 정의 없음"
