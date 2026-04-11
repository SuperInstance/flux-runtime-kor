"""
유체 한국어 인터프리터 테스트

한국어 NL 패턴, 경어 시스템, SOV 구조, VM 실행에 대한 20개 테스트.
"""

import pytest
import math

from flux_kor.vm import Opcode, VM, VMError, CapabilityError
from flux_kor.honorifics import HonorificLevel, HonorificValidator, HonorificError
from flux_kor.interpreter import (
    FluxInterpreterKor,
    KoreanNumeral,
    Particles,
    SOVContinuation,
    ParseError,
)


# ═══════════════════════════════════════════════════════════
# 1. VM 기본 테스트
# ═══════════════════════════════════════════════════════════

class TestVM:
    """VM 기본 동작 테스트"""

    def test_vm_create(self):
        vm = VM()
        assert len(vm.registers) == 8
        assert vm.pc == 0
        assert not vm.halted

    def test_load_imm(self):
        vm = VM()
        vm.execute([(Opcode.LOAD_IMM, 0, 42)])
        assert vm.registers[0] == 42

    def test_add(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 3),
            (Opcode.LOAD_IMM, 1, 5),
            (Opcode.ADD, 2, 0, 1),
        ])
        assert vm.registers[2] == 8

    def test_multiply(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 7),
            (Opcode.LOAD_IMM, 1, 6),
            (Opcode.MUL, 2, 0, 1),
        ])
        assert vm.registers[2] == 42

    def test_subtract(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 10),
            (Opcode.LOAD_IMM, 1, 3),
            (Opcode.SUB, 2, 0, 1),
        ])
        assert vm.registers[2] == 7

    def test_divide(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 20),
            (Opcode.LOAD_IMM, 1, 4),
            (Opcode.DIV, 2, 0, 1),
        ])
        assert vm.registers[2] == 5

    def test_divide_by_zero(self):
        vm = VM()
        with pytest.raises(VMError, match="0으로 나눌 수 없습니다"):
            vm.execute([
                (Opcode.LOAD_IMM, 0, 10),
                (Opcode.LOAD_IMM, 1, 0),
                (Opcode.DIV, 2, 0, 1),
            ])

    def test_halt(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 99),
            (Opcode.HALT,),
            (Opcode.LOAD_IMM, 1, 100),  # 실행되지 않아야 함
        ])
        assert vm.halted
        assert vm.registers[1] == 0

    def test_factorial(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 5),
            (Opcode.FACTORIAL, 0),
        ])
        assert vm.registers[0] == 120

    def test_sum_range(self):
        vm = VM()
        vm.execute([
            (Opcode.LOAD_IMM, 0, 1),
            (Opcode.LOAD_IMM, 1, 10),
            (Opcode.SUM_RANGE, 2, 0, 1),
        ])
        assert vm.registers[2] == 55  # 1+2+...+10

    def test_print_output(self):
        vm = VM()
        output = vm.execute([
            (Opcode.PRINT, "안녕하세요"),
            (Opcode.PRINT, 42),
        ])
        assert output == ["안녕하세요", "42"]

    def test_register_bounds(self):
        vm = VM()
        with pytest.raises(VMError, match="레지스터 범위 초과"):
            vm._set_reg(10, 0)


# ═══════════════════════════════════════════════════════════
# 2. 경어 시스템 테스트
# ═══════════════════════════════════════════════════════════

class TestHonorifics:
    """경어 수준 테스트"""

    def test_hasipsioche_is_highest(self):
        assert HonorificLevel.HASIPSIOCHE.value == 4
        assert HonorificLevel.HASIPSIOCHE.role_name == "admin"

    def test_haerache_is_lowest(self):
        assert HonorificLevel.HAERACHE.value == 1
        assert HonorificLevel.HAERACHE.role_name == "system/internal"

    def test_from_name_korean(self):
        assert HonorificLevel.from_name("하십시오체") == HonorificLevel.HASIPSIOCHE
        assert HonorificLevel.from_name("해요체") == HonorificLevel.HAEOYO
        assert HonorificLevel.from_name("해체") == HonorificLevel.HAE
        assert HonorificLevel.from_name("해라체") == HonorificLevel.HAERACHE

    def test_from_name_english(self):
        assert HonorificLevel.from_name("formal") == HonorificLevel.HASIPSIOCHE
        assert HonorificLevel.from_name("polite") == HonorificLevel.HAEOYO
        assert HonorificLevel.from_name("intimate") == HonorificLevel.HAE
        assert HonorificLevel.from_name("plain") == HonorificLevel.HAERACHE

    def test_capability_hierarchy(self):
        """상위 경어 수준이 하위 경어 수준에 접근 가능"""
        admin = HonorificLevel.HASIPSIOCHE
        user = HonorificLevel.HAEOYO
        peer = HonorificLevel.HAE
        system = HonorificLevel.HAERACHE

        assert admin.can_access(HonorificLevel.HASIPSIOCHE)
        assert admin.can_access(HonorificLevel.HAERACHE)
        assert user.can_access(HonorificLevel.HAERACHE)
        assert not system.can_access(HonorificLevel.HASIPSIOCHE)
        assert not system.can_access(HonorificLevel.HAEOYO)

    def test_korean_name(self):
        assert "하십시오체" in HonorificLevel.HASIPSIOCHE.korean_name
        assert "해요체" in HonorificLevel.HAEOYO.korean_name
        assert "해체" in HonorificLevel.HAE.korean_name
        assert "해라체" in HonorificLevel.HAERACHE.korean_name

    def test_detect_ending_hasipsioche(self):
        validator = HonorificValidator()
        assert validator.detect_from_ending("결과를 보고하십시오") == HonorificLevel.HASIPSIOCHE
        assert validator.detect_from_ending("작업이 완료되었습니다") == HonorificLevel.HASIPSIOCHE

    def test_detect_ending_haeyoche(self):
        validator = HonorificValidator()
        assert validator.detect_from_ending("계산을 시작해요") == HonorificLevel.HAEOYO
        assert validator.detect_from_ending("이에요") == HonorificLevel.HAEOYO

    def test_detect_ending_haerache(self):
        validator = HonorificValidator()
        assert validator.detect_from_ending("작업을 수행한다") == HonorificLevel.HAERACHE
        assert validator.detect_from_ending("계산한다") == HonorificLevel.HAERACHE

    def test_detect_unknown_returns_default(self):
        validator = HonorificValidator(default_level=HonorificLevel.HAERACHE)
        assert validator.detect("계산 3 더하기 5") == HonorificLevel.HAERACHE

    def test_consistency_check_passes(self):
        validator = HonorificValidator()
        ok, errors = validator.validate_consistency([
            "작업을 수행합니다",
            "결과를 보고합니다",
            "완료되었습니다",
        ])
        assert ok
        assert len(errors) == 0

    def test_consistency_check_fails(self):
        validator = HonorificValidator()
        ok, errors = validator.validate_consistency([
            "작업을 수행합니다",    # 하십시오체
            "계산을 해요",          # 해요체
        ])
        assert not ok
        assert len(errors) == 1

    def test_cap_require_opcode(self):
        validator = HonorificValidator()
        opcode = validator.get_cap_require_opcode("보고하십시오")
        assert opcode == (Opcode.CAP_REQUIRE, HonorificLevel.HASIPSIOCHE.value)


# ═══════════════════════════════════════════════════════════
# 3. 권한/CAP 테스트
# ═══════════════════════════════════════════════════════════

class TestCapability:
    """권한 검사 테스트"""

    def test_admin_can_do_everything(self):
        vm = VM()
        vm.set_cap_level(4)  # admin
        # admin은 모든 CAP_REQUIRE를 통과
        vm.execute([
            (Opcode.CAP_REQUIRE, 4),
            (Opcode.CAP_REQUIRE, 3),
            (Opcode.CAP_REQUIRE, 2),
            (Opcode.CAP_REQUIRE, 1),
        ])
        assert True  # 예외 없이 통과

    def test_system_cannot_access_admin(self):
        vm = VM()
        vm.set_cap_level(1)  # system
        with pytest.raises(CapabilityError, match="권한 부족"):
            vm.execute([(Opcode.CAP_REQUIRE, 4)])

    def test_user_cannot_access_admin(self):
        vm = VM()
        vm.set_cap_level(3)  # standard user
        with pytest.raises(CapabilityError, match="권한 부족"):
            vm.execute([(Opcode.CAP_REQUIRE, 4)])

    def test_user_can_access_own_level(self):
        vm = VM()
        vm.set_cap_level(3)
        vm.execute([(Opcode.CAP_REQUIRE, 3)])
        assert True

    def test_cap_set_changes_level(self):
        vm = VM()
        vm.set_cap_level(1)
        vm.execute([
            (Opcode.CAP_SET, 4),  # admin으로 승격
            (Opcode.CAP_REQUIRE, 4),
        ])
        assert vm.cap_level == 4


# ═══════════════════════════════════════════════════════════
# 4. 한자어 숫자 테스트
# ═══════════════════════════════════════════════════════════

class TestKoreanNumeral:
    """한자어 숫자 변환 테스트"""

    def test_basic_numerals(self):
        assert KoreanNumeral.to_int("영") == 0
        assert KoreanNumeral.to_int("일") == 1
        assert KoreanNumeral.to_int("이") == 2
        assert KoreanNumeral.to_int("삼") == 3
        assert KoreanNumeral.to_int("사") == 4
        assert KoreanNumeral.to_int("오") == 5
        assert KoreanNumeral.to_int("육") == 6
        assert KoreanNumeral.to_int("칠") == 7
        assert KoreanNumeral.to_int("팔") == 8
        assert KoreanNumeral.to_int("구") == 9

    def test_ten(self):
        assert KoreanNumeral.to_int("십") == 10

    def test_arabic_numerals(self):
        assert KoreanNumeral.to_int("42") == 42
        assert KoreanNumeral.to_int("100") == 100

    def test_is_numeral(self):
        assert KoreanNumeral.is_numeral("삼") is True
        assert KoreanNumeral.is_numeral("42") is True
        assert KoreanNumeral.is_numeral("abc") is False

    def test_invalid_raises(self):
        with pytest.raises(ParseError):
            KoreanNumeral.to_int("존재하지않음")


# ═══════════════════════════════════════════════════════════
# 5. 조사 테스트
# ═══════════════════════════════════════════════════════════

class TestParticles:
    """조사 정의 테스트"""

    def test_topic_particles(self):
        assert "은" in Particles.TOPIC
        assert "는" in Particles.TOPIC

    def test_subject_particles(self):
        assert "이" in Particles.SUBJECT
        assert "가" in Particles.SUBJECT

    def test_object_particles(self):
        assert "을" in Particles.OBJECT
        assert "를" in Particles.OBJECT

    def test_direction_particles(self):
        assert "에" in Particles.DIRECTION
        assert "에게" in Particles.DIRECTION

    def test_genitive_particle(self):
        assert "의" in Particles.GENITIVE


# ═══════════════════════════════════════════════════════════
# 6. SOV 구조 테스트
# ═══════════════════════════════════════════════════════════

class TestSOVStructure:
    """SOV→CPS 변환 테스트"""

    def test_sov_continuation(self):
        sov = SOVContinuation(["레지스터", "영에", "5"], "대입하세요")
        assert sov.verb == "대입하세요"
        assert sov.arguments == ["레지스터", "영에", "5"]

    def test_sov_verb_at_end(self):
        """한국어 SOV: 동사가 항상 마지막"""
        sov = SOVContinuation(["주어", "목적어"], "동사합니다")
        assert sov.verb == "동사합니다"
        assert len(sov.arguments) == 2


# ═══════════════════════════════════════════════════════════
# 7. 한국어 NL 패턴 테스트
# ═══════════════════════════════════════════════════════════

class TestKoreanPatterns:
    """한국어 자연어 패턴 컴파일 및 실행"""

    def test_calc_add(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("계산 3 더하기 5")
        assert any("결과" in o and "8" in o for o in output)

    def test_multiply(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("7 곱하기 6")
        assert any("42" in o for o in output)

    def test_sum_range(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("1 부터 10 까지 합")
        assert any("55" in o for o in output)

    def test_factorial(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("5 의 팩토리얼")
        assert any("120" in o for o in output)

    def test_register_store(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("레지스터 영 에 42 대입")
        assert interp.vm.registers[0] == 42

    def test_register_add(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute(
            "레지스터 영 에 3 대입\n"
            "레지스터 일 에 5 대입\n"
            "레지스터 영 더하기 레지스터 일"
        )
        assert interp.vm.registers[0] == 8

    def test_send_message(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("함장에게 상황 정상 전달")
        assert any("함장" in o and "상황 정상" in o for o in output)

    def test_question(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("조타수에게 현재 좌표 질문")
        assert any("조타수" in o and "현재 좌표" in o for o in output)

    def test_broadcast(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute("전체에 작업 완료 방송")
        assert any("전체" in o and "작업 완료" in o for o in output)

    def test_multiline_program(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        source = (
            "# 계산 프로그램\n"
            "계산 10 더하기 20\n"
            "5 곱하기 6\n"
            "1 부터 5 까지 합\n"
            "5 의 팩토리얼"
        )
        output = interp.execute(source)
        assert len(output) >= 4  # 최소 4줄 출력

    def test_empty_and_comments(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        output = interp.execute(
            "# 주석입니다\n"
            "\n"
            "# 또 주석\n"
        )
        assert len(output) == 0


# ═══════════════════════════════════════════════════════════
# 8. 컴파일 테스트
# ═══════════════════════════════════════════════════════════

class TestCompilation:
    """바이트코드 컴파일 테스트"""

    def test_compile_only(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        bytecode = interp.compile_only("계산 3 더하기 5")
        assert len(bytecode) > 0
        # 첫 명령어는 CAP_REQUIRE
        assert bytecode[0][0] == Opcode.CAP_REQUIRE

    def test_format_bytecode(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        interp.compile_only("계산 3 더하기 5")
        formatted = interp.format_bytecode()
        assert "CAP_REQUIRE" in formatted
        assert "LOAD_IMM" in formatted
        assert "ADD" in formatted

    def test_parse_history(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        interp.execute("계산 3 더하기 5")
        assert len(interp.parse_history) == 1
        assert interp.parse_history[0].verb != ""

    def test_sov_analysis(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        interp.execute("레지스터 영 에 42 대입")
        assert len(interp.parse_history) == 1
        # SOV에서 인자는 앞부분, 동사는 마지막
        sov = interp.parse_history[0]
        assert sov.verb.endswith("대입")


# ═══════════════════════════════════════════════════════════
# 9. 인터프리터 상태 테스트
# ═══════════════════════════════════════════════════════════

class TestInterpreterState:
    """인터프리터 상태 관리 테스트"""

    def test_get_state(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        interp.execute("레지스터 영 에 99 대입")
        state = interp.get_state()
        assert state["registers"][0] == 99
        assert state["cap_level"] == 1

    def test_reset(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        interp.execute("레지스터 영 에 99 대입")
        interp.reset()
        assert interp.vm.registers[0] == 0
        assert interp.vm.pc == 0

    def test_dump_registers(self):
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        interp.execute("레지스터 영 에 42 대입")
        dump = interp.vm.dump_registers()
        assert "R0" in dump
        assert "42" in dump

    def test_register_names_korean(self):
        """레지스터 이름이 한자어로 출력되는지"""
        vm = VM()
        dump = vm.dump_registers()
        assert "영" in dump
        assert "일" in dump


# ═══════════════════════════════════════════════════════════
# 10. 통합 테스트
# ═══════════════════════════════════════════════════════════

class TestIntegration:
    """전체 시스템 통합 테스트"""

    def test_full_calculation_program(self):
        """완전한 계산 프로그램"""
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        source = """
# 팩토리얼과 합계 계산
5 의 팩토리얼
1 부터 10 까지 합
3 곱하기 7
계산 100 더하기 200
"""
        output = interp.execute(source)
        assert len(output) >= 4
        # 마지막 명령이 "계산 100 더하기 200" → R0 = 100
        state = interp.get_state()
        assert state["registers"][0] == 100

    def test_communication_program(self):
        """에이전트 통신 프로그램"""
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        source = """
함장에게 상황 정상 전달
조타수에게 현재 좌표 질문
전체에 작업 시작 방송
"""
        output = interp.execute(source)
        # 각 명령어에 대한 출력 + 경어 CAP
        assert len(output) >= 3
        assert len(interp.vm.messages) >= 1

    def test_register_operations(self):
        """레지스터 연산 프로그램"""
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        source = """
레지스터 영 에 10 대입
레지스터 일 에 20 대입
레지스터 영 더하기 레지스터 일
"""
        output = interp.execute(source)
        assert interp.vm.registers[0] == 30

    def test_capability_with_honorific_enforcement(self):
        """경어 강제 검증 모드"""
        interp = FluxInterpreterKor(
            honorific_level=HonorificLevel.HAERACHE,
            enforce_honorifics=True,
        )
        # 해라체 수준에서 하십시오체 문장 실행 → 오류
        with pytest.raises(Exception):
            interp.execute("레지스터 영 에 42 대입하십시오")

    def test_honorific_bytecode_injection(self):
        """컴파일된 바이트코드에 CAP_REQUIRE가 포함되는지"""
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAERACHE)
        bytecode = interp.compile_only("계산 3 더하기 5")
        cap_ops = [bc for bc in bytecode if bc[0] == Opcode.CAP_REQUIRE]
        assert len(cap_ops) >= 1

    def test_send_records_in_messages(self):
        """전송 메시지가 기록되는지"""
        interp = FluxInterpreterKor(honorific_level=HonorificLevel.HAE)
        interp.execute("항해사에게 항로 질문")
        assert len(interp.vm.messages) == 1
        msg_type, agent, content = interp.vm.messages[0]
        assert msg_type == "send"
        assert "항해사" in agent
