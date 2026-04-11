"""
유체 한국어 CLI — 한국어 우선 명령행 인터페이스

명령어:
  안녕           인사 / 버전 정보
  컴파일 <파일>   한국어 소스를 바이트코드로 컴파일
  실행 <파일>     한국어 소스 실행
  열기 <파일>     소스 열어보기
  해체 <파일>     바이트코드 디스어셈블

플래그:
  --존댓말        경어 수준을 해요체(standard user)로 설정
  --반말          경어 수준을 해라체(internal/system)로 설정
  --격식          경어 수준을 하십시오체(admin)로 설정
  --검증          경어 수준 일관성 검증 활성화
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from flux_kor.honorifics import HonorificLevel
from flux_kor.interpreter import FluxInterpreterKor

BANNER = r"""
╔══════════════════════════════════════════════════╗
║  유체 · 流體言語通用執行                          ║
║  FLUX — Korean-first Natural Language Runtime     ║
║  경어 시스템 기반 RBAC · SOV→CPS 컴파일          ║
╚══════════════════════════════════════════════════╝
"""

HELP_TEXT = """
사용법:
  flux-kor [옵션] <명령> [인자...]

명령어:
  안녕           인사 및 버전 정보 표시
  컴파일 <파일>   한국어 소스를 바이트코드로 컴파일
  실행 <파일>     한국어 소스 컴파일 후 실행
  열기 <파일>     소스 파일 내용 표시
  해체 <파일>     바이트코드 디스어셈블

옵션:
  --존댓말         해요체 (standard user, CAP level 3)
  --반말           해라체 (internal/system, CAP level 1)
  --격식           하십시오체 (admin, CAP level 4)
  --검증           경어 수준 일관성 검증 활성화
  --입력 <코드>    명령행에서 직접 코드 실행

예시:
  flux-kor --반말 --입력 "계산 3 더하기 5"
  flux-kor --존댓말 실행 program.kor
  flux-kor --검증 컴파일 program.kor
"""


def _determine_honorific_level(args: argparse.Namespace) -> HonorificLevel:
    """명령행 인자에서 경어 수준 결정"""
    if getattr(args, "격식", False):
        return HonorificLevel.HASIPSIOCHE
    if getattr(args, "존댓말", False):
        return HonorificLevel.HAEOYO
    if getattr(args, "반말", False):
        return HonorificLevel.HAERACHE
    # 기본값: 반말 (system/internal)
    return HonorificLevel.HAERACHE


def cmd_greet(args: argparse.Namespace) -> int:
    """안녕 — 인사"""
    level = _determine_honorific_level(args)

    greetings = {
        HonorificLevel.HASIPSIOCHE: "안녕하십니까. 유체 실행시에 오신 것을 환영합니다.",
        HonorificLevel.HAEOYO: "안녕하세요! 유체 실행시에 오신 것을 환영해요.",
        HonorificLevel.HAE: "안녕! 유체 실행시에 온 걸 환영해.",
        HonorificLevel.HAERACHE: "안녕하라. 유체 실행시다.",
    }

    print(BANNER)
    print(f"  {greetings[level]}")
    print(f"  경어 수준: {level.korean_name} (권한: {level.role_name})")
    print(f"  버전: 0.1.0")
    print()
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """컴파일 — 한국어 소스를 바이트코드로"""
    if not args.file:
        print("오류: 파일을 지정하세요.  flux-kor 컴파일 <파일>")
        return 1

    path = Path(args.file)
    if not path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 — {path}")
        return 1

    level = _determine_honorific_level(args)
    enforce = getattr(args, "검증", False)

    interp = FluxInterpreterKor(
        honorific_level=level,
        enforce_honorifics=enforce,
    )

    source = path.read_text(encoding="utf-8")
    print(f"컴파일 중: {path}")
    print(f"경어 수준: {level.korean_name}")
    print()

    try:
        bytecode = interp.compile_only(source)
        print(interp.format_bytecode(bytecode))
        print(f"\n총 {len(bytecode)} 개 명령어")
        return 0
    except Exception as e:
        print(f"컴파일 오류: {e}")
        return 1


def cmd_execute(args: argparse.Namespace) -> int:
    """실행 — 한국어 소스 컴파일 후 실행"""
    level = _determine_honorific_level(args)
    enforce = getattr(args, "검증", False)

    interp = FluxInterpreterKor(
        honorific_level=level,
        enforce_honorifics=enforce,
    )

    # 파일 또는 직접 입력
    source = getattr(args, "입력", None)
    if source:
        source_text = source
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"오류: 파일을 찾을 수 없습니다 — {path}")
            return 1
        source_text = path.read_text(encoding="utf-8")
    else:
        print("오류: 파일 또는 --입력을 지정하세요.")
        return 1

    print(f"경어 수준: {level.korean_name} (권한: {level.role_name})")
    print("─" * 40)

    try:
        output = interp.execute(source_text)
        for line in output:
            print(f"  {line}")

        print("─" * 40)
        print(interp.vm.dump_registers())
        return 0
    except Exception as e:
        print(f"실행 오류: {e}")
        return 1


def cmd_open(args: argparse.Namespace) -> int:
    """열기 — 소스 파일 내용 표시"""
    if not args.file:
        print("오류: 파일을 지정하세요.")
        return 1

    path = Path(args.file)
    if not path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 — {path}")
        return 1

    content = path.read_text(encoding="utf-8")
    print(f"📄 {path}")
    print("─" * 40)
    print(content)
    return 0


def cmd_disassemble(args: argparse.Namespace) -> int:
    """해체 — 바이트코드 디스어셈블"""
    if not args.file:
        print("오류: 파일을 지정하세요.")
        return 1

    path = Path(args.file)
    if not path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 — {path}")
        return 1

    level = _determine_honorific_level(args)
    interp = FluxInterpreterKor(honorific_level=level)

    source = path.read_text(encoding="utf-8")
    print(f"해체 중: {path}")
    print()

    try:
        bytecode = interp.compile_only(source)
        print(interp.format_bytecode(bytecode))

        # SOV 분석 결과도 표시
        if interp.parse_history:
            print("\nSOV 구조 분석:")
            print("─" * 40)
            for i, sov in enumerate(interp.parse_history):
                print(f"  문장 {i + 1}: 주어/인자={sov.arguments} → 동사={sov.verb}")
        return 0
    except Exception as e:
        print(f"해체 오류: {e}")
        return 1


def cmd_repl(args: argparse.Namespace) -> int:
    """대화형 REPL"""
    level = _determine_honorific_level(args)
    enforce = getattr(args, "검증", False)

    interp = FluxInterpreterKor(
        honorific_level=level,
        enforce_honorifics=enforce,
    )

    greetings = {
        HonorificLevel.HASIPSIOCHE: "무엇을 도와드릴까요? (끝내려면 '종료'를 입력하세요)",
        HonorificLevel.HAEOYO: "무엇을 도와줄까요? (끝내려면 '종료'를 입력해요)",
        HonorificLevel.HAE: "뭐 도와줄까? (끝내려면 '종료' 입력)",
        HonorificLevel.HAERACHE: "명령을 내리라. (종료를 입력하면 끝난다)",
    }

    print(BANNER)
    print(f"  {greetings[level]}")
    print(f"  경어 수준: {level.korean_name}")
    print()

    while True:
        try:
            line = input("유체> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("종료", "exit", "quit", "끝"):
            farewells = {
                HonorificLevel.HASIPSIOCHE: "안녕히 가십시오.",
                HonorificLevel.HAEOYO: "안녕히 가세요!",
                HonorificLevel.HAE: "잘 가!",
                HonorificLevel.HAERACHE: "가라.",
            }
            print(farewells[level])
            break
        if line == "상태":
            state = interp.get_state()
            print(interp.vm.dump_registers())
            continue
        if line == "도움":
            print(HELP_TEXT)
            continue

        try:
            output = interp.execute(line)
            for o in output:
                print(f"  {o}")
        except Exception as e:
            print(f"  오류: {e}")

    return 0


def main() -> int:
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        prog="flux-kor",
        description="유체 · 流體言語通用執行 — 한국어 우선 자연어 런타임",
        add_help=False,
        allow_abbrev=False,
    )

    # 옵션
    parser.add_argument("--존댓말", action="store_true", help="경어 수준: 해요체 (standard user)")
    parser.add_argument("--반말", action="store_true", help="경어 수준: 해라체 (system)")
    parser.add_argument("--격식", action="store_true", help="경어 수준: 하십시오체 (admin)")
    parser.add_argument("--검증", action="store_true", help="경어 수준 일관성 검증")
    parser.add_argument("--입력", dest="입력", type=str, help="직접 코드 입력")
    parser.add_argument("--help", "-h", action="store_true", help="도움말")
    parser.add_argument("--version", "-v", action="store_true", help="버전 정보")

    # 명령어 (위치 인자)
    parser.add_argument("command", nargs="?", default=None,
                        help="명령어: 안녕|컴파일|실행|열기|해체")
    parser.add_argument("file", nargs="?", default=None,
                        help="파일 경로")

    args = parser.parse_args()

    # 버전
    if args.version:
        print("유체 (FLUX-kor) 버전 0.1.0")
        return 0

    # 도움말
    if args.help:
        print(HELP_TEXT)
        return 0

    # 명령어 분기
    command = args.command

    if command is None:
        # 인자 없으면 REPL
        return cmd_repl(args)

    # 명령어 → 한국어 → 함수 매핑
    cmd_map = {
        "안녕": cmd_greet,
        "인사": cmd_greet,
        "컴파일": cmd_compile,
        "compile": cmd_compile,
        "실행": cmd_execute,
        "run": cmd_execute,
        "열기": cmd_open,
        "open": cmd_open,
        "보기": cmd_open,
        "해체": cmd_disassemble,
        "disassemble": cmd_disassemble,
    }

    handler = cmd_map.get(command)
    if handler is None:
        # 명령어를 직접 입력 코드로 처리
        args.입력 = command
        args.file = None
        return cmd_execute(args)

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
