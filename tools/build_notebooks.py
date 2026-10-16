"""Build standalone contest notebooks from reviewed source files (stdlib only)."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "day1_find_the_order": "Find the Order",
    "day1_robot_chasing": "Robot Chasing",
    "day1_potato": "Potato",
    "day2_double_agent_dilemma": "Double Agent Dilemma",
    "day2_ghost_of_the_machine": "Ghost of the Machine",
    "day2_ioai_field": "IOAI Field",
}


def remove_main_guard(source):
    """Remove only top-level __name__ guards; never rewrite function bodies."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in reversed(tree.body):
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id == "__name__"
            for child in ast.walk(node.test)
        ):
            del lines[node.lineno - 1:node.end_lineno]
    return "".join(lines).rstrip() + "\n"


def cell(kind, source, identifier):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
              "id": identifier}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook_for(slug):
    source = (ROOT / "solutions" / f"{slug}.py").read_text(encoding="utf-8")
    definitions = remove_main_guard(source)
    intro = (
        f"# IOAI 2026 — {TASKS[slug]}\n\n"
        "공식 자료를 참고한 학습용 제안 풀이입니다. 공식 정답 또는 최고점 재현 코드가 아닙니다.\n\n"
        "해당 문제의 공식 작업 폴더에서 실행하세요. 공개 데이터와 허용 모델이 필요하며, "
        "외부 파일 다운로드나 숨겨진 정답 읽기를 수행하지 않습니다.\n\n"
        f"한국어 해설: https://github.com/Sy5h/IOAI/blob/main/docs/{slug}.ko.md\n\n"
        "코드는 이 노트북 안에 포함되어 있습니다. 아래 실행 셀의 인자를 바꾸어 "
        "경로·배치 크기·시간 예산을 조절할 수 있습니다.\n"
    )
    if slug == "day1_potato":
        intro += "\n대화형 문제: Run All은 응답 클래스를 준비합니다. 공식 local_test.py 또는 grader가 노트북을 Python으로 변환해 stdin/stdout으로 실행합니다.\n"
        entry = (
            "# Jupyter에서는 stdin을 기다리지 않고, 공식 nbconvert 실행에서만 시작합니다.\n"
            "if __name__ == '__main__' and '__file__' in globals():\n"
            "    main()\n"
            "else:\n"
            "    print('Player ready. Use the official local_test.py or grader.')\n"
        )
    else:
        entry = "# 이 셀에서 한 번만 실행합니다. 예: main(['--output', 'answers.json'])\nmain([])\n"
    cells = [cell("markdown", intro, "introduction")]
    if slug == "day2_ioai_field":
        cells.append(cell("code", "from pathlib import Path\nif not Path('custom_model.py').is_file():\n    raise FileNotFoundError('Copy the supplied custom_model.py beside solution.ipynb.')\n", "model-file-check"))
    cells += [cell("code", definitions, "solution-source"), cell("code", entry, "run-once")]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "ioai_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(check=False):
    errors = []
    for slug in TASKS:
        directory = ROOT / "notebooks" / slug
        content = json.dumps(notebook_for(slug), ensure_ascii=False, indent=1) + "\n"
        path = directory / "solution.ipynb"
        limit = 20_000_000 if "ghost" in slug else 1_000_000
        companion = (ROOT / "solutions" / "field_custom_model.py").read_text(encoding="utf-8") if "ioai_field" in slug else ""
        if len(content.encode()) + len(companion.encode()) > limit:
            raise ValueError(f"{slug} exceeds its notebook/file size limit")
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                errors.append(str(path.relative_to(ROOT)))
            if companion and (not (directory / "custom_model.py").is_file() or
                              (directory / "custom_model.py").read_text(encoding="utf-8") != companion):
                errors.append(str((directory / "custom_model.py").relative_to(ROOT)))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            if companion:
                (directory / "custom_model.py").write_text(companion, encoding="utf-8", newline="\n")
        print(f"{slug}: {len(content.encode())} bytes")
    if errors:
        raise SystemExit("Notebooks are stale. Run python tools/build_notebooks.py:\n" + "\n".join(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)

