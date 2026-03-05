#!/usr/bin/env python3
"""Generate a local API/contract snapshot for installed laboneq_applications.

Run this script with the repository virtual environment Python so the snapshot reflects
the exact package version used for experiment development.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _decorator_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _format_args(args: ast.arguments) -> str:
    arg_names: list[str] = []
    arg_names.extend(arg.arg for arg in args.posonlyargs)
    arg_names.extend(arg.arg for arg in args.args)
    if args.vararg:
        arg_names.append(f"*{args.vararg.arg}")
    arg_names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg:
        arg_names.append(f"**{args.kwarg.arg}")
    return ", ".join(arg_names)


def _safe_version() -> str:
    for name in ("laboneq-applications", "laboneq_applications"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _load_python_file(path: Path) -> tuple[ast.Module, str]:
    text = path.read_text(encoding="utf-8")
    return ast.parse(text, filename=str(path)), text


def _is_workflow_decorator(name: str) -> bool:
    return name.endswith("workflow.workflow") or name == "workflow.workflow"


def _is_task_decorator(name: str) -> bool:
    return name.endswith("workflow.task") or name == "workflow.task"


def _is_options_decorator(name: str) -> bool:
    return (
        name.endswith("workflow.workflow_options")
        or name.endswith("workflow.task_options")
        or name == "workflow_options"
        or name == "task_options"
    )


def _module_name(package_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(package_root).with_suffix("")
    return ".".join(("laboneq_applications", *rel.parts))


def _collect_files(package_root: Path, relative_dir: str) -> list[Path]:
    base = package_root / relative_dir
    if not base.exists():
        return []
    return sorted(
        p
        for p in base.rglob("*.py")
        if p.name != "__init__.py" and "__pycache__" not in p.parts
    )


def _find_package_root() -> Path:
    spec = importlib.util.find_spec("laboneq_applications")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Cannot find installed package 'laboneq_applications'. "
            "Run with the repository .venv Python."
        )
    return Path(spec.origin).resolve().parent


def _summarize_experiment_modules(package_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for file_path in _collect_files(package_root, "experiments"):
        tree, text = _load_python_file(file_path)
        workflow_fn = "-"
        create_fn = "-"
        task_count = 0
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = [_decorator_name(d) for d in node.decorator_list]
            if any(_is_workflow_decorator(d) for d in decorators):
                workflow_fn = f"{node.name}({ _format_args(node.args) })"
            if node.name == "create_experiment" or any(
                d.endswith("dsl.qubit_experiment") or d == "dsl.qubit_experiment"
                for d in decorators
            ):
                create_fn = f"{node.name}({ _format_args(node.args) })"
            if any(_is_task_decorator(d) for d in decorators):
                task_count += 1
        rows.append(
            {
                "module": _module_name(package_root, file_path),
                "workflow": workflow_fn,
                "create": create_fn,
                "temporary_qpu": "yes" if "temporary_qpu(" in text else "no",
                "temporary_elements": "yes"
                if "temporary_quantum_elements_from_qpu(" in text
                else "no",
                "update_qpu": "yes" if "update_qpu(" in text else "no",
                "task_count": str(task_count),
            }
        )
    return rows


def _summarize_analysis_modules(package_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for file_path in _collect_files(package_root, "analysis"):
        tree, text = _load_python_file(file_path)
        workflow_fn = "-"
        task_count = 0
        returns_new_values = "yes" if "new_parameter_values" in text else "no"
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = [_decorator_name(d) for d in node.decorator_list]
            if any(_is_workflow_decorator(d) for d in decorators):
                workflow_fn = f"{node.name}({ _format_args(node.args) })"
            if any(_is_task_decorator(d) for d in decorators):
                task_count += 1
        rows.append(
            {
                "module": _module_name(package_root, file_path),
                "workflow": workflow_fn,
                "task_count": str(task_count),
                "new_parameter_values": returns_new_values,
            }
        )
    return rows


def _summarize_options_classes(
    package_root: Path, relative_path: str
) -> list[dict[str, str]]:
    file_path = package_root / relative_path
    if not file_path.exists():
        return []
    tree, _ = _load_python_file(file_path)
    rows: list[dict[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = [_decorator_name(d) for d in node.decorator_list]
        if not any(_is_options_decorator(d) for d in decorators):
            continue
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(_decorator_name(base))
        rows.append(
            {
                "class": node.name,
                "module": _module_name(package_root, file_path),
                "decorators": ", ".join(decorators) if decorators else "-",
                "bases": ", ".join(base_names) if base_names else "-",
            }
        )
    return rows


def _summarize_contrib_modules(package_root: Path, relative_dir: str) -> list[str]:
    return [_module_name(package_root, p) for p in _collect_files(package_root, relative_dir)]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["(none)"]
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [line1, line2]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_markdown(package_root: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    version = _safe_version()
    experiments = _summarize_experiment_modules(package_root)
    analysis = _summarize_analysis_modules(package_root)
    contrib_experiments = _summarize_contrib_modules(package_root, "contrib/experiments")
    contrib_analysis = _summarize_contrib_modules(package_root, "contrib/analysis")
    exp_options = _summarize_options_classes(package_root, "experiments/options.py")
    ana_options = _summarize_options_classes(package_root, "analysis/options.py")

    workflow_names = Counter()
    for row in experiments:
        if row["workflow"] != "-":
            workflow_names[row["workflow"].split("(")[0]] += 1
    for row in analysis:
        if row["workflow"] != "-":
            workflow_names[row["workflow"].split("(")[0]] += 1

    lines: list[str] = []
    lines.append("# laboneq_applications venv snapshot")
    lines.append("")
    lines.append(f"- generated_utc: `{now}`")
    lines.append(f"- python: `{sys.executable}`")
    lines.append(f"- package_version: `{version}`")
    lines.append(f"- package_path: `{package_root}`")
    lines.append("")
    lines.append("## Inventory")
    lines.append(f"- experiments modules: `{len(experiments)}`")
    lines.append(f"- analysis modules: `{len(analysis)}`")
    lines.append(f"- contrib experiment modules: `{len(contrib_experiments)}`")
    lines.append(f"- contrib analysis modules: `{len(contrib_analysis)}`")
    lines.append("")
    lines.append("## Core experiment contracts")
    exp_rows = [
        [
            r["module"],
            r["workflow"],
            r["create"],
            r["temporary_qpu"],
            r["temporary_elements"],
            r["update_qpu"],
            r["task_count"],
        ]
        for r in experiments
    ]
    lines.extend(
        _markdown_table(
            [
                "module",
                "workflow",
                "create_experiment",
                "temporary_qpu",
                "temporary_quantum_elements",
                "update_qpu",
                "task_count",
            ],
            exp_rows,
        )
    )
    lines.append("")
    lines.append("## Core analysis contracts")
    ana_rows = [
        [r["module"], r["workflow"], r["task_count"], r["new_parameter_values"]]
        for r in analysis
    ]
    lines.extend(
        _markdown_table(
            ["module", "workflow", "task_count", "mentions_new_parameter_values"],
            ana_rows,
        )
    )
    lines.append("")
    lines.append("## Options classes")
    opt_rows = [
        [r["module"], r["class"], r["decorators"], r["bases"]]
        for r in (exp_options + ana_options)
    ]
    lines.extend(_markdown_table(["module", "class", "decorators", "bases"], opt_rows))
    lines.append("")
    lines.append("## Contrib modules")
    lines.append("- contrib.experiments:")
    for mod in contrib_experiments:
        lines.append(f"  - `{mod}`")
    lines.append("- contrib.analysis:")
    for mod in contrib_analysis:
        lines.append(f"  - `{mod}`")
    lines.append("")
    lines.append("## Canonical naming frequency")
    if workflow_names:
        for name, count in workflow_names.most_common():
            lines.append(f"- `{name}`: `{count}` modules")
    else:
        lines.append("- no workflow functions found")
    lines.append("")
    lines.append("## New module guardrails inferred from snapshot")
    lines.append("- Define workflow entrypoints with `@workflow.workflow`.")
    lines.append("- Keep experiment modules centered on `experiment_workflow` + `create_experiment`.")
    lines.append(
        "- Keep runtime overrides on temporary copies via `temporary_qpu` and `temporary_quantum_elements_from_qpu`."
    )
    lines.append(
        "- For update-capable workflows, keep persistent updates on "
        "`analysis_results.output[\"new_parameter_values\"]` + `update_qpu`."
    )
    lines.append("- Use option classes based on `@task_options` / `@workflow_options` from applications patterns.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot installed laboneq_applications contracts into markdown."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write markdown to file. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    try:
        package_root = _find_package_root()
        markdown = _render_markdown(package_root)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
