"""Formal v0.7 documentation, archive, and retired-entry contracts."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

CURRENT_FORMAL_PATHS = (
    "README.md",
    "docs/DETERMINISTIC_RUNTIME.md",
    "docs/PROJECT_OVERVIEW.md",
    "experiments/EXPERIMENT_PLAN.md",
    "services/world_r1_strict/README.md",
)
W06_FORMAL_PATHS = (
    "docs/V0_7_USER_GUIDE.md",
    "docs/V0_7_ACCEPTANCE.md",
    "experiments/v0_7/README.md",
    "CHANGELOG.md",
)
FORMAL_PATHS = CURRENT_FORMAL_PATHS + W06_FORMAL_PATHS

ARCHIVE_PATHS = (
    "experiments/archive/v0_6/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md",
    "experiments/archive/v0_6/INFRA_VALIDATION_WORKLOG.md",
)
ARCHIVE_SOURCES = (
    "experiments/EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md",
    "experiments/INFRA_VALIDATION_WORKLOG.md",
)
ARCHIVE_BANNER = (
    "> Historical v0.6 evidence; not a v0.7 usage/config contract."
)

FENCE_RE = re.compile(
    r"^```(?P<language>[^\n]*)\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)")

RETIRED_COMMAND_PATTERNS = (
    re.compile(r"(?m)^\s*(?:\$\s*)?visual-rl(?:\s|$)"),
    re.compile(r"(?m)^\s*(?:\$\s*)?python(?:3)?\s+train\.py(?:\s|$)"),
)
RETIRED_FENCED_PATTERNS = (
    re.compile(
        r"preset:[A-Za-z0-9_.-]+|visual_rl/configs/presets|--plugin|"
        r"plugin_modules|visual_rl\.plugins|"
        r"register_(?:model_adapter|rollout_engine|reward_client|algorithm|"
        r"optimizer_plugin)"
    ),
    re.compile(
        r"\b(?:ExperimentRunner|ComponentRegistry|RunCallback)\s*\(|"
        r"visual_rl\.(?:callbacks|evaluation)"
    ),
    re.compile(r"--set|--resume|\bload_config\s*\(|\bvalidate_config\s*\("),
)
RETIRED_YAML_TOP_LEVEL_RE = re.compile(
    r"(?m)^(?:sample|train|runner|paths|rewards|plugins|evaluation|callbacks):"
    r"\s*(?:$|#)"
)

REQUIRED_YAML_TOP_LEVEL = {
    "schema_version",
    "run",
    "model",
    "dataset",
    "rollout",
    "reward",
    "algorithm",
    "optimizer",
    "runtime",
    "artifacts",
    "resume",
}


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _fences(text: str) -> list[tuple[str, str]]:
    return [
        (match.group("language").strip().lower(), match.group("body"))
        for match in FENCE_RE.finditer(text)
    ]


def _relative_link_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return unquote(parsed.path)


def test_exact_formal_documents_exist_and_navigation_is_current() -> None:
    assert all((ROOT / path).is_file() for path in FORMAL_PATHS)

    readme = _read("README.md")
    overview = _read("docs/PROJECT_OVERVIEW.md")
    experiment_plan = _read("experiments/EXPERIMENT_PLAN.md")

    for link in (
        "docs/V0_7_USER_GUIDE.md",
        "docs/V0_7_ACCEPTANCE.md",
        "experiments/v0_7/README.md",
    ):
        assert link in readme
    for link in (
        "V0_7_USER_GUIDE.md",
        "V0_7_ACCEPTANCE.md",
        "../experiments/EXPERIMENT_PLAN.md",
    ):
        assert link in overview
    assert "v0_7/README.md" in experiment_plan

    current_navigation = "\n".join((readme, overview, experiment_plan))
    for historical_name in (
        "EXPERIMENT_RESULTS_SUMMARY_2026-07-15.md",
        "INFRA_VALIDATION_WORKLOG.md",
    ):
        assert historical_name not in current_navigation


def test_readme_exposes_one_complete_yaml_and_fixed_launch_script() -> None:
    readme = _read("README.md")
    yaml_blocks = [
        body for language, body in _fences(readme) if language in {"yaml", "yml"}
    ]
    assert len(yaml_blocks) == 1

    top_level = {
        match.group(1)
        for match in re.finditer(r"(?m)^([a-z][a-z0-9_]*):(?:\s|$)", yaml_blocks[0])
    }
    assert top_level == REQUIRED_YAML_TOP_LEVEL
    assert "python run_experiment.py" in readme
    assert (
        "torchrun --standalone --nproc-per-node=2 run_experiment.py" in readme
    )


def test_formal_docs_do_not_advertise_retired_executable_paths() -> None:
    for relative_path in FORMAL_PATHS:
        text = _read(relative_path)
        for pattern in RETIRED_COMMAND_PATTERNS:
            assert pattern.search(text) is None, (
                f"{relative_path} contains retired executable command "
                f"{pattern.pattern!r}"
            )

        for language, body in _fences(text):
            for pattern in RETIRED_FENCED_PATTERNS:
                assert pattern.search(body) is None, (
                    f"{relative_path} contains retired fenced example "
                    f"{pattern.pattern!r}"
                )
            if language in {"yaml", "yml"}:
                assert RETIRED_YAML_TOP_LEVEL_RE.search(body) is None, (
                    f"{relative_path} contains a retired top-level YAML key"
                )


def test_historical_evidence_has_an_exact_archive_allowlist_and_banner() -> None:
    assert all(not (ROOT / path).exists() for path in ARCHIVE_SOURCES)

    archive_root = ROOT / "experiments/archive/v0_6"
    actual = {
        path.relative_to(ROOT).as_posix() for path in archive_root.rglob("*.md")
    }
    assert actual == set(ARCHIVE_PATHS)

    for relative_path in ARCHIVE_PATHS:
        lines = _read(relative_path).splitlines()
        assert lines and lines[0].startswith("# ")
        first_nonempty_after_title = next(line for line in lines[1:] if line)
        assert first_nonempty_after_title == ARCHIVE_BANNER


def test_all_formal_relative_markdown_links_resolve() -> None:
    broken: list[str] = []
    for relative_path in FORMAL_PATHS:
        document_path = ROOT / relative_path
        for match in MARKDOWN_LINK_RE.finditer(_read(relative_path)):
            target = _relative_link_target(match.group(1))
            if target is None:
                continue
            resolved = (document_path.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{relative_path} -> {target}")
    assert not broken, "broken relative Markdown links:\n" + "\n".join(broken)


def test_real_execution_status_remains_explicitly_not_run() -> None:
    for relative_path in (
        "README.md",
        "docs/PROJECT_OVERVIEW.md",
        "experiments/EXPERIMENT_PLAN.md",
        "docs/V0_7_ACCEPTANCE.md",
    ):
        assert "`not_run`" in _read(relative_path)
