"""Synthetic archive tests for the stdlib-only wheel contract checker."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

from scripts.verify_wheel_contract import (
    verify_wheel_contract,
    verify_wheel_path_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_REQUIRES = (
    "numpy>=1.26,<2.3",
    "pyyaml>=6.0",
    "tqdm>=4.66",
)
OPTIONAL_REQUIRES = {
    "train": (
        "torch>=2.6",
        "torchvision>=0.21",
        "accelerate>=1.4",
        "diffusers>=0.33",
        "transformers>=4.40",
        "peft>=0.10",
        "ml-collections>=0.1",
        "wandb>=0.18",
        "requests>=2.32",
        "pillow>=10",
        "imageio>=2.34",
    ),
    "dev": ("pytest>=8.2", "ruff>=0.8"),
}


def _record(payloads: dict[str, bytes], record_name: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted((*payloads, record_name)):
        if name == record_name:
            writer.writerow((name, "", ""))
        else:
            payload = payloads[name]
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(payload).digest()
            ).rstrip(b"=").decode("ascii")
            writer.writerow((name, f"sha256={digest}", str(len(payload))))
    return stream.getvalue().encode("utf-8")


def _build_wheel(
    directory: Path,
    *,
    entry_points: bool = False,
    omit_source: bool = False,
    corrupt_record: bool = False,
    omit_dependency: bool = False,
    duplicate_dependency: bool = False,
    wrong_extra_marker: bool = False,
    normalized_metadata_order: bool = False,
    invalid_python_specifier: bool = False,
    invalid_dependency_specifier: bool = False,
) -> Path:
    dist_info = "visual_rl-0.7.0.dist-info"
    package_paths = sorted(
        path
        for path in (ROOT / "visual_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if omit_source:
        package_paths.pop()
    payloads = {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in package_paths
    }
    requirements = list(CORE_REQUIRES)
    requirements.extend(
        f'{dependency}; extra == "{extra}"'
        for extra, dependencies in OPTIONAL_REQUIRES.items()
        for dependency in dependencies
    )
    if omit_dependency:
        requirements.pop()
    if duplicate_dependency:
        requirements.append(requirements[0])
    if wrong_extra_marker:
        requirements[len(CORE_REQUIRES)] = requirements[
            len(CORE_REQUIRES)
        ].replace('extra == "train"', 'extra == "dev"')
    if normalized_metadata_order:
        requirements[0] = "numpy<2.3,>=1.26"
    if invalid_dependency_specifier:
        requirements[0] = "numpy>=1.26,,<2.3"
    requires_python = (
        "<3.12,>=3.10"
        if normalized_metadata_order
        else ">=3.10,<3.12"
    )
    if invalid_python_specifier:
        requires_python = ">=3.10,,<3.12"
    metadata_lines = [
        "Metadata-Version: 2.3",
        "Name: visual-rl",
        "Version: 0.7.0",
        f"Requires-Python: {requires_python}",
        "Provides-Extra: dev",
        "Provides-Extra: train",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        "",
    ]
    payloads[f"{dist_info}/METADATA"] = "\n".join(metadata_lines).encode()
    payloads[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: test\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n\n"
    ).encode()
    if entry_points:
        payloads[f"{dist_info}/entry_points.txt"] = (
            "[console_scripts]\nvisual-rl=visual_rl.cli:main\n"
        ).encode()
    record_name = f"{dist_info}/RECORD"
    record = _record(payloads, record_name)
    if corrupt_record:
        record = record.replace(b"sha256=", b"sha256=broken", 1)
    wheel = directory / "visual_rl-0.7.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr(record_name, record)
    return wheel


def test_valid_synthetic_wheel_matches_source_and_metadata(tmp_path: Path) -> None:
    _build_wheel(tmp_path)
    assert verify_wheel_contract(ROOT, tmp_path) == ()


def test_checker_accepts_equivalent_normalized_specifier_order(
    tmp_path: Path,
) -> None:
    _build_wheel(tmp_path, normalized_metadata_order=True)
    assert verify_wheel_contract(ROOT, tmp_path) == ()


def test_checker_rejects_empty_or_malformed_specifier_items(
    tmp_path: Path,
) -> None:
    wheel = _build_wheel(tmp_path, invalid_python_specifier=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("cannot validate METADATA dependency contract" in error for error in errors)
    wheel.unlink()

    _build_wheel(tmp_path, invalid_dependency_specifier=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("cannot validate METADATA dependency contract" in error for error in errors)


def test_checker_requires_exactly_one_wheel(tmp_path: Path) -> None:
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert errors and "exactly one wheel" in errors[0]
    _build_wheel(tmp_path)
    (tmp_path / "second.whl").write_bytes(b"not-a-wheel")
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert errors and "exactly one wheel" in errors[0]


def test_explicit_wheel_path_is_required_and_unambiguous(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    assert verify_wheel_path_contract(ROOT, wheel) == ()

    missing = tmp_path / "missing.whl"
    errors = verify_wheel_path_contract(ROOT, missing)
    assert errors and "cannot resolve wheel" in errors[0]

    (tmp_path / "second.whl").write_bytes(b"not-a-wheel")
    errors = verify_wheel_path_contract(ROOT, wheel)
    assert errors and "expected exactly one wheel" in errors[0]


def test_checker_rejects_entry_points_and_forbidden_payload(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, entry_points=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("entry points" in error for error in errors)
    wheel.unlink()

    _build_wheel(tmp_path)
    original = tmp_path / "visual_rl-0.7.0-py3-none-any.whl"
    with zipfile.ZipFile(original, "a") as archive:
        archive.writestr("tests/not_allowed.py", b"pass\n")
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("forbidden wheel payload" in error for error in errors)


def test_checker_rejects_singular_model_and_run_payloads(tmp_path: Path) -> None:
    for forbidden in (
        "visual_rl/model/checkpoint.bin",
        "visual_rl/run/result.json",
    ):
        _build_wheel(tmp_path)
        wheel = tmp_path / "visual_rl-0.7.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(forbidden, b"forbidden")
        errors = verify_wheel_contract(ROOT, tmp_path)
        assert any(forbidden in error for error in errors)
        wheel.unlink()


def test_checker_rejects_source_set_and_record_drift(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, omit_source=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("file set mismatch" in error for error in errors)
    wheel.unlink()
    _build_wheel(tmp_path, corrupt_record=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("RECORD digest mismatch" in error for error in errors)


def test_checker_rejects_missing_or_duplicate_requires_dist(
    tmp_path: Path,
) -> None:
    wheel = _build_wheel(tmp_path, omit_dependency=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("Requires-Dist mismatch" in error for error in errors)
    wheel.unlink()
    _build_wheel(tmp_path, duplicate_dependency=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("Requires-Dist mismatch" in error for error in errors)
    wheel = tmp_path / "visual_rl-0.7.0-py3-none-any.whl"
    wheel.unlink()
    _build_wheel(tmp_path, wrong_extra_marker=True)
    errors = verify_wheel_contract(ROOT, tmp_path)
    assert any("Requires-Dist mismatch" in error for error in errors)
