"""Standard-library-only VisualRL v0.7 wheel contract checker."""

from __future__ import annotations

import ast
import base64
from collections import Counter
import csv
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

EXPECTED_NAME = "visual-rl"
EXPECTED_VERSION = "0.7.0"
FORBIDDEN_PAYLOAD_SEGMENTS = {
    "experiments",
    "model",
    "models",
    "reference",
    "reference_code",
    "run",
    "runs",
    "scripts",
    "tests",
}
SERVICE_PREFIX = "services/world_r1_strict/"


def verify_wheel_contract(
    repo_root: str | Path,
    dist_dir: str | Path | None = None,
) -> tuple[str, ...]:
    """Return every deterministic archive/metadata/source contract error."""

    root = Path(repo_root).resolve(strict=True)
    distribution = (
        Path(dist_dir).resolve(strict=True)
        if dist_dir is not None
        else root / "dist"
    )
    wheels = sorted(distribution.glob("*.whl"))
    if len(wheels) != 1:
        return (f"expected exactly one wheel in {distribution}, found {len(wheels)}",)
    wheel = wheels[0]
    errors: list[str] = []
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                errors.append("wheel contains duplicate archive member names")
            for name in names:
                if not _safe_member(name):
                    errors.append(f"unsafe wheel member path: {name}")

            dist_info = _dist_info_dir(names)
            if dist_info is None:
                errors.append("wheel must contain one visual_rl-0.7.0.dist-info tree")
                return tuple(errors)
            metadata_name = f"{dist_info}/METADATA"
            record_name = f"{dist_info}/RECORD"
            if metadata_name not in names:
                errors.append("wheel is missing METADATA")
            else:
                try:
                    errors.extend(
                        _metadata_errors(
                            archive.read(metadata_name),
                            root / "pyproject.toml",
                        )
                    )
                except (OSError, ValueError) as exc:
                    errors.append(
                        "cannot validate METADATA dependency contract: "
                        f"{exc}"
                    )
            if f"{dist_info}/entry_points.txt" in names:
                errors.append("wheel must not contain console or plugin entry points")

            package_names = {
                name
                for name in names
                if name.startswith("visual_rl/") or name.startswith(SERVICE_PREFIX)
            }
            expected_package_names = {
                path.relative_to(root).as_posix()
                for path in (root / "visual_rl").rglob("*.py")
                if path.is_file()
                and "__pycache__" not in path.parts
            }
            expected_package_names.update(
                path.relative_to(root).as_posix()
                for path in (root / "services/world_r1_strict").rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
            if package_names != expected_package_names:
                missing = sorted(expected_package_names - package_names)
                extra = sorted(package_names - expected_package_names)
                errors.append(
                    f"wheel/source runtime file set mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            for name in names:
                path = PurePosixPath(name)
                if name.startswith(SERVICE_PREFIX):
                    continue
                if path.parts and path.parts[0] == "services" and not name.startswith(
                    SERVICE_PREFIX
                ):
                    errors.append(f"forbidden wheel payload: {name}")
                if any(part in FORBIDDEN_PAYLOAD_SEGMENTS for part in path.parts):
                    errors.append(f"forbidden wheel payload: {name}")
            if record_name not in names:
                errors.append("wheel is missing RECORD")
            else:
                errors.extend(
                    _record_errors(
                        archive,
                        names=names,
                        record_name=record_name,
                    )
                )
    except (OSError, zipfile.BadZipFile) as exc:
        return (f"cannot read wheel {wheel}: {exc}",)
    return tuple(errors)


def verify_wheel_path_contract(
    repo_root: str | Path,
    wheel_path: str | Path,
) -> tuple[str, ...]:
    """Validate the explicitly selected wheel and reject ambiguous siblings."""

    try:
        wheel = Path(wheel_path).resolve(strict=True)
    except OSError as exc:
        return (f"cannot resolve wheel {wheel_path}: {exc}",)
    if not wheel.is_file() or wheel.suffix != ".whl":
        return (f"wheel path is not a regular .whl file: {wheel}",)
    siblings = sorted(wheel.parent.glob("*.whl"))
    if len(siblings) != 1:
        return (
            f"expected exactly one wheel in {wheel.parent}, "
            f"found {len(siblings)}",
        )
    if siblings[0].resolve() != wheel:
        return (f"selected wheel is not the unique wheel in {wheel.parent}",)
    return verify_wheel_contract(repo_root, wheel.parent)


def _metadata_errors(metadata_bytes: bytes, pyproject_path: Path) -> tuple[str, ...]:
    project, optional_dependencies, core_dependencies = _project_contract(
        pyproject_path
    )
    expected_extras = set(optional_dependencies)
    metadata = BytesParser(policy=compat32).parsebytes(metadata_bytes)
    errors: list[str] = []
    expected = {
        "Name": project["name"],
        "Version": project["version"],
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            errors.append(
                f"METADATA {field} is {metadata.get(field)!r}, expected {value!r}"
            )
    actual_python = metadata.get("Requires-Python")
    if (
        not isinstance(actual_python, str)
        or _normalize_specifier_list(actual_python)
        != _normalize_specifier_list(project["requires-python"])
    ):
        errors.append(
            "METADATA Requires-Python mismatch: "
            f"got={actual_python!r}, expected={project['requires-python']!r}"
        )
    extras = set(metadata.get_all("Provides-Extra", []))
    if extras != expected_extras:
        errors.append(
            f"METADATA extras mismatch: got {sorted(extras)}, "
            f"expected {sorted(expected_extras)}"
        )
    expected_requires = list(core_dependencies)
    for extra, dependencies in optional_dependencies.items():
        expected_requires.extend(
            f'{dependency}; extra == "{extra}"'
            for dependency in dependencies
        )
    actual_requires = metadata.get_all("Requires-Dist", [])
    normalized_expected = Counter(
        _normalize_requirement(value) for value in expected_requires
    )
    normalized_actual = Counter(
        _normalize_requirement(value) for value in actual_requires
    )
    if normalized_actual != normalized_expected:
        errors.append(
            "METADATA Requires-Dist mismatch: "
            f"got={sorted(normalized_actual.elements())}, "
            f"expected={sorted(normalized_expected.elements())}"
        )
    return tuple(errors)


def _project_contract(
    path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Parse the bounded project scalar/dependency subset needed by the checker.

    Python 3.10 has no stdlib TOML module. The wheel checker intentionally
    implements this fail-closed string-array subset instead of adding a
    dependency.
    """

    section = ""
    project: dict[str, str] = {}
    core_dependencies: tuple[str, ...] | None = None
    optional: dict[str, tuple[str, ...]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, raw_value = (item.strip() for item in line.split("=", 1))
        if section == "project" and key in {"name", "version", "requires-python"}:
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"invalid pyproject scalar {key}") from exc
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid pyproject scalar {key}")
            project[key] = value
        elif section == "project" and key == "dependencies":
            if core_dependencies is not None:
                raise ValueError("duplicate project dependencies")
            core_dependencies, index = _string_array(
                lines,
                index,
                raw_value,
                label="project dependencies",
            )
        elif section == "project.optional-dependencies":
            if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or not raw_value.startswith(
                "["
            ):
                continue
            if key in optional:
                raise ValueError("invalid or duplicate optional dependency group")
            optional[key], index = _string_array(
                lines,
                index,
                raw_value,
                label=f"optional dependency group {key}",
            )
    if set(project) != {"name", "requires-python", "version"}:
        raise ValueError("pyproject is missing required project metadata")
    if core_dependencies is None or not core_dependencies or not optional:
        raise ValueError("pyproject dependency declarations are incomplete")
    if project["name"] != EXPECTED_NAME or project["version"] != EXPECTED_VERSION:
        raise ValueError("pyproject does not describe visual-rl 0.7.0")
    return project, optional, core_dependencies


def _string_array(
    lines: list[str],
    index: int,
    initial: str,
    *,
    label: str,
) -> tuple[tuple[str, ...], int]:
    text = initial
    while text.count("[") > text.count("]"):
        if index >= len(lines):
            raise ValueError(f"unterminated {label}")
        text += "\n" + lines[index]
        index += 1
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must be a non-empty unique string array")
    return tuple(value), index


def _normalize_requirement(value: str) -> str:
    text = " ".join(value.split())
    base, separator, marker = text.partition(";")
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)(.*)", base.strip())
    if match is None:
        raise ValueError(f"invalid Requires-Dist value: {value!r}")
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    constraints = re.sub(r"\s+", "", match.group(2))
    if constraints.startswith("(") and constraints.endswith(")"):
        constraints = constraints[1:-1]
    normalized = name + _normalize_specifier_list(constraints)
    if not separator:
        return normalized
    marker_match = re.fullmatch(
        r"""\s*extra\s*==\s*(['"])([A-Za-z0-9_-]+)\1\s*""",
        marker,
    )
    if marker_match is None:
        raise ValueError(f"unsupported Requires-Dist marker: {value!r}")
    return f"{normalized};extra=={marker_match.group(2)}"


def _normalize_specifier_list(value: str) -> str:
    if value == "":
        return ""
    raw_parts = value.split(",")
    if any(not part.strip() for part in raw_parts):
        raise ValueError(f"invalid empty version specifier in {value!r}")
    parts: list[str] = []
    for raw_part in raw_parts:
        part = raw_part.strip()
        if re.fullmatch(r"(?:===|==|~=|!=|<=|>=|<|>)[^,;\s]+", part) is None:
            raise ValueError(f"invalid version specifier: {part!r}")
        parts.append(part)
    parts.sort()
    return ",".join(parts)


def _record_errors(
    archive: zipfile.ZipFile,
    *,
    names: list[str],
    record_name: str,
) -> tuple[str, ...]:
    try:
        text = archive.read(record_name).decode("utf-8")
    except UnicodeDecodeError:
        return ("RECORD must be UTF-8 CSV",)
    rows = list(csv.reader(io.StringIO(text, newline="")))
    errors: list[str] = []
    if any(len(row) != 3 for row in rows):
        return ("every RECORD row must have exactly three columns",)
    record_names = [row[0] for row in rows]
    if len(record_names) != len(set(record_names)):
        errors.append("RECORD contains duplicate paths")
    if set(record_names) != set(names):
        errors.append("RECORD paths are not the exact archive regular-file set")
    by_name = {row[0]: row for row in rows}
    for name in names:
        row = by_name.get(name)
        if row is None:
            continue
        digest, size = row[1], row[2]
        if name == record_name:
            if digest or size:
                errors.append("RECORD must leave its own hash and size empty")
            continue
        payload = archive.read(name)
        expected_digest = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(payload).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != expected_digest:
            errors.append(f"RECORD digest mismatch: {name}")
        if size != str(len(payload)):
            errors.append(f"RECORD size mismatch: {name}")
    return tuple(errors)


def _dist_info_dir(names: list[str]) -> str | None:
    candidates = {
        name.split("/", 1)[0]
        for name in names
        if "/" in name and name.split("/", 1)[0].endswith(".dist-info")
    }
    expected = f"visual_rl-{EXPECTED_VERSION}.dist-info"
    return expected if candidates == {expected} else None


def _safe_member(name: str) -> bool:
    if "\\" in name:
        return False
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and path.as_posix() == name
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python scripts/verify_wheel_contract.py WHEEL_PATH"
        )
    repository = Path(__file__).resolve().parents[1]
    failures = verify_wheel_path_contract(repository, sys.argv[1])
    if failures:
        raise SystemExit("\n".join(failures))
