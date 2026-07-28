"""Static guardrails for the W02/W03 single component cutover."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "visual_rl"


def _modules():
    return tuple(sorted(PACKAGE.rglob("*.py")))


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_registry_and_public_plugin_modules_are_physically_absent():
    assert not (PACKAGE / "core" / "registry.py").exists()
    assert not (PACKAGE / "plugins.py").exists()


def test_no_registry_import_or_registration_side_effect_remains():
    violations = []
    for path in _modules():
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "visual_rl.core.registry":
                violations.append((path, node.lineno, "registry import"))
            if isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else None
                )
                if name in {
                    "register",
                    "register_builtin",
                    "register_builtin_plugins",
                }:
                    violations.append((path, node.lineno, name))
    assert violations == []


def test_manifest_tuple_and_lookup_each_have_one_owner():
    function_owners: dict[str, list[Path]] = {
        "builtin_components": [],
        "get_builtin_component": [],
    }
    tuple_owners = []
    for path in _modules():
        tree = _tree(path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in function_owners:
                    function_owners[node.name].append(path)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "_BUILTIN_COMPONENTS":
                    tuple_owners.append(path)
    expected = PACKAGE / "builtins.py"
    assert function_owners == {
        "builtin_components": [expected],
        "get_builtin_component": [expected],
    }
    assert tuple_owners == [expected]


def test_component_description_module_has_no_concrete_factory_imports():
    path = PACKAGE / "core" / "components.py"
    imported_modules = {
        node.module
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert all(
        not module.startswith(
            (
                "visual_rl.model_adapters",
                "visual_rl.rollout",
                "visual_rl.feedback",
                "visual_rl.optimizers",
            )
        )
        for module in imported_modules
    )


def test_no_component_registry_class_or_mutation_surface_exists():
    class_names = {
        node.name
        for path in _modules()
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ClassDef)
    }
    assert "ComponentRegistry" not in class_names
    import visual_rl.core.components as components

    for name in ("add", "register", "freeze", "snapshot", "override"):
        assert not hasattr(components, name)


def test_legacy_adapter_and_rollout_contract_symbols_are_absent():
    banned_definitions = {
        "recompute_log_probs",
        "prepare_for_sampling",
        "prepare_for_training",
        "resolve_context",
        "runtime_config",
        "finalize_batch",
        "sample_single_step",
        "sample_branching",
        "branch_transition_count",
    }
    violations = []
    owned_roots = (
        PACKAGE / "model_adapters",
        PACKAGE / "rollout",
    )
    for root in owned_roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(_tree(path)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in banned_definitions:
                        violations.append((path, node.lineno, node.name))
    assert violations == []
