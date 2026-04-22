"""
Code Graph Builder
==================

Build the structural candidate space G_CG required by paper section 3.1.1.

The paper constrains G_CG to:
- code syntax / semantics entities
- containment relations
- reference relations

This builder therefore normalizes all discovered non-containment dependencies
into the paper-level `references` family while preserving a more specific
`reference_kind` in metadata for debugging and downstream analysis.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from .causal_relevance_graph import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
    EntityType,
    RelationType,
)

CODE_GRAPH_SCHEMA_VERSION = 2


class ASTAnalyzer(ast.NodeVisitor):
    """
    AST visitor that extracts code entities and structural relations.

    Newly emitted relations are normalized to:
    - RelationType.CONTAINS
    - RelationType.REFERENCES
    """

    def __init__(self, file_path: str, module_name: Optional[str] = None):
        self.file_path = file_path
        self.module_name = module_name
        self.entities: Dict[str, CodeEntity] = {}
        self.relations: List[CodeRelation] = []
        self.scope_stack: List[Tuple[str, str, EntityType]] = []
        self.name_to_ids: Dict[str, Set[str]] = {}
        self.import_aliases: Dict[str, str] = {}
        self.ignored_reference_names = set(dir(builtins)) | {
            "self",
            "cls",
            "super",
            "__name__",
            "__file__",
        }

    def _register_name(self, name: str, entity_id: str) -> None:
        self.name_to_ids.setdefault(name, set()).add(entity_id)

    def _register_import_alias(self, local_name: str, qualified_name: str) -> None:
        if not local_name or not qualified_name:
            return
        self.import_aliases[local_name] = qualified_name

    def _make_entity_id(self, name: str, entity_type: EntityType) -> str:
        scope = "::".join(scope_name for scope_name, _, _ in self.scope_stack)
        if scope:
            return f"{self.file_path}::{scope}::{name}:{entity_type.value}"
        return f"{self.file_path}::{name}:{entity_type.value}"

    def _current_parent_id(self) -> Optional[str]:
        return self.scope_stack[-1][1] if self.scope_stack else None

    def _current_class_name(self) -> Optional[str]:
        for scope_name, _, scope_type in reversed(self.scope_stack):
            if scope_type == EntityType.CLASS:
                return scope_name
        return None

    def _current_function_name(self) -> Optional[str]:
        for scope_name, _, scope_type in reversed(self.scope_stack):
            if scope_type == EntityType.FUNCTION:
                return scope_name
        return None

    def _add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        **metadata,
    ) -> None:
        self.relations.append(
            CodeRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                metadata=metadata,
            )
        )

    def _add_reference(self, source_id: str, target_id: str, reference_kind: str, **metadata) -> None:
        payload = {"reference_kind": reference_kind}
        payload.update(metadata)
        self._add_relation(source_id, target_id, RelationType.REFERENCES, **payload)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        entity_id = self._make_entity_id(node.name, EntityType.CLASS)
        entity = CodeEntity(
            id=entity_id,
            name=node.name,
            entity_type=EntityType.CLASS,
            file_path=self.file_path,
            class_name=node.name,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            parent_id=self._current_parent_id(),
        )
        self.entities[entity_id] = entity
        self._register_name(node.name, entity_id)

        parent_id = self._current_parent_id() or f"file:{self.file_path}"
        self._add_relation(parent_id, entity_id, RelationType.CONTAINS)

        for base in node.bases:
            base_name = self._resolve_symbol_name(base)
            if base_name:
                self._add_reference(entity_id, base_name, "inherits", line=node.lineno)

        self.scope_stack.append((node.name, entity_id, EntityType.CLASS))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        entity_id = self._make_entity_id(node.name, EntityType.FUNCTION)
        entity = CodeEntity(
            id=entity_id,
            name=node.name,
            entity_type=EntityType.FUNCTION,
            file_path=self.file_path,
            class_name=self._current_class_name(),
            function_name=node.name,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            parent_id=self._current_parent_id(),
        )
        self.entities[entity_id] = entity
        self._register_name(node.name, entity_id)

        parent_id = self._current_parent_id() or f"file:{self.file_path}"
        self._add_relation(parent_id, entity_id, RelationType.CONTAINS)

        for arg in node.args.args:
            param_id = f"{entity_id}::param:{arg.arg}"
            param_entity = CodeEntity(
                id=param_id,
                name=arg.arg,
                entity_type=EntityType.PARAMETER,
                file_path=self.file_path,
                class_name=self._current_class_name(),
                function_name=node.name,
                variable_name=arg.arg,
                line_start=node.lineno,
                line_end=node.lineno,
                parent_id=entity_id,
            )
            self.entities[param_id] = param_entity
            self._register_name(arg.arg, param_id)
            self._add_relation(entity_id, param_id, RelationType.CONTAINS)

        self.scope_stack.append((node.name, entity_id, EntityType.FUNCTION))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        source_id = self._current_parent_id() or f"file:{self.file_path}"
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            var_name = target.id
            var_id = self._make_entity_id(var_name, EntityType.VARIABLE)
            if var_id not in self.entities:
                self.entities[var_id] = CodeEntity(
                    id=var_id,
                    name=var_name,
                    entity_type=EntityType.VARIABLE,
                    file_path=self.file_path,
                    class_name=self._current_class_name(),
                    function_name=self._current_function_name(),
                    variable_name=var_name,
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    parent_id=self._current_parent_id(),
                )
                self._register_name(var_name, var_id)
                if self._current_parent_id():
                    self._add_relation(self._current_parent_id(), var_id, RelationType.CONTAINS)

            self._add_reference(source_id, var_id, "write", line=node.lineno)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        source_id = self._current_parent_id()
        if not source_id:
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword)
            return

        callee_name = self._resolve_symbol_name(node.func)
        if callee_name and callee_name not in self.ignored_reference_names:
            self._add_reference(source_id, callee_name, "call", line=node.lineno)

        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return

        source_id = self._current_parent_id()
        if not source_id or node.id in self.ignored_reference_names:
            self.generic_visit(node)
            return

        self._add_reference(source_id, node.id, "read", line=node.lineno)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        source_id = self._current_parent_id() or f"file:{self.file_path}"
        for alias in node.names:
            target_name = alias.name
            local_name = alias.asname or alias.name.split(".")[0]
            import_id = self._make_entity_id(target_name, EntityType.IMPORT)
            if import_id not in self.entities:
                self.entities[import_id] = CodeEntity(
                    id=import_id,
                    name=target_name,
                    entity_type=EntityType.IMPORT,
                    file_path=self.file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                    parent_id=self._current_parent_id(),
                )
                self._register_name(local_name, import_id)
            self._register_import_alias(local_name, target_name)
            self._add_reference(source_id, import_id, "import", line=node.lineno)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source_id = self._current_parent_id() or f"file:{self.file_path}"
        module_name = self._resolve_import_from_module(node.module or "", node.level)
        for alias in node.names:
            target_name = f"{module_name}.{alias.name}" if module_name else alias.name
            local_name = alias.asname or alias.name
            import_id = self._make_entity_id(target_name, EntityType.IMPORT)
            if import_id not in self.entities:
                self.entities[import_id] = CodeEntity(
                    id=import_id,
                    name=target_name,
                    entity_type=EntityType.IMPORT,
                    file_path=self.file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                    parent_id=self._current_parent_id(),
                )
                self._register_name(local_name, import_id)
            self._register_import_alias(local_name, target_name)
            self._add_reference(source_id, import_id, "import_from", line=node.lineno)

        self.generic_visit(node)

    def _resolve_symbol_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return self.import_aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            root = self._resolve_symbol_name(node.value)
            return f"{root}.{node.attr}" if root else node.attr
        return None

    def _resolve_import_from_module(self, module_name: str, level: int) -> str:
        if level <= 0 or not self.module_name:
            return module_name

        module_parts = self.module_name.split(".")
        if level > len(module_parts):
            base_parts: List[str] = []
        else:
            base_parts = module_parts[:-level]

        if module_name:
            base_parts.extend(part for part in module_name.split(".") if part)

        return ".".join(base_parts)


class CodeGraphBuilder:
    """Orchestrates AST extraction and reference resolution for G_CG."""

    def __init__(self):
        self.code_graph = CodeGraph()
        self.repo_root: Optional[Path] = None
        self._resolution_name_index: Dict[str, List[CodeEntity]] = {}
        logger.info("Initialized CodeGraphBuilder")

    def build_from_repository(self, repo_path: str) -> CodeGraph:
        repo_root = Path(repo_path).resolve()
        self.repo_root = repo_root
        py_files = list(repo_root.rglob("*.py"))
        logger.info(f"Found {len(py_files)} Python files")

        for py_file in py_files:
            try:
                self._analyze_file(py_file)
            except Exception as exc:
                logger.warning(f"Error analyzing {py_file}: {exc}")

        self._build_resolution_indexes()
        self._resolve_references()
        logger.info(
            f"Built CodeGraph: {len(self.code_graph.entities)} entities, "
            f"{len(self.code_graph.relations)} relations"
        )
        return self.code_graph

    def build_from_file(self, file_path: str) -> CodeGraph:
        resolved_file = Path(file_path).resolve()
        if self.repo_root is None:
            self.repo_root = resolved_file.parent
        self._analyze_file(resolved_file)
        self._build_resolution_indexes()
        self._resolve_references()
        return self.code_graph

    def _analyze_file(self, file_path: Path) -> None:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning(f"Syntax error in {file_path}: {exc}")
            return
        except Exception as exc:
            logger.warning(f"Error reading {file_path}: {exc}")
            return

        file_entity_id = f"file:{file_path}"
        self.code_graph.add_entity(
            CodeEntity(
                id=file_entity_id,
                name=file_path.name,
                entity_type=EntityType.FILE,
                file_path=str(file_path),
            )
        )

        analyzer = ASTAnalyzer(
            str(file_path),
            module_name=self._module_name_for_path(file_path),
        )
        analyzer.visit(tree)

        for entity in analyzer.entities.values():
            self.code_graph.add_entity(entity)

        for relation in analyzer.relations:
            self.code_graph.add_relation(relation)

        logger.debug(f"Analyzed {file_path}: {len(analyzer.entities)} entities")

    def _module_name_for_path(self, file_path: Path) -> Optional[str]:
        if not self.repo_root:
            return None

        try:
            relative = file_path.resolve().relative_to(self.repo_root)
        except ValueError:
            return None

        module_parts = list(relative.with_suffix("").parts)
        if module_parts and module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        return ".".join(module_parts) if module_parts else None

    def _build_resolution_indexes(self) -> None:
        index: Dict[str, List[CodeEntity]] = {}

        for entity in self.code_graph.entities.values():
            names = {
                entity.name,
                entity.function_name,
                entity.class_name,
                entity.variable_name,
            }
            for name in names:
                if not name:
                    continue
                index.setdefault(name, []).append(entity)

        self._resolution_name_index = index

    def _resolve_references(self) -> None:
        """
        Resolve unresolved symbolic references to concrete entity ids.

        Unresolved targets are preserved during analysis, then resolved here by
        name matching with a local-file preference to reduce noise.
        """

        resolved_relations: List[CodeRelation] = []

        for relation in self.code_graph.relations:
            if relation.source_id not in self.code_graph.entities:
                continue

            if relation.target_id in self.code_graph.entities:
                resolved_relations.append(relation)
                continue

            candidates = self._find_matching_entities(relation)
            if not candidates:
                continue

            for entity in candidates:
                resolved_relations.append(
                    CodeRelation(
                        source_id=relation.source_id,
                        target_id=entity.id,
                        relation_type=relation.relation_type,
                        metadata=dict(relation.metadata),
                    )
                )

        self.code_graph.relations = resolved_relations
        self.code_graph.rebuild_graph()

    def _find_matching_entities(self, relation: CodeRelation) -> List[CodeEntity]:
        reference_kind = relation.metadata.get("reference_kind")
        source_entity = self.code_graph.get_entity(relation.source_id)
        if not source_entity:
            return []

        qualified_target = relation.target_id
        raw_target = relation.target_id.split(".")[-1]
        raw_target = raw_target.split("::")[-1]
        raw_target = raw_target.split(":")[0]
        if not raw_target:
            return []

        matches = [
            entity
            for entity in self._resolution_name_index.get(raw_target, [])
            if entity.name == raw_target
            or entity.function_name == raw_target
            or entity.class_name == raw_target
            or entity.variable_name == raw_target
        ]

        if not matches:
            return []

        if reference_kind == "call":
            callable_types = {EntityType.FUNCTION, EntityType.CLASS}
            matches = [entity for entity in matches if entity.entity_type in callable_types]
        elif reference_kind == "inherits":
            matches = [entity for entity in matches if entity.entity_type == EntityType.CLASS]

        if not matches:
            return []

        module_scoped = self._prefer_module_scoped_matches(qualified_target, matches)
        if module_scoped:
            matches = module_scoped

        same_file = [entity for entity in matches if entity.file_path == source_entity.file_path]
        if same_file and reference_kind != "call":
            matches = same_file

        scores = {entity.id: self._entity_match_score(entity, source_entity, reference_kind, qualified_target) for entity in matches}
        if not scores:
            return []

        best_score = max(scores.values())
        best_matches = [entity for entity in matches if scores[entity.id] == best_score]
        best_matches.sort(key=lambda entity: entity.id)
        return best_matches

    def _prefer_module_scoped_matches(
        self,
        qualified_target: str,
        matches: List[CodeEntity],
    ) -> List[CodeEntity]:
        if "." not in qualified_target:
            return []

        module_name = qualified_target.rsplit(".", 1)[0]
        file_suffixes = [
            module_name.replace(".", "/") + ".py",
            module_name.replace(".", "/") + "/__init__.py",
        ]

        scoped = [
            entity
            for entity in matches
            if any(entity.file_path.endswith(suffix) for suffix in file_suffixes)
        ]
        return scoped

    def _entity_match_score(
        self,
        entity: CodeEntity,
        source_entity: CodeEntity,
        reference_kind: Optional[str],
        qualified_target: str,
    ) -> int:
        score = 0

        if entity.file_path == source_entity.file_path:
            score += 4

        if "." in qualified_target:
            module_name = qualified_target.rsplit(".", 1)[0]
            file_suffixes = {
                module_name.replace(".", "/") + ".py",
                module_name.replace(".", "/") + "/__init__.py",
            }
            if any(entity.file_path.endswith(suffix) for suffix in file_suffixes):
                score += 8

        if reference_kind == "call":
            if entity.entity_type == EntityType.FUNCTION:
                score += 6
            elif entity.entity_type == EntityType.CLASS:
                score += 4
            else:
                score -= 8
        elif reference_kind == "inherits":
            if entity.entity_type == EntityType.CLASS:
                score += 6
            else:
                score -= 8
        elif reference_kind in {"read", "write"}:
            if entity.entity_type in {EntityType.PARAMETER, EntityType.VARIABLE}:
                score += 3
            elif entity.entity_type == EntityType.IMPORT:
                score += 2
            elif entity.entity_type in {EntityType.FUNCTION, EntityType.CLASS}:
                score += 1

        return score


if __name__ == "__main__":
    logger.info("CodeGraphBuilder module ready.")
