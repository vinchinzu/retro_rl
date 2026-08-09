"""Bounded, namespaced progression primitives for adventure graphs.

This module deliberately models only monotonic item logic.  Capabilities are
qualified by game namespace, requirements only test possession, and checks
add an item at most once.  Resource counts, consumables, keys, and generator
formats belong above this small offline model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Hashable, TypeAlias


def _normalize_part(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if ":" in normalized:
        raise ValueError(f"{label} must not contain ':'")
    return normalized


def _normalize_legacy_capability(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True, order=True, slots=True)
class CapabilityId:
    """An immutable, qualified capability such as ``sm:bombs``."""

    namespace: str
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _normalize_part(self.namespace, label="capability namespace"),
        )
        object.__setattr__(
            self,
            "name",
            _normalize_part(self.name, label="capability name"),
        )

    @property
    def qualified(self) -> str:
        return f"{self.namespace}:{self.name}"

    def __str__(self) -> str:
        return self.qualified

    @classmethod
    def parse(cls, value: str) -> CapabilityId:
        """Parse a qualified ``namespace:name`` spelling."""
        if not isinstance(value, str):
            raise TypeError("qualified capability must be a string")
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"qualified capability must use namespace:name, got {value!r}"
            )
        return cls(parts[0], parts[1])

    from_string = parse

    @classmethod
    def coerce(
        cls,
        value: CapabilityId | str,
        *,
        namespace: str | None = None,
    ) -> CapabilityId:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("capability must be a CapabilityId or string")
        if ":" in value:
            return cls.parse(value)
        if namespace is None:
            raise ValueError(
                f"unqualified capability {value!r} needs a namespace"
            )
        return cls(namespace, value)

    def to_dict(self) -> dict[str, str]:
        return {"namespace": self.namespace, "name": self.name}

    def to_json(self) -> str:
        return json.dumps(self.qualified, separators=(",", ":"))

    to_string = __str__


CapabilityValue: TypeAlias = CapabilityId | str


class Requirement:
    """Monotonic predicate over a set of capabilities."""

    def satisfied_by(self, capabilities: Iterable[CapabilityValue]) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    to_canonical_json = canonical_json

    def to_json(self) -> str:
        return self.canonical_json()

    serialize = to_json

    def __call__(self, capabilities: Iterable[CapabilityValue]) -> bool:
        return self.satisfied_by(capabilities)

    is_satisfied_by = __call__


def _qualified_capabilities(
    capabilities: Iterable[CapabilityValue],
) -> frozenset[CapabilityId]:
    qualified: set[CapabilityId] = set()
    for value in capabilities:
        if isinstance(value, CapabilityId):
            qualified.add(value)
        elif isinstance(value, str) and ":" in value:
            qualified.add(CapabilityId.parse(value))
    return frozenset(qualified)


def _coerce_requirement(value: Requirement | CapabilityId | str) -> Requirement:
    if isinstance(value, Requirement):
        return value
    if isinstance(value, CapabilityId):
        return Has(value)
    if isinstance(value, str) and ":" in value:
        return Has(CapabilityId.parse(value))
    raise TypeError(
        "requirements need Has/AllOf/AnyOf or qualified CapabilityId values"
    )


@dataclass(frozen=True, slots=True)
class Has(Requirement):
    """Requirement satisfied by possession of one capability."""

    capability: CapabilityId

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability",
            CapabilityId.coerce(self.capability),
        )

    def satisfied_by(self, capabilities: Iterable[CapabilityValue]) -> bool:
        return self.capability in _qualified_capabilities(capabilities)

    def to_dict(self) -> dict[str, str]:
        return {"has": str(self.capability)}


def _argument_values(
    values: tuple[object, ...],
    *,
    requirements: Iterable[Requirement | CapabilityId | str] | None,
    items: Iterable[Requirement | CapabilityId | str] | None,
) -> tuple[object, ...]:
    if requirements is not None and items is not None:
        raise TypeError("pass only one of requirements= or items=")
    if requirements is not None:
        if values:
            raise TypeError("requirements= cannot be combined with positional values")
        return tuple(requirements)
    if items is not None:
        if values:
            raise TypeError("items= cannot be combined with positional values")
        return tuple(items)
    if (
        len(values) == 1
        and isinstance(values[0], Iterable)
        and not isinstance(values[0], (str, bytes, CapabilityId, Requirement))
    ):
        return tuple(values[0])  # type: ignore[arg-type]
    return values


def _canonical_children(
    values: tuple[object, ...],
    container_type: type[AllOf] | type[AnyOf],
) -> tuple[Requirement, ...]:
    children: list[Requirement] = []
    for value in values:
        requirement = _coerce_requirement(value)  # type: ignore[arg-type]
        if isinstance(requirement, container_type):
            children.extend(requirement.requirements)
        else:
            children.append(requirement)
    unique = {child.canonical_json(): child for child in children}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True, init=False)
class AllOf(Requirement):
    """Requirement satisfied when every child requirement is satisfied."""

    requirements: tuple[Requirement, ...]

    def __init__(
        self,
        *values: Requirement | CapabilityId | str | Iterable[Requirement | CapabilityId | str],
        requirements: Iterable[Requirement | CapabilityId | str] | None = None,
        items: Iterable[Requirement | CapabilityId | str] | None = None,
    ) -> None:
        raw = _argument_values(values, requirements=requirements, items=items)
        object.__setattr__(self, "requirements", _canonical_children(raw, AllOf))

    @property
    def items(self) -> tuple[Requirement, ...]:
        return self.requirements

    def satisfied_by(self, capabilities: Iterable[CapabilityValue]) -> bool:
        values = tuple(capabilities)
        return all(requirement.satisfied_by(values) for requirement in self.requirements)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"allOf": [requirement.to_dict() for requirement in self.requirements]}


@dataclass(frozen=True, slots=True, init=False)
class AnyOf(Requirement):
    """Requirement satisfied when at least one child requirement is satisfied."""

    requirements: tuple[Requirement, ...]

    def __init__(
        self,
        *values: Requirement | CapabilityId | str | Iterable[Requirement | CapabilityId | str],
        requirements: Iterable[Requirement | CapabilityId | str] | None = None,
        items: Iterable[Requirement | CapabilityId | str] | None = None,
        options: Iterable[Requirement | CapabilityId | str] | None = None,
    ) -> None:
        if options is not None:
            if requirements is not None or items is not None:
                raise TypeError("pass only one of options=, requirements=, or items=")
            requirements = options
        raw = _argument_values(values, requirements=requirements, items=items)
        object.__setattr__(self, "requirements", _canonical_children(raw, AnyOf))

    @property
    def items(self) -> tuple[Requirement, ...]:
        return self.requirements

    def satisfied_by(self, capabilities: Iterable[CapabilityValue]) -> bool:
        values = tuple(capabilities)
        return any(requirement.satisfied_by(values) for requirement in self.requirements)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"anyOf": [requirement.to_dict() for requirement in self.requirements]}


TRUE_REQUIREMENT = AllOf()
FALSE_REQUIREMENT = AnyOf()


def _placement_assignments(
    values: Mapping[str, CapabilityValue]
    | Iterable[tuple[str, CapabilityValue]],
) -> tuple[tuple[str, CapabilityId], ...]:
    if isinstance(values, Mapping):
        raw = tuple(values.items())
    else:
        raw = tuple(values)
    assignments: dict[str, CapabilityId] = {}
    for check_id, capability in raw:
        if not isinstance(check_id, str) or not check_id:
            raise ValueError("placement check IDs must be non-empty strings")
        qualified = CapabilityId.coerce(capability)
        previous = assignments.get(check_id)
        if previous is not None and previous != qualified:
            raise ValueError(f"conflicting placement for check {check_id!r}")
        assignments[check_id] = qualified
    return tuple(sorted(assignments.items()))


@dataclass(frozen=True, slots=True, init=False)
class SeedPlacement:
    """Immutable check-to-item assignments for one seed overlay.

    The normal form is an immutable collection of assignments.  The compact
    ``SeedPlacement(check_id, capability)`` spelling is also accepted for a
    single assignment, which keeps fixture declarations readable.
    """

    assignments: tuple[tuple[str, CapabilityId], ...]
    seed_id: str

    def __init__(
        self,
        check_id: str
        | Mapping[str, CapabilityValue]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        capability: CapabilityValue | None = None,
        *,
        placements: Mapping[str, CapabilityValue]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        items: Mapping[str, CapabilityValue]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        assignments: Mapping[str, CapabilityValue]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        item: CapabilityValue | None = None,
        seed_id: str = "",
        seed: str | None = None,
    ) -> None:
        if seed is not None:
            if seed_id and seed_id != seed:
                raise TypeError("seed_id and seed disagree")
            seed_id = seed
        supplied_collections = [
            value for value in (placements, items, assignments) if value is not None
        ]
        if (
            supplied_collections
            and isinstance(check_id, str)
            and capability is None
            and item is None
        ):
            seed_id = seed_id or check_id
            check_id = None
        if (
            isinstance(check_id, str)
            and isinstance(capability, Mapping)
            and not supplied_collections
            and item is None
        ):
            seed_id = seed_id or check_id
            check_id = capability
            capability = None
        if len(supplied_collections) > 1:
            raise TypeError("pass only one of placements=, items=, or assignments=")
        if supplied_collections and (check_id is not None or capability is not None or item is not None):
            raise TypeError("mapping placements cannot be combined with a single assignment")
        if item is not None:
            if capability is not None:
                raise TypeError("pass only one of capability= or item=")
            capability = item
        if supplied_collections:
            raw = supplied_collections[0]
        elif isinstance(check_id, Mapping):
            if capability is not None:
                raise TypeError("mapping placement cannot have capability=")
            raw = check_id
        elif check_id is None:
            if capability is not None:
                raise TypeError("capability= requires a check ID")
            raw = ()
        else:
            if capability is None:
                raise TypeError("single placement requires capability=")
            raw = ((check_id, capability),)
        if not isinstance(seed_id, str):
            raise TypeError("seed_id must be a string")
        object.__setattr__(self, "assignments", _placement_assignments(raw))
        object.__setattr__(self, "seed_id", seed_id)

    @classmethod
    def from_mapping(
        cls,
        placements: Mapping[str, CapabilityValue],
        *,
        seed_id: str = "",
    ) -> SeedPlacement:
        return cls(placements=placements, seed_id=seed_id)

    overlay = from_mapping

    @property
    def check_id(self) -> str | None:
        return self.assignments[0][0] if len(self.assignments) == 1 else None

    @property
    def capability(self) -> CapabilityId | None:
        return self.assignments[0][1] if len(self.assignments) == 1 else None

    @property
    def item(self) -> CapabilityId | None:
        return self.capability

    @property
    def placements(self) -> Mapping[str, CapabilityId]:
        return MappingProxyType(dict(self.assignments))

    @property
    def items(self) -> Mapping[str, CapabilityId]:
        return self.placements

    @property
    def mapping(self) -> Mapping[str, CapabilityId]:
        return self.placements

    def item_for(self, check_id: str) -> CapabilityId | None:
        return dict(self.assignments).get(check_id)

    capability_for = item_for

    def __getitem__(self, check_id: str) -> CapabilityId:
        return self.placements[check_id]

    def __contains__(self, check_id: object) -> bool:
        return check_id in self.placements

    def __iter__(self):
        return iter(self.assignments)

    def merge(self, other: SeedPlacement) -> SeedPlacement:
        if not isinstance(other, SeedPlacement):
            raise TypeError("can only merge another SeedPlacement")
        return SeedPlacement(
            placements=(*self.assignments, *other.assignments),
            seed_id=self.seed_id or other.seed_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seedId": self.seed_id,
            "placements": {check_id: str(capability) for check_id, capability in self.assignments},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def coerce_placement(
    placements: SeedPlacement
    | Mapping[str, CapabilityValue]
    | Iterable[SeedPlacement]
    | Iterable[tuple[str, CapabilityValue]]
    | None,
) -> SeedPlacement:
    """Normalize one overlay, a mapping, or a sequence of overlays."""
    if placements is None:
        return SeedPlacement()
    if isinstance(placements, SeedPlacement):
        return placements
    if isinstance(placements, Mapping):
        return SeedPlacement(placements=placements)
    values = tuple(placements)
    if not values:
        return SeedPlacement()
    if all(isinstance(value, SeedPlacement) for value in values):
        merged = SeedPlacement()
        for value in values:
            merged = merged.merge(value)  # type: ignore[arg-type]
        return merged
    return SeedPlacement(placements=values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, init=False)
class ItemCheck:
    """A reachable item location whose result is supplied by a seed overlay."""

    check_id: str
    node_id: Hashable
    requirement: Requirement
    label: str
    item: CapabilityId | None

    def __init__(
        self,
        check_id: str,
        node_id: Hashable | None = None,
        requirement: Requirement | CapabilityId | str | Iterable[Requirement | CapabilityId | str] | None = None,
        *,
        node: Hashable | None = None,
        requires: Requirement | CapabilityId | str | Iterable[Requirement | CapabilityId | str] | None = None,
        label: str = "",
        item: CapabilityValue | None = None,
        capability: CapabilityValue | None = None,
    ) -> None:
        if node_id is not None and node is not None and node_id != node:
            raise TypeError("node_id and node disagree")
        resolved_node = node_id if node_id is not None else node
        if resolved_node is None:
            raise TypeError("ItemCheck requires node_id (or node=)")
        if requirement is not None and requires is not None:
            raise TypeError("pass only one of requirement= or requires=")
        raw_requirement = requirement if requirement is not None else requires
        if raw_requirement is None:
            resolved_requirement = TRUE_REQUIREMENT
        elif isinstance(raw_requirement, Requirement):
            resolved_requirement = raw_requirement
        elif isinstance(raw_requirement, (CapabilityId, str)):
            resolved_requirement = _coerce_requirement(raw_requirement)
        else:
            resolved_requirement = AllOf(raw_requirement)
        if item is not None and capability is not None:
            raise TypeError("pass only one of item= or capability=")
        raw_item = item if item is not None else capability
        resolved_item = None if raw_item is None else CapabilityId.coerce(raw_item)
        if not isinstance(check_id, str) or not check_id:
            raise ValueError("check_id must be a non-empty string")
        object.__setattr__(self, "check_id", check_id)
        object.__setattr__(self, "node_id", resolved_node)
        object.__setattr__(self, "requirement", resolved_requirement)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "item", resolved_item)

    @property
    def node(self) -> Hashable:
        return self.node_id

    @property
    def id(self) -> str:
        return self.check_id

    @property
    def requires(self) -> Requirement:
        return self.requirement

    @property
    def capability(self) -> CapabilityId | None:
        return self.item

    @property
    def location(self) -> Hashable:
        return self.node_id

    def can_collect(self, state: ProgressionState) -> bool:
        return (
            state.node == self.node_id
            and self.check_id not in state.collected_checks
            and self.requirement.satisfied_by(state.capabilities)
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "checkId": self.check_id,
            "nodeId": self.node_id,
            "requires": self.requirement.to_dict(),
        }
        if self.label:
            result["label"] = self.label
        if self.item is not None:
            result["item"] = str(self.item)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _state_capability(value: CapabilityValue) -> CapabilityValue:
    if isinstance(value, CapabilityId):
        return value
    if not isinstance(value, str):
        raise TypeError("state capabilities must be CapabilityId or strings")
    if ":" in value:
        return CapabilityId.parse(value)
    return _normalize_legacy_capability(value)


@dataclass(frozen=True, slots=True, init=False)
class ProgressionState:
    """Immutable monotonic state for a node, inventory, and collected checks."""

    node: Hashable
    capabilities: frozenset[CapabilityValue]
    collected_checks: frozenset[str]

    def __init__(
        self,
        node: Hashable,
        capabilities: Iterable[CapabilityValue] = frozenset(),
        collected_checks: Iterable[str] = frozenset(),
    ) -> None:
        normalized_caps = frozenset(_state_capability(value) for value in capabilities)
        normalized_checks = frozenset(collected_checks)
        if not all(isinstance(value, str) for value in normalized_checks):
            raise TypeError("collected check IDs must be strings")
        object.__setattr__(self, "node", node)
        object.__setattr__(self, "capabilities", normalized_caps)
        object.__setattr__(self, "collected_checks", normalized_checks)

    @classmethod
    def from_legacy(
        cls,
        node: Hashable,
        namespace: str,
        capabilities: Iterable[str] = frozenset(),
        collected_checks: Iterable[str] = frozenset(),
    ) -> ProgressionState:
        return cls(
            node,
            (CapabilityId(namespace, value) for value in capabilities),
            collected_checks,
        )

    def satisfies(
        self,
        requirement: Requirement | CapabilityId | str | Iterable[CapabilityId | str],
    ) -> bool:
        if isinstance(requirement, Requirement):
            return requirement.satisfied_by(self.capabilities)
        if isinstance(requirement, CapabilityId):
            return Has(requirement).satisfied_by(self.capabilities)
        if isinstance(requirement, str):
            if ":" in requirement:
                return Has(CapabilityId.parse(requirement)).satisfied_by(self.capabilities)
            return _normalize_legacy_capability(requirement) in self.capabilities
        return AllOf(requirement).satisfied_by(self.capabilities)

    has = satisfies

    def can_collect(self, check: ItemCheck) -> bool:
        return check.can_collect(self)

    def collect(
        self,
        check: ItemCheck,
        placement: SeedPlacement
        | Mapping[str, CapabilityValue]
        | Iterable[SeedPlacement]
        | Iterable[tuple[str, CapabilityValue]]
        | CapabilityId
        | str
        | None = None,
        *,
        item: CapabilityValue | None = None,
    ) -> ProgressionState:
        if self.node != check.node_id:
            raise ValueError(
                f"check {check.check_id!r} is at node {check.node_id!r}, "
                f"not {self.node!r}"
            )
        if check.check_id in self.collected_checks:
            raise ValueError(f"check {check.check_id!r} has already been collected")
        if not check.requirement.satisfied_by(self.capabilities):
            raise ValueError(f"check {check.check_id!r} requirements are not satisfied")
        if isinstance(placement, CapabilityId) or (
            isinstance(placement, str) and ":" in placement
        ):
            if item is not None:
                raise ValueError("pass only one direct item or item=")
            item = placement
            placement = None
        placed = coerce_placement(placement).item_for(check.check_id)
        if placed is not None and item is not None and placed != CapabilityId.coerce(item):
            raise ValueError(f"conflicting item for check {check.check_id!r}")
        acquired = placed
        if acquired is None and item is not None:
            acquired = CapabilityId.coerce(item)
        if acquired is None:
            acquired = check.item
        next_capabilities = self.capabilities
        if acquired is not None:
            next_capabilities = next_capabilities | frozenset({acquired})
        return ProgressionState(
            self.node,
            next_capabilities,
            self.collected_checks | frozenset({check.check_id}),
        )

    collect_check = collect

    @property
    def collected_check_ids(self) -> frozenset[str]:
        return self.collected_checks

    def at(self, node: Hashable) -> ProgressionState:
        return ProgressionState(node, self.capabilities, self.collected_checks)

    with_node = at

    def qualified_capabilities(self, namespace: str) -> frozenset[CapabilityId]:
        result: set[CapabilityId] = set()
        for value in self.capabilities:
            if isinstance(value, CapabilityId):
                result.add(value)
            else:
                result.add(CapabilityId(namespace, value))
        return frozenset(result)

    def legacy_capabilities(self, namespace: str | None = None) -> frozenset[str]:
        result: set[str] = set()
        for value in self.capabilities:
            if isinstance(value, CapabilityId):
                if namespace is None or value.namespace != namespace:
                    raise ValueError(
                        "cannot adapt capabilities from multiple namespaces to strings"
                    )
                result.add(value.name)
            else:
                result.add(value)
        return frozenset(result)

    as_legacy = legacy_capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "capabilities": sorted(str(value) for value in self.capabilities),
            "collectedChecks": sorted(self.collected_checks),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


__all__ = [
    "AllOf",
    "AnyOf",
    "CapabilityId",
    "CapabilityValue",
    "FALSE_REQUIREMENT",
    "Has",
    "ItemCheck",
    "ProgressionState",
    "Requirement",
    "SeedPlacement",
    "TRUE_REQUIREMENT",
    "coerce_placement",
]
