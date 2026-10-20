"""SI data contract. A dataset represents one machine, fluid and power boundary."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

TYPES = {"rolling_piston", "scroll"}
INPUT_COLUMNS = ("point_id", "suction_pressure_pa", "discharge_pressure_pa", "suction_temperature_k", "speed_rpm")
TARGETS = ("mass_flow_kg_s", "power_w")


def positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name}: expected finite positive SI value")
    return number


def identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: non-empty string required")
    return value.strip()


@dataclass(frozen=True)
class CompressorSpec:
    compressor_type: str
    manufacturer: str
    model_id: str
    refrigerant: str
    displacement_m3_rev: float

    def __post_init__(self):
        if self.compressor_type not in TYPES:
            raise ValueError(f"Unsupported compressor_type: {self.compressor_type}")
        for name in ("manufacturer", "model_id", "refrigerant"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "displacement_m3_rev", positive(self.displacement_m3_rev, "displacement_m3_rev"))


@dataclass(frozen=True)
class OperatingPoint:
    point_id: str
    suction_pressure_pa: float
    discharge_pressure_pa: float
    suction_temperature_k: float
    speed_rpm: float

    def __post_init__(self):
        object.__setattr__(self, "point_id", identifier(self.point_id, "point_id"))
        for name in INPUT_COLUMNS[1:]:
            object.__setattr__(self, name, positive(getattr(self, name), name))
        if self.discharge_pressure_pa <= self.suction_pressure_pa:
            raise ValueError(f"{self.point_id}: discharge absolute pressure must exceed suction pressure")


@dataclass(frozen=True)
class Observation:
    point: OperatingPoint
    mass_flow_kg_s: float
    power_w: float
    group_id: str

    def __post_init__(self):
        object.__setattr__(self, "group_id", identifier(self.group_id, "group_id"))
        for name in TARGETS:
            object.__setattr__(self, name, positive(getattr(self, name), name))


@dataclass(frozen=True)
class Dataset:
    spec: CompressorSpec
    observations: list[Observation]
    metadata: dict
    fingerprint: str


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV has missing or duplicate headers")
        if not set(INPUT_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"CSV requires {INPUT_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no operating points")
    if any(None in row for row in rows):
        raise ValueError("CSV row contains excess columns")
    return rows


def load_points(path):
    points = [OperatingPoint(**{k: row[k] for k in INPUT_COLUMNS}) for row in read_rows(path)]
    if len({p.point_id for p in points}) != len(points):
        raise ValueError("Duplicate point_id")
    return points


def load_dataset(csv_path, meta_path):
    raw = Path(csv_path).read_bytes()
    meta_raw = Path(meta_path).read_bytes()
    meta = json.loads(meta_raw.decode("utf-8-sig"))
    if not isinstance(meta, dict):
        raise ValueError("Metadata root must be a JSON object")
    required = (
        "dataset_id",
        "source_kind",
        "validation_role",
        "source",
        "license",
        "pressure_basis",
        "speed_definition",
        "power_boundary",
        "compressor",
    )
    for key in required:
        if key not in meta or not meta[key]:
            raise ValueError(f"metadata.{key} is required")
    for key in ("dataset_id", "source", "license", "power_boundary"):
        identifier(meta[key], f"metadata.{key}")
    if meta["pressure_basis"] != "absolute" or meta["speed_definition"] not in {
        "mechanical_rpm",
        "nominal_mechanical_rpm",
    }:
        raise ValueError(
            "Explicit absolute pressure and mechanical_rpm (or sourced fixed nominal_mechanical_rpm) required; no frequency conversion is inferred"
        )
    if meta["source_kind"] not in {"experiment", "digitized_figure", "manufacturer_map", "synthetic"}:
        raise ValueError("Invalid source_kind")
    if meta["validation_role"] not in {"software_test", "public_validation", "partner_application"}:
        raise ValueError("Invalid validation_role")
    if meta["source_kind"] == "synthetic" and meta["validation_role"] != "software_test":
        raise ValueError("Synthetic data can only be software_test")
    if "reference_enthalpy_difference_j_kg" in meta:
        positive(meta["reference_enthalpy_difference_j_kg"], "reference_enthalpy_difference_j_kg")
        identifier(meta.get("capacity_reference"), "capacity_reference")
    spec = CompressorSpec(**meta["compressor"])
    observations = []
    ids, inputs_to_group = set(), {}
    for row in read_rows(csv_path):
        if not set((*TARGETS, "group_id")).issubset(row):
            raise ValueError("Observed CSV requires mass_flow_kg_s, power_w and group_id")
        p = OperatingPoint(**{k: row[k] for k in INPUT_COLUMNS})
        o = Observation(p, row[TARGETS[0]], row[TARGETS[1]], row["group_id"])
        if p.point_id in ids:
            raise ValueError(f"Duplicate point_id: {p.point_id}")
        ids.add(p.point_id)
        key = tuple(getattr(p, k) for k in INPUT_COLUMNS[1:])
        if key in inputs_to_group and inputs_to_group[key] != o.group_id:
            raise ValueError("Identical conditions must share a group_id to avoid leakage")
        inputs_to_group[key] = o.group_id
        observations.append(o)
    if meta["speed_definition"] == "nominal_mechanical_rpm":
        provenance = meta.get("nominal_speed_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("nominal_mechanical_rpm requires nominal_speed_provenance")
        identifier(provenance.get("source_location"), "nominal_speed_provenance.source_location")
        rate = positive(provenance.get("volume_rate_m3_s"), "nominal_speed_provenance.volume_rate_m3_s")
        expected = 60 * rate / spec.displacement_m3_rev
        if any(not math.isclose(o.point.speed_rpm, expected, rel_tol=1e-9, abs_tol=1e-6) for o in observations):
            raise ValueError("Nominal speed must be fixed and match the sourced rated volume-rate/displacement ratio")
    fingerprint = hashlib.sha256(b"octwin-dataset-v1\0" + raw + b"\0" + meta_raw).hexdigest()
    return Dataset(spec, observations, meta, fingerprint)
