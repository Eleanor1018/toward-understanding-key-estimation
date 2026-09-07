#!/usr/bin/env python3
"""Apply offline acceptance gates to phase-RSI A/B evaluation reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateSpec:
    """Description of one metric gate in an ``evaluate.py`` result."""

    name: str
    metric_path: tuple[str, ...]
    operator: str
    threshold: float

    def accepts(self, value: float) -> bool:
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<":
            return value < self.threshold
        raise ValueError(f"unsupported gate operator: {self.operator}")


GATES = (
    GateSpec("survival", ("survival_fraction",), ">=", 0.80),
    GateSpec(
        "fixed_horizon_forward_directional_velocity",
        (
            "survival_adjusted_command_bin_tracking",
            "forward",
            "mean_directional_velocity",
        ),
        ">=",
        0.15,
    ),
    GateSpec("valid_landing_rate", ("gait", "landing_rate_hz"), ">=", 1.0),
    GateSpec(
        "alternating_landing_fraction",
        ("gait", "alternating_landing_fraction"),
        ">=",
        0.80,
    ),
    GateSpec(
        "action_saturation",
        ("raw_action_saturation_fraction",),
        "<",
        0.02,
    ),
)

_MISSING = object()


class EvaluationReportError(ValueError):
    """Raised when an evaluation report cannot identify a final checkpoint."""


def select_final_checkpoint_result(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the checkpoint result with the greatest checkpoint iteration."""

    results = report.get("results", _MISSING)
    if results is _MISSING:
        raise EvaluationReportError("missing top-level 'results' list")
    if not isinstance(results, list) or not results:
        raise EvaluationReportError("top-level 'results' must be a non-empty list")

    checkpoint_results: list[tuple[int, int, Mapping[str, Any]]] = []
    for position, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise EvaluationReportError(f"results[{position}] must be an object")

        checkpoint = result.get("checkpoint", _MISSING)
        if checkpoint is None:
            # ``evaluate.py`` represents the optional zero-action baseline this way.
            continue
        if not isinstance(checkpoint, str) or not checkpoint:
            raise EvaluationReportError(
                f"results[{position}].checkpoint must be a non-empty string or null"
            )

        iteration = result.get("checkpoint_iteration", _MISSING)
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise EvaluationReportError(
                f"results[{position}].checkpoint_iteration must be an integer"
            )
        checkpoint_results.append((iteration, position, result))

    if not checkpoint_results:
        raise EvaluationReportError("report contains no checkpoint results")

    return max(checkpoint_results, key=lambda item: (item[0], item[1]))[2]


def _read_metric(result: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = result
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _evaluate_gate(result: Mapping[str, Any], gate: GateSpec) -> dict[str, Any]:
    metric = ".".join(gate.metric_path)
    raw_value = _read_metric(result, gate.metric_path)
    gate_result: dict[str, Any] = {
        "metric": metric,
        "operator": gate.operator,
        "threshold": gate.threshold,
        "value": None,
        "passed": False,
    }

    if raw_value is _MISSING:
        gate_result["reason"] = f"missing metric: {metric}"
        return gate_result
    if raw_value is None:
        gate_result["reason"] = f"null metric: {metric}"
        return gate_result
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        gate_result["value"] = raw_value
        gate_result["reason"] = f"non-numeric metric: {metric}"
        return gate_result

    value = float(raw_value)
    if not math.isfinite(value):
        gate_result["value"] = str(raw_value)
        gate_result["reason"] = f"non-finite metric: {metric}"
        return gate_result

    gate_result["value"] = value
    gate_result["passed"] = gate.accepts(value)
    if not gate_result["passed"]:
        gate_result["reason"] = (
            f"metric {metric}={value} does not satisfy {gate.operator} {gate.threshold}"
        )
    return gate_result


def _selection_failure_gates(reason: str) -> dict[str, dict[str, Any]]:
    return {
        gate.name: {
            "metric": ".".join(gate.metric_path),
            "operator": gate.operator,
            "threshold": gate.threshold,
            "value": None,
            "passed": False,
            "reason": f"final checkpoint unavailable: {reason}",
        }
        for gate in GATES
    }


def evaluate_arm(report: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Select and gate one control or treatment evaluation report."""

    try:
        result = select_final_checkpoint_result(report)
    except EvaluationReportError as exc:
        return {
            "source": source,
            "checkpoint": None,
            "checkpoint_iteration": None,
            "passed": False,
            "error": str(exc),
            "gates": _selection_failure_gates(str(exc)),
        }

    gates = {gate.name: _evaluate_gate(result, gate) for gate in GATES}
    return {
        "source": source,
        "checkpoint": result["checkpoint"],
        "checkpoint_iteration": result["checkpoint_iteration"],
        "passed": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
    }


def evaluate_ab(
    control_report: Mapping[str, Any],
    treatment_report: Mapping[str, Any],
    *,
    control_source: str = "control",
    treatment_source: str = "treatment",
) -> dict[str, Any]:
    """Build the complete A/B gate report."""

    control = evaluate_arm(control_report, control_source)
    treatment = evaluate_arm(treatment_report, treatment_source)
    return {
        "schema_version": "phase_rsi_ab_gates_v1",
        "passed": control["passed"] and treatment["passed"],
        "control": control,
        "treatment": treatment,
    }


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationReportError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationReportError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationReportError(f"evaluation report {path} must contain an object")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply phase-RSI acceptance gates to control/treatment reports."
    )
    parser.add_argument("control", type=Path, help="control evaluation JSON")
    parser.add_argument("treatment", type=Path, help="treatment evaluation JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the emitted gate report to this JSON file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        control_report = _load_json_object(args.control)
        treatment_report = _load_json_object(args.treatment)
    except EvaluationReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = evaluate_ab(
        control_report,
        treatment_report,
        control_source=str(args.control),
        treatment_source=str(args.treatment),
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
