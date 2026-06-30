#!/usr/bin/env python3
"""
Generate Triton config.pbtxt + adapter.yaml for a detection ONNX model, or an
adapter.yaml-only package for a pseudo-model. Mirrors the task types and v3 rule
types consumed by the Savant worker.

Detection adapter schema versions:

    v1 — flat alert_labels + alert_category/alert_severity
    v2 — per-label alert_rules (presence/absence) with optional default_alert
    v3 — typed alert_rules (zones, dwell, gathering, temporal, speed_violation)
         + OC-SORT tracker

Pseudo-models (--task-type, no ONNX, adapter.yaml only):

    image_quality — poor visibility (#7) + covered/dirty camera (#30):
                    brisque / laplacian_var / intensity_var metric rules
    scene         — cross-model rules over merged detections:
                    suspended_load (#26/#27/#31) + line_of_fire (#15/#23/#24)

Files are saved in the SAME folder as the model.onnx by default (detection), or
in --output-dir / ./<name> for pseudo-models.

Usage:
    python generate_model_package.py --model fire_model/model.onnx
    python generate_model_package.py --model helmet/best_trained.onnx --schema 1
    python generate_model_package.py --model vehicle/model.onnx --schema 3   # speed_violation
    python generate_model_package.py --task-type image_quality --name poor_visibility
    python generate_model_package.py --task-type scene --name scene_rig
    python generate_model_package.py --model fire_model/model.onnx --inspect-only
    python generate_model_package.py --model fire_model/model.onnx --force

When --schema is omitted the wizard asks which version to emit (default 3).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# inspect_model.py is in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_model import (
    build_pbtxt,
    detect_labels_path,
    detect_model_family,
    extract_metadata,
    infer_decoder,
    infer_layout,
    load_labels,
    load_onnx_io,
    parse_metadata_labels,
    print_inspection_summary,
    read_onnx_model,
    resolve_output_dir,
)

VALID_SEVERITIES     = ["low", "medium", "high", "critical"]
VALID_MODES          = ["presence", "absence", "absence_unconditional"]
VALID_INSTANCE_KINDS = ["KIND_CPU", "KIND_GPU"]

# ── v3 rule schema (mirrors V3_RULE_TYPES in savant_worker.py) ──────────────────
# rule type -> type-specific required fields (besides rule_id + type)
V3_RULE_FIELDS = {
    "presence":               [],
    "absence":                [],
    "absence_unconditional":  [],
    "temporal_presence":      ["min_consecutive_frames"],
    "zone_violation":         ["zone_id", "person_label"],
    "zone_duration":          ["zone_id", "person_label", "duration_seconds"],
    "zone_absence":           ["zone_id", "absent_label", "guard_label"],
    "gathering":              ["min_count", "duration_seconds"],
    "speed_violation":        ["label", "speed_threshold"],
    # cross-model scene rules — valid only inside a task_type=scene adapter
    "suspended_load":         ["load_label", "person_label", "duration_seconds"],
    "line_of_fire":           ["zone_id", "equipment_label", "person_label"],
    # pose keypoint rules — valid only inside a task_type=pose adapter
    "keypoint_zone_violation": ["zone_id", "keypoint_indices"],
    "handrail_contact":        ["zone_id", "keypoint_indices", "contact_buffer_frames"],
    "stair_descent_violation": ["zone_id", "safe_facing_angle_deg"],
}
# rule types that need a `label` field drawn from the model's labels list
V3_LABEL_REQUIRED   = {"presence", "absence", "absence_unconditional",
                       "temporal_presence", "gathering", "speed_violation",
                       "keypoint_zone_violation", "handrail_contact", "stair_descent_violation"}
# rule types that require tracker.enabled: true
V3_TRACKER_RULES    = {"zone_duration", "gathering", "speed_violation",
                       "suspended_load", "line_of_fire",
                       "keypoint_zone_violation", "handrail_contact", "stair_descent_violation"}
# scene-only rule types (operate on merged cross-model detections, not this model's)
V3_SCENE_RULE_TYPES = {"suspended_load", "line_of_fire"}
# pose-only rule types (require task_type=pose with yolo_pose decoder)
V3_POSE_RULE_TYPES  = {"keypoint_zone_violation", "handrail_contact", "stair_descent_violation"}
# rule types selectable for a single detection model (scene/pose rules live in their own adapters)
V3_DETECTION_RULE_TYPES = [t for t in V3_RULE_FIELDS if t not in V3_SCENE_RULE_TYPES | V3_POSE_RULE_TYPES]
# rule types selectable inside a pose adapter
V3_POSE_DETECTION_RULE_TYPES = [t for t in V3_RULE_FIELDS if t not in V3_SCENE_RULE_TYPES]
# rule fields that must reference an entry in the labels list
V3_LABEL_FIELDS     = ("label", "person_label", "absent_label", "guard_label")
# image_quality pseudo-model rule types and the metrics they may test
V3_IMAGE_QUALITY_RULE_TYPES = ["image_quality", "camera_obstruction"]
IMAGE_QUALITY_METRICS       = ["brisque", "laplacian_var", "intensity_var"]
COMPARATORS                 = ["gt", "lt"]
SLUG_RE             = re.compile(r"^[a-z0-9_]+$")

BOLD = "\033[1m"; CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; RESET = "\033[0m"
h    = lambda t: f"{BOLD}{CYAN}{t}{RESET}"
ok   = lambda t: f"{GREEN}{t}{RESET}"
warn = lambda t: f"{YELLOW}{t}{RESET}"


# ── Prompt helpers ────────────────────────────────────────────────────────────

def ask(prompt: str, default: str = "", *, choices: list[str] | None = None) -> str:
    ch = f" ({'/'.join(choices)})" if choices else ""
    dh = f" [{default}]"          if default  else ""
    while True:
        raw = input(f"  {prompt}{ch}{dh}: ").strip()
        val = raw or default
        if not val:
            print("  ✗ Required."); continue
        if choices and val not in choices:
            print(f"  ✗ Choose one of: {', '.join(choices)}"); continue
        return val


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw: return default
        try: return float(raw)
        except ValueError: print("  ✗ Enter a number.")


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw: return default
        try:
            v = int(raw)
            if v <= 0: print("  ✗ Must be a positive integer."); continue
            return v
        except ValueError: print("  ✗ Enter a whole number.")


def ask_slug(prompt: str, default: str = "") -> str:
    dh = f" [{default}]" if default else ""
    while True:
        raw = input(f"  {prompt}{dh}: ").strip()
        val = raw or default
        if not val:
            print("  ✗ Required."); continue
        if not SLUG_RE.match(val):
            print("  ✗ Lowercase letters, digits and underscores only."); continue
        return val


def ask_bool(prompt: str, default: bool = True) -> bool:
    raw = input(f"  {prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not raw else raw in ("y", "yes", "1", "true")


def ask_optional(prompt: str) -> str:
    """Ask for an optional string — empty string means 'auto-derive'."""
    return input(f"  {prompt} [Enter to auto-derive]: ").strip()


def ask_labels(detected: list[str]) -> list[str]:
    print(f"  Auto-detected: {ok(', '.join(detected))}")
    raw = input("  Edit labels (comma-separated, Enter to keep): ").strip()
    if not raw: return detected
    ls = [l.strip() for l in raw.split(",") if l.strip()]
    return ls if ls else (print("  ✗ Keeping auto-detected.") or detected)


def ask_triplet(prompt: str, default: str) -> list[float]:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip() or default
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            print("  ✗ Need exactly 3 comma-separated numbers."); continue
        try: return [float(p) for p in parts]
        except ValueError: print("  ✗ All values must be numbers.")


# ── Alert-rule wizard ─────────────────────────────────────────────────────────

def _one_rule(label: str, labels: list[str]) -> dict | None:
    print(f"\n  {h(f'Label: {label}')}")
    if not ask_bool(f"Alert on '{label}'?", default=True):
        return None

    rule_id = ask("rule_id", default=label.lower().replace(" ", "_").replace("-", "_"))
    mode    = ask("mode", default="presence", choices=VALID_MODES)

    guard = None
    if mode == "absence":
        others = [l for l in labels if l != label]
        if not others:
            print(warn("  ⚠ No other labels — switching to absence_unconditional."))
            mode = "absence_unconditional"
        else:
            print(f"  Available guard labels: {', '.join(others)}")
            guard = ask("guard_label", default=others[0])

    category = ask("category", default="safety")
    severity = ask("severity", default="high", choices=VALID_SEVERITIES)
    cooldown = int(ask_float("cooldown_seconds", default=30))
    raw_conf = input("  confidence_threshold [Enter = use global]: ").strip()
    conf     = float(raw_conf) if raw_conf else None
    default_msg = (
        f"{label} detected on camera {{camera}}" if mode == "presence"
        else f"{label} missing on camera {{camera}}"
    )
    message    = ask("message  ({camera} {stream_key} {label} {model})", default=default_msg)
    alert_name = ask_optional("alert_name (display name shown in the alert dashboard)")

    rule: dict = {"rule_id": rule_id, "label": label, "mode": mode}
    if guard:             rule["guard_label"]           = guard
    if conf is not None:  rule["confidence_threshold"]  = conf
    if alert_name:        rule["alert_name"]            = alert_name
    rule.update({"category": category, "severity": severity,
                 "message": message, "cooldown_seconds": cooldown})
    return rule


def alert_wizard(labels: list[str]) -> tuple[list[dict], dict | None, list[str]]:
    print(f"\n{h('─' * 55)}\n{h('Alert Rules Wizard')}")
    print(f"  Labels: {ok(', '.join(labels))}\n")

    use_default = ask_bool("Use a shared default_alert for labels with identical rules?", True)
    default_alert: dict | None = None
    if use_default:
        print(f"\n  {h('Default alert settings')}")
        _def_alert_name = ask_optional("default alert_name (dashboard display name, e.g. 'Intrusion Detected')")
        default_alert = {
            "mode":             ask("default mode",     default="presence", choices=VALID_MODES),
            "category":         ask("default category", default="safety"),
            "severity":         ask("default severity", default="medium",   choices=VALID_SEVERITIES),
            "cooldown_seconds": int(ask_float("default cooldown_seconds", default=30)),
        }
        if _def_alert_name:
            default_alert["alert_name"] = _def_alert_name

    rules:        list[dict] = []
    alert_labels: list[str]  = []

    for label in labels:
        print(f"\n{h('─' * 55)}")
        if default_alert:
            if ask_bool(f"Custom rule for '{label}' (overrides default)?", default=False):
                r = _one_rule(label, labels)
                if r: rules.append(r)
            elif ask_bool(f"Include '{label}' in alert_labels (uses default)?", default=True):
                alert_labels.append(label)
        else:
            r = _one_rule(label, labels)
            if r: rules.append(r)

    return rules, default_alert, alert_labels


# ── v1 alert wizard ─────────────────────────────────────────────────────────

def alert_wizard_v1(labels: list[str]) -> tuple[list[str], str, str]:
    """schema_version=1: one flat set of alert_labels + a single category/severity."""
    print(f"\n{h('─' * 55)}\n{h('Alert Settings (v1)')}")
    print(f"  Labels: {ok(', '.join(labels))}\n")
    raw = input("  alert_labels (comma-separated subset, Enter = all labels): ").strip()
    if not raw:
        alert_labels = list(labels)
    else:
        wanted = [l.strip() for l in raw.split(",") if l.strip()]
        unknown = [l for l in wanted if l not in labels]
        if unknown:
            print(warn(f"  ⚠ Ignoring unknown labels: {', '.join(unknown)}"))
        alert_labels = [l for l in wanted if l in labels] or list(labels)
    category = ask("alert_category", default="detection")
    severity = ask("alert_severity", default="medium", choices=VALID_SEVERITIES)
    return alert_labels, category, severity


# ── v3 alert wizard ─────────────────────────────────────────────────────────

def _one_rule_v3(labels: list[str], index: int, pose_mode: bool = False) -> dict:
    print(f"\n  {h(f'Rule #{index}')}")
    available = sorted(V3_POSE_DETECTION_RULE_TYPES if pose_mode else V3_DETECTION_RULE_TYPES)
    default_type = "keypoint_zone_violation" if pose_mode else "presence"
    rtype = ask("rule type", default=default_type, choices=available)
    rule_id = ask("rule_id", default=f"{rtype}_{index}")
    rule: dict = {"rule_id": rule_id, "type": rtype}

    if rtype in V3_LABEL_REQUIRED:
        rule["label"] = ask("label (detection class)", default=labels[0], choices=labels)

    for field in V3_RULE_FIELDS[rtype]:
        if field == "zone_id":
            rule[field] = ask_slug("zone_id (must match a zone drawn in the dashboard)")
        elif field in ("person_label", "absent_label", "guard_label"):
            rule[field] = ask(field, default=labels[0], choices=labels)
        elif field == "min_consecutive_frames":
            rule[field] = ask_int("min_consecutive_frames", default=3)
        elif field == "min_count":
            rule[field] = ask_int("min_count (people in cluster)", default=4)
        elif field == "duration_seconds":
            rule[field] = ask_float("duration_seconds", default=15.0)
        elif field == "speed_threshold":
            rule[field] = ask_float("speed_threshold (world units/sec, e.g. 8.0 ≈ 8 m/s)", default=8.0)
        elif field == "keypoint_indices":
            print(ok("    COCO indices: 9=l_wrist 10=r_wrist  5=l_shoulder 6=r_shoulder  11=l_hip 12=r_hip  15=l_ankle 16=r_ankle"))
            raw = ask("keypoint_indices (comma-separated COCO indices, e.g. 9,10 = both wrists)", default="9,10")
            rule[field] = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        elif field == "contact_buffer_frames":
            rule[field] = ask_int("contact_buffer_frames (no-contact frames before hand-removed alert)", default=10)
        elif field == "safe_facing_angle_deg":
            print(ok("    0=right  90=down  180=left  270=up  (hip→shoulder vector in image space)"))
            rule[field] = ask_float("safe_facing_angle_deg (expected body facing angle when descending safely)", default=90.0)

    if rtype == "gathering":
        rule["cluster_eps"]         = ask_float("cluster_eps (image-space distance)", default=0.0625)
        rule["grace_seconds"]       = ask_float("grace_seconds", default=1.0)
        rule["group_match_jaccard"] = ask_float("group_match_jaccard", default=0.4)

    if rtype == "speed_violation":
        rule["min_track_frames"] = ask_int("min_track_frames (samples before judging speed)", default=2)
        print(warn("  ⚠ speed_violation needs a per-camera homography: "
                   "PUT /api/v1/cameras/{id}/calibration before this rule will fire."))

    if rtype == "stair_descent_violation":
        rule["angle_tolerance_deg"] = ask_float("angle_tolerance_deg (deviation before UNSAFE fires)", default=60.0)
        print(warn("  ⚠ stair_descent_violation requires a YOLOv8-pose ONNX model (task_type: pose)."))
        print(warn("    Draw the stair zone polygon in the dashboard before testing."))

    if rtype in ("keypoint_zone_violation", "handrail_contact"):
        print(warn("  ⚠ Pose rules require a YOLOv8-pose ONNX model (task_type: pose)."))
        print(warn("    Draw the zone polygon in the dashboard before testing."))

    rule["category"]         = ask("category", default="safety")
    rule["severity"]         = ask("severity", default="high", choices=VALID_SEVERITIES)
    rule["cooldown_seconds"] = int(ask_float("cooldown_seconds", default=30))
    default_msg = f"{rule.get('label', rtype)} on camera {{camera}} ({{stream_key}})"
    rule["message"]    = ask("message  ({camera} {stream_key} {label} {model})", default=default_msg)
    alert_name = ask_optional("alert_name (dashboard display name, e.g. 'Zone Violation', 'Worker Gathering')")
    if alert_name:
        rule["alert_name"] = alert_name
    return rule


def alert_wizard_v3(labels: list[str], pose_mode: bool = False) -> list[dict]:
    print(f"\n{h('─' * 55)}\n{h('Alert Rules Wizard (v3)')}")
    print(f"  Labels: {ok(', '.join(labels))}")
    available_types = sorted(V3_POSE_DETECTION_RULE_TYPES if pose_mode else V3_DETECTION_RULE_TYPES)
    print("  Rule types: " + ok(", ".join(available_types)))
    print(f"  Zone rules need zones drawn in the dashboard (zone_id matches the slug there).")
    if not pose_mode:
        print(f"  speed_violation also needs a per-camera homography calibration.")
        print(f"  (suspended_load / line_of_fire are cross-model — use --task-type scene.)")
    else:
        print(ok("  Pose rule types: keypoint_zone_violation  handrail_contact  stair_descent_violation"))
        print(warn("  Pose rules need a YOLOv8-pose ONNX model and zones drawn in the dashboard."))
    print()

    rules: list[dict] = []
    index = 1
    while True:
        if rules and not ask_bool("Add another rule?", default=False):
            break
        if not rules and not ask_bool("Add an alert rule?", default=True):
            break
        rules.append(_one_rule_v3(labels, index, pose_mode=pose_mode))
        index += 1
    return rules


def tracker_wizard(force_enabled: bool) -> dict:
    print(f"\n{h('─' * 55)}\n{h('Tracker (OC-SORT)')}")
    if force_enabled:
        print(ok("  tracker-dependent rules present → tracker.enabled forced to true"))
        enabled = True
    else:
        enabled = ask_bool("Enable the OC-SORT tracker?", default=True)
    if not enabled:
        return {"enabled": False}
    return {
        "enabled":              True,
        "type":                 "ocsort",
        "max_age":              ask_int("tracker max_age (frames to keep lost tracks)", default=30),
        "min_hits":             ask_int("tracker min_hits (frames before a track is confirmed)", default=3),
        "grace_period_seconds": ask_float("tracker grace_period_seconds (occlusion tolerance)", default=2.0),
    }


# ── Build adapter dict ────────────────────────────────────────────────────────

def _base_adapter(
    *, schema_version, task_type="detection", decoder, labels, layout, color_order,
    scale, mean, std, input_size, confidence_threshold, nms_threshold,
) -> dict:
    pre: dict = {"layout": layout, "color_order": color_order,
                 "scale": scale, "mean": mean, "std": std}
    if input_size: pre["input_size"] = input_size
    return {
        "schema_version": schema_version, "task_type": task_type, "decoder": decoder,
        "preprocess": pre,
        "postprocess": {"confidence_threshold": confidence_threshold,
                        "nms_threshold": nms_threshold},
        "labels": labels,
    }


def build_adapter_v1(
    *, alert_labels, alert_category, alert_severity, task_type="detection", **base,
) -> dict:
    a = _base_adapter(schema_version=1, task_type=task_type, **base)
    a["alert_labels"]   = alert_labels
    a["alert_category"] = alert_category
    a["alert_severity"] = alert_severity
    return a


def build_adapter_v3(*, tracker, alert_rules, task_type="detection", **base) -> dict:
    a = _base_adapter(schema_version=3, task_type=task_type, **base)
    if tracker:     a["tracker"]     = tracker
    if alert_rules: a["alert_rules"] = alert_rules
    return a


def build_adapter(*, alert_rules, default_alert, alert_labels, task_type="detection", **base) -> dict:
    a = _base_adapter(schema_version=2, task_type=task_type, **base)
    if default_alert: a["default_alert"] = default_alert
    if alert_labels:  a["alert_labels"]  = alert_labels
    if alert_rules:   a["alert_rules"]   = alert_rules
    return a


# ── Pseudo-model wizards (no ONNX, no config.pbtxt) ─────────────────────────────

def _one_rule_image_quality(index: int) -> dict:
    print(f"\n  {h(f'Rule #{index}')}")
    rtype  = ask("rule type", default="image_quality", choices=V3_IMAGE_QUALITY_RULE_TYPES)
    rule_id = ask("rule_id", default=f"{rtype}_{index}")
    metric  = ask("metric", default="brisque", choices=IMAGE_QUALITY_METRICS)
    if metric == "brisque":
        print(ok("    brisque: no-reference quality, higher = worse (needs the brisque package)."))
    comparator = ask("comparator (gt = above threshold fires, lt = below)",
                     default="gt" if metric == "brisque" else "lt", choices=COMPARATORS)
    threshold  = ask_float("threshold", default=55.0 if metric == "brisque" else 80.0)

    rule: dict = {"rule_id": rule_id, "type": rtype, "metric": metric,
                  "comparator": comparator, "threshold": threshold}
    if rtype == "camera_obstruction":
        rule["duration_seconds"] = ask_float("duration_seconds (must hold this long)", default=10.0)

    rule["category"]         = ask("category", default="safety")
    rule["severity"]         = ask("severity", default="medium", choices=VALID_SEVERITIES)
    rule["cooldown_seconds"] = int(ask_float("cooldown_seconds", default=120))
    default_msg = f"{metric} {comparator} {threshold} on camera {{camera}} ({{stream_key}})"
    rule["message"]    = ask("message  ({camera} {stream_key})", default=default_msg)
    alert_name = ask_optional("alert_name (dashboard display name, e.g. 'Poor Visibility', 'Camera Blocked')")
    if alert_name:
        rule["alert_name"] = alert_name
    return rule


def image_quality_wizard() -> dict:
    print(f"\n{h('─' * 55)}\n{h('Image-Quality Pseudo-Model Wizard')}")
    print("  Poor visibility (#7) and covered/dirty camera (#30) — no Triton model.")
    print(f"  Rule types: {ok(', '.join(V3_IMAGE_QUALITY_RULE_TYPES))}")
    print(f"  Metrics: {ok(', '.join(IMAGE_QUALITY_METRICS))}\n")

    grace = ask_float("grace_period_seconds (pauses obstruction timers across frame gaps)", default=4.0)
    rules: list[dict] = []
    index = 1
    while True:
        if rules and not ask_bool("Add another rule?", default=False):
            break
        if not rules and not ask_bool("Add an image-quality rule?", default=True):
            break
        rules.append(_one_rule_image_quality(index))
        index += 1

    adapter = {"schema_version": 3, "task_type": "image_quality",
               "grace_period_seconds": grace, "alert_rules": rules}
    return adapter


def _one_rule_scene(index: int) -> dict:
    print(f"\n  {h(f'Rule #{index}')}")
    rtype  = ask("rule type", default="suspended_load", choices=sorted(V3_SCENE_RULE_TYPES))
    rule_id = ask("rule_id", default=f"{rtype}_{index}")
    rule: dict = {"rule_id": rule_id, "type": rtype}
    print(ok("    Scene labels come from OTHER models assigned to the same camera."))

    if rtype == "suspended_load":
        rule["load_label"]              = ask("load_label (e.g. crane_load, from the machinery model)")
        rule["person_label"]            = ask("person_label", default="person")
        rule["duration_seconds"]        = ask_float("duration_seconds (time under the load)", default=4.0)
        rule["fall_zone_height_factor"] = ask_float("fall_zone_height_factor (downward depth × load height)", default=3.0)
        rule["fall_zone_widen"]         = ask_float("fall_zone_widen (cone widen, fraction of load width)", default=0.25)
        raw_conf = input("  confidence_threshold for loads [Enter = scene default 0.3]: ").strip()
        if raw_conf:
            rule["confidence_threshold"] = float(raw_conf)
    else:  # line_of_fire
        rule["zone_id"]            = ask_slug("zone_id (corridor polygon drawn in the dashboard)")
        rule["equipment_label"]    = ask("equipment_label (e.g. vehicle, from the machinery model)")
        rule["person_label"]       = ask("person_label", default="person")
        rule["motion_threshold_px"] = ask_int("motion_threshold_px (min displacement to count as moving)", default=12)

    rule["category"]         = ask("category", default="safety")
    rule["severity"]         = ask("severity", default="high", choices=VALID_SEVERITIES)
    rule["cooldown_seconds"] = int(ask_float("cooldown_seconds", default=30))
    default_msg = f"{rtype} on camera {{camera}} ({{stream_key}})"
    rule["message"]    = ask("message  ({camera} {stream_key})", default=default_msg)
    alert_name = ask_optional("alert_name (dashboard display name, e.g. 'Suspended Load Risk', 'Line Of Fire')")
    if alert_name:
        rule["alert_name"] = alert_name
    return rule


def scene_wizard() -> dict:
    print(f"\n{h('─' * 55)}\n{h('Scene Pseudo-Model Wizard')}")
    print("  Cross-model rules evaluated on the MERGED detections of every model on the camera.")
    print("  suspended_load (#26/#27/#31) + line_of_fire (#15/#23/#24).")
    print(warn("  The camera must ALSO run the detection models that emit the referenced labels.\n"))

    rules: list[dict] = []
    index = 1
    while True:
        if rules and not ask_bool("Add another rule?", default=False):
            break
        if not rules and not ask_bool("Add a scene rule?", default=True):
            break
        rules.append(_one_rule_scene(index))
        index += 1

    # Scene rules require the tracker (person dwell + equipment motion).
    tracker = tracker_wizard(force_enabled=True)
    adapter = {"schema_version": 3, "task_type": "scene", "tracker": tracker,
               "alert_category": "safety", "alert_severity": "high", "alert_rules": rules}
    return adapter


def write_pseudo_package(task_type: str, out_dir: Path, force: bool) -> int:
    """Generate adapter.yaml only (no ONNX / config.pbtxt) for image_quality or scene."""
    adapter_path = out_dir / "adapter.yaml"
    if adapter_path.exists() and not force:
        sys.exit(f"Already exists: {adapter_path}\nUse --force to overwrite.")
    if not sys.stdin.isatty():
        sys.exit("Interactive wizard requires a terminal.")

    print(f"\n{h('═' * 55)}")
    print(f"{h(f' {task_type} pseudo-model  →  output: {out_dir}')}")
    print(f"{h('═' * 55)}\nPress Enter to accept suggestions.\n")

    adapter = image_quality_wizard() if task_type == "image_quality" else scene_wizard()
    if not adapter["alert_rules"]:
        sys.exit("No rules defined — nothing to write.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with adapter_path.open("w", encoding="utf-8") as fh:
        yaml.dump(adapter, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n{h('═' * 55)}")
    print(f"{ok('✓ Done — ' + str(out_dir))}")
    print(f"{h('═' * 55)}")
    print(f"  {ok('adapter.yaml')} → {adapter_path}")
    print(f"  Task type   : {task_type}  (no Triton model — pseudo-model)")
    print(f"  Rules       : {len(adapter['alert_rules'])}")
    print(f"\n{h('Next:')}")
    print(f"  1. Copy this folder into the worker model root: /models/{out_dir.name}/adapter.yaml")
    print(f"  2. Register an AI Model with triton_model_name='{out_dir.name}' and assign it to camera(s).")
    if task_type == "scene":
        print("  3. Assign the detection models that emit the referenced labels to the same camera.")
    print()
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate config.pbtxt + adapter.yaml for a detection ONNX model.\n"
            "Files are written into the SAME folder as model.onnx by default."
        )
    )
    p.add_argument("--model",         type=Path,
                   help="Path to model.onnx (required for --task-type detection).")
    p.add_argument("--task-type",     default="detection",
                   choices=["detection", "image_quality", "scene", "pose"],
                   help="detection = ONNX model (default); pose = YOLOv8-pose ONNX model; "
                        "image_quality / scene = pseudo-model (no ONNX, writes adapter.yaml only).")
    p.add_argument("--name",          help="Triton/model name. Default: parent folder name.")
    p.add_argument("--output-dir",    type=Path, help="Override output directory.")
    p.add_argument("--labels-path",   type=Path)
    p.add_argument("--labels",        help="Comma-separated labels (overrides auto-detection)")
    p.add_argument("--decoder",       default="auto",
                   choices=["auto", "yolo_v8", "yolo_single_class", "ssd"])
    p.add_argument("--instance-kind", default="KIND_CPU", choices=VALID_INSTANCE_KINDS)
    p.add_argument("--schema",        choices=["1", "2", "3"],
                   help="Adapter schema version to emit. Default: ask (3).")
    p.add_argument("--inspect-only",  action="store_true")
    p.add_argument("--force",         action="store_true", help="Overwrite existing files.")
    args = p.parse_args()

    # ── Pseudo-models (image_quality / scene): no ONNX, adapter.yaml only ──────
    if args.task_type in ("image_quality", "scene"):  # type: ignore[comparison-overlap]
        if args.model and args.model.suffix.lower() == ".onnx":
            sys.exit(f"--task-type {args.task_type} is a pseudo-model — drop --model "
                     "(it has no ONNX). Use --name / --output-dir to set the folder.")
        if not (args.name or args.output_dir):
            sys.exit(f"--task-type {args.task_type} requires --name (or --output-dir) "
                     "to name the model folder.")
        out_dir = (args.output_dir.resolve() if args.output_dir
                   else (Path.cwd() / args.name).resolve())
        return write_pseudo_package(args.task_type, out_dir, args.force)

    if not args.model:
        sys.exit("--model is required for --task-type detection.")
    model_path = args.model.resolve()
    if not model_path.exists(): sys.exit(f"Not found: {model_path}")
    if model_path.suffix.lower() != ".onnx": sys.exit("Only .onnx supported.")

    print(f"\n{h('Loading ONNX…')}")
    model       = read_onnx_model(model_path)
    metadata    = extract_metadata(model)
    onnx_info   = load_onnx_io(model)
    meta_labels = parse_metadata_labels(metadata)
    lpath       = detect_labels_path(
        model_path, args.labels_path.resolve() if args.labels_path else None
    )
    labels, lsrc = load_labels(lpath, args.labels, args.decoder, metadata_labels=meta_labels)
    dec_auto     = infer_decoder(onnx_info, labels, args.decoder, metadata=metadata)
    lay_auto     = infer_layout(onnx_info)
    family       = detect_model_family(model, metadata, onnx_info, labels)

    print_inspection_summary(
        model_path=model_path, model=model, onnx_info=onnx_info, metadata=metadata,
        model_family=family, labels=labels, label_source=lsrc,
        labels_path=lpath, decoder=dec_auto, layout=lay_auto,
    )

    if args.inspect_only: return 0
    if not sys.stdin.isatty(): sys.exit("Interactive wizard requires a terminal.")

    # Default output = same folder as the model
    out_dir = (
        args.output_dir.resolve() if args.output_dir
        else resolve_output_dir(model_path, None)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    default_name = args.name or (
        model_path.parent.parent.name if model_path.parent.name.isdigit()
        else model_path.parent.name
    )
    cfg_path     = out_dir / "config.pbtxt"
    adapter_path = out_dir / "adapter.yaml"

    if not args.force:
        existing = [str(x) for x in (cfg_path, adapter_path) if x.exists()]
        if existing:
            sys.exit(f"Already exists: {', '.join(existing)}\nUse --force to overwrite.")

    # ── Wizard ──────────────────────────────────────────────────────────────
    print(f"\n{h('═' * 55)}")
    print(f"{h(f' Wizard  →  output: {out_dir}')}")
    print(f"{h('═' * 55)}\nPress Enter to accept suggestions.\n")

    print(f"{h('── 1. Model identity ──')}")
    model_name = ask("Triton model name", default=default_name)

    print(f"\n{h('── 2. Labels ──')}")
    labels = ask_labels(labels)

    print(f"\n{h('── 3. Decoder ──')}")
    if args.task_type == "pose":
        decoder = "yolo_pose"
        print(f"  Decoder forced to: {ok('yolo_pose')}  (task_type=pose)")
        print(ok("  Pose keypoint reference (COCO-17):"))
        print(ok("    0=nose  1=l_eye  2=r_eye  3=l_ear  4=r_ear"))
        print(ok("    5=l_shoulder  6=r_shoulder  7=l_elbow  8=r_elbow"))
        print(ok("    9=l_wrist  10=r_wrist  11=l_hip  12=r_hip"))
        print(ok("    13=l_knee  14=r_knee  15=l_ankle  16=r_ankle"))
    else:
        print(f"  Auto-detected: {ok(dec_auto)}")
        decoder = ask("Decoder", default=dec_auto, choices=["yolo_v8", "yolo_single_class", "ssd"])
        if decoder == "yolo_single_class" and len(labels) != 1:
            print(warn("  ⚠ Keeping first label only for yolo_single_class."))
            labels = labels[:1]

    print(f"\n{h('── 4. Preprocessing ──')}")
    print(f"  Auto-detected layout: {ok(lay_auto)}")
    layout      = ask("Layout",      default=lay_auto, choices=["NCHW", "NHWC"])
    color_order = ask("Color order", default="RGB",    choices=["RGB", "BGR"])
    scale       = ask_float("Scale (1/255 ≈ 0.00392)", default=round(1/255, 10))
    mean        = ask_triplet("Mean (e.g. 0,0,0 for YOLO)", "0,0,0")
    std         = ask_triplet("Std  (e.g. 1,1,1 for YOLO)", "1,1,1")
    raw_sz      = input("  Fixed input [w,h] (Enter to auto-detect): ").strip()
    input_size: list[int] | None = None
    if raw_sz:
        try: w, hv = [int(v.strip()) for v in raw_sz.split(",")]; input_size = [w, hv]
        except ValueError: print(warn("  ⚠ Could not parse — will auto-detect."))

    print(f"\n{h('── 5. Postprocessing ──')}")
    conf = ask_float("Confidence threshold (0–1)", default=0.45)
    nms  = ask_float("NMS threshold (0–1)",        default=0.45)

    print(f"\n{h('── 6. Instance group ──')}")
    kind = ask("Instance kind", default=args.instance_kind, choices=VALID_INSTANCE_KINDS)

    print(f"\n{h('── 7. Adapter schema version ──')}")
    is_pose = args.task_type == "pose"
    if is_pose:
        schema = 3
        print(f"  Schema forced to: {ok('3')}  (pose rules are v3-only)")
        print(ok("  Available pose rule types: keypoint_zone_violation  handrail_contact  stair_descent_violation"))
        print(warn("  Draw zone polygons in the dashboard before testing — zone_id must match exactly."))
    else:
        print("  1 = flat alert_labels   2 = presence/absence rules   3 = zones/dwell/gathering")
        schema = int(args.schema) if args.schema else int(
            ask("Schema version", default="3", choices=["1", "2", "3"])
        )

    # ── Alert configuration (per schema version) ──────────────────────────────
    base_kwargs = dict(
        task_type=args.task_type,
        decoder=decoder, labels=labels, layout=layout, color_order=color_order,
        scale=scale, mean=mean, std=std, input_size=input_size,
        confidence_threshold=conf, nms_threshold=nms,
    )

    if schema == 1:
        alert_labels, alert_category, alert_severity = alert_wizard_v1(labels)
        adapter = build_adapter_v1(
            alert_labels=alert_labels, alert_category=alert_category,
            alert_severity=alert_severity, **base_kwargs,
        )
        summary = f"  Alert labels: {len(alert_labels)}  |  {alert_category}/{alert_severity}"
    elif schema == 2:
        alert_rules, default_alert, alert_labels = alert_wizard(labels)
        adapter = build_adapter(
            alert_rules=alert_rules, default_alert=default_alert,
            alert_labels=alert_labels, **base_kwargs,
        )
        n_r, n_d = len(alert_rules), len(alert_labels)
        summary = f"  Alert rules : {n_r} explicit" + (f", {n_d} via default_alert" if n_d else "")
    else:  # schema == 3
        alert_rules = alert_wizard_v3(labels, pose_mode=is_pose)
        # Pose rules all require the tracker; force it on so the user isn't asked
        needs_tracker = is_pose or any(r["type"] in V3_TRACKER_RULES for r in alert_rules)
        tracker = tracker_wizard(needs_tracker)
        adapter = build_adapter_v3(tracker=tracker, alert_rules=alert_rules, **base_kwargs)
        by_type: dict[str, int] = {}
        for r in alert_rules:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        types_str = ", ".join(f"{k}×{v}" for k, v in by_type.items()) or "none"
        summary = (f"  Alert rules : {len(alert_rules)} ({types_str})\n"
                   f"  Tracker     : {'ocsort' if tracker.get('enabled') else 'disabled'}")

    # ── Write ────────────────────────────────────────────────────────────────
    pbtxt = build_pbtxt(model_name, onnx_info, instance_kind=kind)
    cfg_path.write_text(pbtxt, encoding="utf-8")
    with adapter_path.open("w", encoding="utf-8") as fh:
        yaml.dump(adapter, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{h('═' * 55)}")
    print(f"{ok('✓ Done — ' + str(out_dir))}")
    print(f"{h('═' * 55)}")
    print(f"  {ok('config.pbtxt')}  → {cfg_path}")
    print(f"  {ok('adapter.yaml')} → {adapter_path}")
    print(f"  Model name  : {model_name}")
    print(f"  Schema      : v{schema}")
    print(f"  Decoder     : {decoder}  |  Labels: {len(labels)}")
    print(summary)
    print(f"\n{h('Next:')}")
    print("  Dashboard → AI Models → Add → upload model.onnx + config.pbtxt + adapter.yaml")
    if is_pose:
        print(warn("  Pose model checklist:"))
        print(warn("    1. Draw zone polygon(s) in the dashboard — zone_id must match what you entered above."))
        print(warn("    2. Assign this model to a camera with a live RTSP feed."))
        print(warn("    3. Confirm the model outputs 17 COCO keypoints per person (check Triton logs)."))
        print(warn("    4. Keypoint indices used: check adapter.yaml → alert_rules[*].keypoint_indices"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
