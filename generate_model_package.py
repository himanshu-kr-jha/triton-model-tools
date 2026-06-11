#!/usr/bin/env python3
"""
Generate Triton config.pbtxt + adapter.yaml (v2) for a detection ONNX model.
Files are saved in the SAME folder as the model.onnx by default.

Usage:
    python generate_model_package.py --model fire_model/model.onnx
    python generate_model_package.py --model helmet/best_trained.onnx
    python generate_model_package.py --model fire_model/model.onnx --inspect-only
    python generate_model_package.py --model fire_model/model.onnx --force
"""
from __future__ import annotations

import argparse
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


def ask_bool(prompt: str, default: bool = True) -> bool:
    raw = input(f"  {prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not raw else raw in ("y", "yes", "1", "true")


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
    message = ask("message  ({camera} {stream_key} {label} {model})", default=default_msg)

    rule: dict = {"rule_id": rule_id, "label": label, "mode": mode}
    if guard:             rule["guard_label"]           = guard
    if conf is not None:  rule["confidence_threshold"]  = conf
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
        default_alert = {
            "mode":             ask("default mode",     default="presence", choices=VALID_MODES),
            "category":         ask("default category", default="safety"),
            "severity":         ask("default severity", default="medium",   choices=VALID_SEVERITIES),
            "cooldown_seconds": int(ask_float("default cooldown_seconds", default=30)),
        }

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


# ── Build adapter dict ────────────────────────────────────────────────────────

def build_adapter(
    *, decoder, labels, layout, color_order, scale, mean, std,
    input_size, confidence_threshold, nms_threshold,
    alert_rules, default_alert, alert_labels,
) -> dict:
    pre: dict = {"layout": layout, "color_order": color_order,
                 "scale": scale, "mean": mean, "std": std}
    if input_size: pre["input_size"] = input_size
    a: dict = {
        "schema_version": 2, "task_type": "detection", "decoder": decoder,
        "preprocess": pre,
        "postprocess": {"confidence_threshold": confidence_threshold,
                        "nms_threshold": nms_threshold},
        "labels": labels,
    }
    if default_alert: a["default_alert"] = default_alert
    if alert_labels:  a["alert_labels"]  = alert_labels
    if alert_rules:   a["alert_rules"]   = alert_rules
    return a


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate config.pbtxt + adapter.yaml for a detection ONNX model.\n"
            "Files are written into the SAME folder as model.onnx by default."
        )
    )
    p.add_argument("--model",         type=Path, required=True)
    p.add_argument("--name",          help="Triton model name. Default: parent folder name.")
    p.add_argument("--output-dir",    type=Path, help="Override output directory.")
    p.add_argument("--labels-path",   type=Path)
    p.add_argument("--labels",        help="Comma-separated labels (overrides auto-detection)")
    p.add_argument("--decoder",       default="auto",
                   choices=["auto", "yolo_v8", "yolo_single_class", "ssd"])
    p.add_argument("--instance-kind", default="KIND_CPU", choices=VALID_INSTANCE_KINDS)
    p.add_argument("--inspect-only",  action="store_true")
    p.add_argument("--force",         action="store_true", help="Overwrite existing files.")
    args = p.parse_args()

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

    alert_rules, default_alert, alert_labels = alert_wizard(labels)

    # ── Write ────────────────────────────────────────────────────────────────
    pbtxt   = build_pbtxt(model_name, onnx_info, instance_kind=kind)
    adapter = build_adapter(
        decoder=decoder, labels=labels, layout=layout, color_order=color_order,
        scale=scale, mean=mean, std=std, input_size=input_size,
        confidence_threshold=conf, nms_threshold=nms,
        alert_rules=alert_rules, default_alert=default_alert, alert_labels=alert_labels,
    )
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
    print(f"  Decoder     : {decoder}  |  Labels: {len(labels)}")
    n_r, n_d = len(alert_rules), len(alert_labels)
    print(f"  Alert rules : {n_r} explicit" + (f", {n_d} via default_alert" if n_d else ""))
    print(f"\n{h('Next:')}")
    print("  Dashboard → AI Models → Add → upload model.onnx + config.pbtxt + adapter.yaml")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
