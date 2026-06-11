#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import onnx


TRITON_DTYPE_BY_ONNX = {
    onnx.TensorProto.FLOAT: "TYPE_FP32",
    onnx.TensorProto.UINT8: "TYPE_UINT8",
    onnx.TensorProto.INT8: "TYPE_INT8",
    onnx.TensorProto.UINT16: "TYPE_UINT16",
    onnx.TensorProto.INT16: "TYPE_INT16",
    onnx.TensorProto.INT32: "TYPE_INT32",
    onnx.TensorProto.INT64: "TYPE_INT64",
    onnx.TensorProto.BOOL: "TYPE_BOOL",
    onnx.TensorProto.FLOAT16: "TYPE_FP16",
    onnx.TensorProto.DOUBLE: "TYPE_FP64",
}

VALID_DECODERS = {"auto", "yolo_v8", "yolo_single_class", "ssd"}
VALID_LAYOUTS = {"NCHW", "NHWC"}
VALID_COLOR_ORDERS = {"RGB", "BGR"}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_INSTANCE_KINDS = {"KIND_CPU", "KIND_GPU"}
DETECTABLE_CHANNEL_COUNTS = {1, 3, 4}


def _parse_onnx_dims(value_info) -> list[int | str]:
    dims: list[int | str] = []
    tensor_shape = value_info.type.tensor_type.shape.dim
    for dim in tensor_shape:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def load_onnx_io(model: onnx.ModelProto) -> dict[str, list[dict]]:
    return {
        "inputs": [
            {
                "name": value.name,
                "dtype": TRITON_DTYPE_BY_ONNX.get(value.type.tensor_type.elem_type, "TYPE_FP32"),
                "dims": _parse_onnx_dims(value),
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "dtype": TRITON_DTYPE_BY_ONNX.get(value.type.tensor_type.elem_type, "TYPE_FP32"),
                "dims": _parse_onnx_dims(value),
            }
            for value in model.graph.output
        ],
    }


def read_onnx_model(model_path: Path) -> onnx.ModelProto:
    return onnx.load(str(model_path))


def extract_metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _normalize_labels_object(value: object) -> list[str]:
    if isinstance(value, dict):
        ordered_items = sorted(value.items(), key=lambda item: str(item[0]))
        return [str(label).strip() for _, label in ordered_items if str(label).strip()]
    if isinstance(value, (list, tuple)):
        return [str(label).strip() for label in value if str(label).strip()]
    if isinstance(value, str):
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        if "\n" in value:
            return [part.strip() for part in value.splitlines() if part.strip()]
        if value.strip():
            return [value.strip()]
    return []


def parse_metadata_labels(metadata: dict[str, str]) -> list[str]:
    for key in ("names", "labels", "classes"):
        raw = metadata.get(key)
        if not raw:
            continue
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = raw
        labels = _normalize_labels_object(parsed)
        if labels:
            return labels
    return []


def infer_layout(onnx_info: dict[str, list[dict]]) -> str:
    if not onnx_info["inputs"]:
        return "NCHW"

    dims = onnx_info["inputs"][0]["dims"]
    if len(dims) >= 4:
        if isinstance(dims[1], int) and dims[1] in DETECTABLE_CHANNEL_COUNTS:
            return "NCHW"
        if isinstance(dims[-1], int) and dims[-1] in DETECTABLE_CHANNEL_COUNTS:
            return "NHWC"
    return "NCHW"


def detect_model_family(
    model: onnx.ModelProto,
    metadata: dict[str, str],
    onnx_info: dict[str, list[dict]],
    labels: list[str],
) -> str:
    producer = model.producer_name.lower()
    metadata_blob = " ".join(f"{key}={value}" for key, value in metadata.items()).lower()
    output_dims = onnx_info["outputs"][0]["dims"] if onnx_info["outputs"] else []
    label_count = len(labels)

    if "ultralytics" in metadata_blob or "ultralytics" in producer:
        return "Ultralytics/YOLO export"
    if "yolo" in metadata_blob or "yolo" in producer:
        return "YOLO-style export"
    if len(output_dims) == 3 and any(
        isinstance(dim, int) and dim in {label_count + 4, label_count + 5} for dim in output_dims[1:]
    ):
        return "YOLO-style dense detector"
    if len(output_dims) == 3 and output_dims[-1] == 6:
        return "Detection model with decoded boxes output"
    if len(output_dims) == 3:
        return "Detection model with dense predictions output"
    return "Generic ONNX detection model"


def quote_dim(dim: int | str) -> str:
    return str(dim) if isinstance(dim, int) else f'"{dim}"'


def format_dims(dims: list[int | str]) -> str:
    return ", ".join(quote_dim(dim) for dim in dims)


def triton_dim(dim: int | str) -> int:
    return dim if isinstance(dim, int) else -1


def triton_dims(dims: list[int | str]) -> list[int]:
    return [triton_dim(dim) for dim in dims]


def build_pbtxt(
    model_name: str,
    onnx_info: dict[str, list[dict]],
    platform: str = "onnxruntime_onnx",
    instance_kind: str = "KIND_CPU",
) -> str:
    input_blocks = []
    for item in onnx_info["inputs"]:
        input_blocks.append(
            "\n".join(
                [
                    "  {",
                    f'    name: "{item["name"]}"',
                    f'    data_type: {item["dtype"]}',
                    f"    dims: [ {format_dims(triton_dims(item['dims']))} ]",
                    "  }",
                ]
            )
        )

    output_blocks = []
    for item in onnx_info["outputs"]:
        output_blocks.append(
            "\n".join(
                [
                    "  {",
                    f'    name: "{item["name"]}"',
                    f'    data_type: {item["dtype"]}',
                    f"    dims: [ {format_dims(triton_dims(item['dims']))} ]",
                    "  }",
                ]
            )
        )

    sections = [
        f'name: "{model_name}"',
        f'platform: "{platform}"',
        "max_batch_size: 0",
        "",
        "input [",
        ",\n".join(input_blocks),
        "]",
        "",
        "output [",
        ",\n".join(output_blocks),
        "]",
        "",
        "instance_group [",
        "  {",
        f"    kind: {instance_kind}",
        "    count: 1",
        "  }",
        "]",
        "",
    ]
    return "\n".join(sections)


def load_labels(
    labels_path: Path | None,
    labels_csv: str | None,
    decoder: str,
    metadata_labels: list[str] | None = None,
) -> tuple[list[str], str]:
    if labels_csv:
        labels = [label.strip() for label in labels_csv.split(",") if label.strip()]
        if labels:
            return labels, "cli"

    if labels_path and labels_path.exists():
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels:
            return labels, "labels.txt"

    if metadata_labels:
        return metadata_labels, "onnx_metadata"

    if decoder == "yolo_single_class":
        return ["object"], "default"
    return ["class_0"], "default"


def infer_decoder(
    onnx_info: dict[str, list[dict]],
    labels: list[str],
    decoder: str,
    metadata: dict[str, str] | None = None,
) -> str:
    if decoder != "auto":
        return decoder

    if not onnx_info["outputs"]:
        return "ssd"

    dims = onnx_info["outputs"][0]["dims"]
    label_count = len(labels)
    metadata = metadata or {}

    if len(dims) == 3 and dims[-1] == 6:
        return "ssd"

    if len(dims) == 3:
        if dims[1] == 5 or dims[2] == 5 or dims[-1] == 5:
            return "yolo_single_class"
        for dim in dims[1:]:
            if isinstance(dim, int) and dim in {label_count + 4, label_count + 5}:
                return "yolo_v8" if label_count > 1 else "yolo_single_class"
        if isinstance(dims[1], int) and isinstance(dims[2], int) and max(dims[1], dims[2]) > 20:
            task = metadata.get("task", "").lower()
            if task == "detect":
                return "yolo_v8" if label_count > 1 else "yolo_single_class"
            return "yolo_v8" if label_count > 1 else "yolo_single_class"

    return "ssd"


def normalize_alert_labels(raw_alert_labels: str | None, labels: list[str]) -> list[str]:
    if not raw_alert_labels:
        return []
    if raw_alert_labels.strip().lower() == "all":
        return list(labels)
    selected = [label.strip() for label in raw_alert_labels.split(",") if label.strip()]
    unknown = sorted(set(selected) - set(labels))
    if unknown:
        raise ValueError(f"alert_labels must be a subset of labels; unknown={unknown}")
    return selected


def build_adapter(
    *,
    decoder: str,
    labels: list[str],
    alert_labels: list[str],
    layout: str,
    color_order: str,
    scale: float,
    mean: list[float],
    std: list[float],
    confidence_threshold: float,
    nms_threshold: float,
    alert_category: str,
    alert_severity: str,
) -> dict:
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(VALID_LAYOUTS)}")
    if color_order not in VALID_COLOR_ORDERS:
        raise ValueError(f"color_order must be one of {sorted(VALID_COLOR_ORDERS)}")
    if decoder not in VALID_DECODERS - {"auto"}:
        raise ValueError(f"decoder must be one of {sorted(VALID_DECODERS - {'auto'})}")
    if alert_severity not in VALID_SEVERITIES:
        raise ValueError(f"alert_severity must be one of {sorted(VALID_SEVERITIES)}")

    return {
        "schema_version": 1,
        "task_type": "detection",
        "decoder": decoder,
        "preprocess": {
            "layout": layout,
            "color_order": color_order,
            "scale": scale,
            "mean": mean,
            "std": std,
        },
        "postprocess": {
            "confidence_threshold": confidence_threshold,
            "nms_threshold": nms_threshold,
        },
        "labels": labels,
        "alert_labels": alert_labels,
        "alert_category": alert_category,
        "alert_severity": alert_severity,
    }


def resolve_output_dir(model_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir

    if model_path.parent.name.isdigit():
        return model_path.parent.parent
    return model_path.parent


def detect_labels_path(model_path: Path, explicit_labels_path: Path | None) -> Path | None:
    if explicit_labels_path is not None:
        return explicit_labels_path

    candidates = []
    if model_path.parent.name.isdigit():
        candidates.append(model_path.parent.parent / "labels.txt")
    candidates.append(model_path.parent / "labels.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_triplet(raw: str, field_name: str) -> list[float]:
    values = [part.strip() for part in raw.split(",")]
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 comma-separated numbers")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain only numbers") from exc


def format_label_source(label_source: str, labels_path: Path | None) -> str:
    if label_source == "cli":
        return "CLI --labels"
    if label_source == "labels.txt":
        return f"labels.txt ({labels_path})" if labels_path else "labels.txt"
    if label_source == "onnx_metadata":
        return "ONNX metadata"
    return "fallback default"


def prompt_text(message: str, default: str) -> str:
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        value = raw or default
        if value:
            return value


def prompt_choice(message: str, default: str, choices: list[str]) -> str:
    choices_str = ", ".join(choices)
    while True:
        raw = input(f"{message} [{default}] ({choices_str}): ").strip()
        value = raw or default
        if value in choices:
            return value
        print(f"Please choose one of: {choices_str}")


def prompt_float(message: str, default: float) -> float:
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def prompt_triplet(message: str, default: str, field_name: str) -> list[float]:
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        try:
            return parse_triplet(raw or default, field_name)
        except ValueError as exc:
            print(exc)


def prompt_labels(default_labels: list[str]) -> list[str]:
    default = ",".join(default_labels)
    while True:
        raw = input(f"Labels [{default}]: ").strip()
        labels = [label.strip() for label in (raw or default).split(",") if label.strip()]
        if labels:
            return labels
        print("Please provide at least one label.")


def prompt_alert_labels(labels: list[str], default_alert_labels: list[str]) -> list[str]:
    default = ",".join(default_alert_labels) if default_alert_labels else "none"
    print(f"Available labels for alerts: {', '.join(labels)}")
    while True:
        raw = input(f"Alert labels [{default}] (comma-separated, 'all', or 'none'): ").strip()
        value = raw or default
        if value.lower() == "none":
            return []
        try:
            return normalize_alert_labels(value, labels)
        except ValueError as exc:
            print(exc)


def run_interactive_wizard(
    *,
    labels: list[str],
    decoder: str,
    alert_labels: list[str],
    layout: str,
    color_order: str,
    scale: float,
    mean_raw: str,
    std_raw: str,
    confidence_threshold: float,
    nms_threshold: float,
    alert_category: str,
    alert_severity: str,
    instance_kind: str,
) -> dict[str, object]:
    print("\nInteractive adapter setup")
    print("Press Enter to accept the suggested value.\n")

    chosen_labels = prompt_labels(labels)
    chosen_decoder = prompt_choice("Decoder", decoder, sorted(VALID_DECODERS - {"auto"}))
    if chosen_decoder == "yolo_single_class" and len(chosen_labels) != 1:
        print("yolo_single_class requires exactly one label. Keeping only the first label.")
        chosen_labels = chosen_labels[:1]

    chosen_alert_labels = prompt_alert_labels(chosen_labels, alert_labels)
    chosen_alert_category = prompt_text("Alert category", alert_category)
    chosen_alert_severity = prompt_choice("Alert severity", alert_severity, sorted(VALID_SEVERITIES))
    chosen_layout = prompt_choice("Layout", layout, sorted(VALID_LAYOUTS))
    chosen_color_order = prompt_choice("Color order", color_order, sorted(VALID_COLOR_ORDERS))
    chosen_scale = prompt_float("Scale", scale)
    chosen_mean = prompt_triplet("Mean triplet", mean_raw, "mean")
    chosen_std = prompt_triplet("Std triplet", std_raw, "std")
    chosen_confidence = prompt_float("Confidence threshold", confidence_threshold)
    chosen_nms = prompt_float("NMS threshold", nms_threshold)
    chosen_instance_kind = prompt_choice("Instance kind", instance_kind, sorted(VALID_INSTANCE_KINDS))

    return {
        "labels": chosen_labels,
        "decoder": chosen_decoder,
        "alert_labels": chosen_alert_labels,
        "alert_category": chosen_alert_category,
        "alert_severity": chosen_alert_severity,
        "layout": chosen_layout,
        "color_order": chosen_color_order,
        "scale": chosen_scale,
        "mean": chosen_mean,
        "std": chosen_std,
        "confidence_threshold": chosen_confidence,
        "nms_threshold": chosen_nms,
        "instance_kind": chosen_instance_kind,
    }


def print_inspection_summary(
    *,
    model_path: Path,
    model: onnx.ModelProto,
    onnx_info: dict[str, list[dict]],
    metadata: dict[str, str],
    model_family: str,
    labels: list[str],
    label_source: str,
    labels_path: Path | None,
    decoder: str,
    layout: str,
) -> None:
    print("Inspection summary")
    print(f"Model:            {model_path}")
    print(f"Producer:         {model.producer_name or 'unknown'} {model.producer_version or ''}".rstrip())
    print(f"Detected family:  {model_family}")
    print(f"Suggested layout: {layout}")
    print(f"Suggested decoder: {decoder}")
    print(f"Labels:           {labels}")
    print(f"Labels source:    {format_label_source(label_source, labels_path)}")
    for item in onnx_info["inputs"]:
        print(f"Input:            {item['name']} {item['dims']} {item['dtype']}")
    for item in onnx_info["outputs"]:
        print(f"Output:           {item['name']} {item['dims']} {item['dtype']}")
    interesting_keys = ("task", "names", "imgsz", "stride", "author", "description")
    available_keys = [key for key in interesting_keys if key in metadata]
    if available_keys:
        print("Metadata:")
        for key in available_keys:
            print(f"  {key}: {metadata[key]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Triton config.pbtxt and adapter.json for a detection ONNX model."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model.onnx")
    parser.add_argument("--name", help="Triton model name. Defaults to the model directory name.")
    parser.add_argument("--output-dir", type=Path, help="Directory to write config.pbtxt and adapter.json")
    parser.add_argument("--labels-path", type=Path, help="Optional labels.txt path")
    parser.add_argument("--labels", help="Comma-separated labels. Overrides labels.txt")
    parser.add_argument(
        "--decoder",
        default="auto",
        choices=sorted(VALID_DECODERS),
        help="Decoder family. Defaults to auto inference from output shape.",
    )
    parser.add_argument("--alert-labels", help="Comma-separated alert labels, or 'all'. Defaults to none.")
    parser.add_argument("--alert-category", default="detection", help="Alert category for generated adapter.json")
    parser.add_argument(
        "--alert-severity",
        default="medium",
        choices=sorted(VALID_SEVERITIES),
        help="Alert severity for generated adapter.json",
    )
    parser.add_argument("--layout", default="NCHW", choices=sorted(VALID_LAYOUTS))
    parser.add_argument("--color-order", default="RGB", choices=sorted(VALID_COLOR_ORDERS))
    parser.add_argument("--scale", type=float, default=1.0 / 255.0, help="Pixel scaling factor")
    parser.add_argument("--mean", default="0,0,0", help="Mean triplet, e.g. 0,0,0")
    parser.add_argument("--std", default="1,1,1", help="Std triplet, e.g. 1,1,1")
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument(
        "--instance-kind",
        default="KIND_CPU",
        choices=sorted(VALID_INSTANCE_KINDS),
        help="Triton instance group kind. Use KIND_GPU only when GPUs are available to Triton.",
    )
    parser.add_argument("--interactive", action="store_true", help="Inspect the model and prompt for adapter choices.")
    parser.add_argument("--inspect-only", action="store_true", help="Print detected model details and exit.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing config.pbtxt/adapter.json")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise SystemExit("Only .onnx model files are supported")

    model_name = args.name or (model_path.parent.parent.name if model_path.parent.name.isdigit() else model_path.parent.name)
    output_dir = resolve_output_dir(model_path, args.output_dir.resolve() if args.output_dir else None)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.pbtxt"
    adapter_path = output_dir / "adapter.json"
    if not args.force:
        existing = [str(path) for path in (config_path, adapter_path) if path.exists()]
        if existing:
            raise SystemExit(f"Refusing to overwrite existing files: {', '.join(existing)}. Use --force to replace them.")

    model = read_onnx_model(model_path)
    metadata = extract_metadata(model)
    onnx_info = load_onnx_io(model)
    metadata_labels = parse_metadata_labels(metadata)
    labels_path = detect_labels_path(model_path, args.labels_path.resolve() if args.labels_path else None)
    labels, label_source = load_labels(labels_path, args.labels, args.decoder, metadata_labels=metadata_labels)
    decoder = infer_decoder(onnx_info, labels, args.decoder, metadata=metadata)
    layout = infer_layout(onnx_info) if args.layout == "NCHW" else args.layout
    model_family = detect_model_family(model, metadata, onnx_info, labels)

    print_inspection_summary(
        model_path=model_path,
        model=model,
        onnx_info=onnx_info,
        metadata=metadata,
        model_family=model_family,
        labels=labels,
        label_source=label_source,
        labels_path=labels_path,
        decoder=decoder,
        layout=layout,
    )

    if args.inspect_only:
        return 0

    mean = parse_triplet(args.mean, "mean")
    std = parse_triplet(args.std, "std")

    if args.interactive:
        if not sys.stdin.isatty():
            raise SystemExit("--interactive requires a TTY")
        chosen = run_interactive_wizard(
            labels=labels,
            decoder=decoder,
            alert_labels=normalize_alert_labels(args.alert_labels, labels) if args.alert_labels else [],
            layout=layout,
            color_order=args.color_order,
            scale=args.scale,
            mean_raw=args.mean,
            std_raw=args.std,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            alert_category=args.alert_category,
            alert_severity=args.alert_severity,
            instance_kind=args.instance_kind,
        )
        labels = chosen["labels"]
        decoder = str(chosen["decoder"])
        alert_labels = chosen["alert_labels"]
        layout = str(chosen["layout"])
        mean = chosen["mean"]
        std = chosen["std"]
        color_order = str(chosen["color_order"])
        scale = float(chosen["scale"])
        confidence_threshold = float(chosen["confidence_threshold"])
        nms_threshold = float(chosen["nms_threshold"])
        alert_category = str(chosen["alert_category"])
        alert_severity = str(chosen["alert_severity"])
        instance_kind = str(chosen["instance_kind"])
    else:
        alert_labels = normalize_alert_labels(args.alert_labels, labels)
        color_order = args.color_order
        scale = args.scale
        confidence_threshold = args.confidence_threshold
        nms_threshold = args.nms_threshold
        alert_category = args.alert_category
        alert_severity = args.alert_severity
        instance_kind = args.instance_kind

    if decoder == "yolo_single_class" and len(labels) != 1:
        raise SystemExit("yolo_single_class decoder requires exactly one label")

    pbtxt = build_pbtxt(model_name, onnx_info, instance_kind=instance_kind)
    adapter = build_adapter(
        decoder=decoder,
        labels=labels,
        alert_labels=alert_labels,
        layout=layout,
        color_order=color_order,
        scale=scale,
        mean=mean,
        std=std,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        alert_category=alert_category,
        alert_severity=alert_severity,
    )

    config_path.write_text(pbtxt, encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter, indent=2) + "\n", encoding="utf-8")

    print(f"Model:       {model_path}")
    print(f"Output dir:  {output_dir}")
    print(f"config.pbtxt: {config_path}")
    print(f"adapter.json: {adapter_path}")
    print(f"Decoder:     {decoder}")
    print(f"Labels:      {labels}")
    print(f"Alert labels:{alert_labels}")
    print(f"Layout:      {layout}")
    print(f"Instance:    {instance_kind}")
    if labels_path:
        print(f"labels.txt:  {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
