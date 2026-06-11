# Cognecto Utils

Tools for generating Triton Inference Server configuration files (`config.pbtxt`) and adapter files (`adapter.json` / `adapter.yaml`) from ONNX detection models.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+.

---

## Scripts

### `inspect_model.py` — Quick config generator (adapter v1)

Generates `config.pbtxt` and `adapter.json` from a detection ONNX model. Supports both fully non-interactive (CLI flags only) and guided interactive modes.

**Inspect a model without writing files:**

```bash
python3 inspect_model.py --model path/to/model.onnx --inspect-only
```

**Generate files with explicit flags (non-interactive):**

```bash
python3 inspect_model.py \
  --model path/to/model.onnx \
  --labels helmet,no_helmet \
  --alert-labels no_helmet \
  --alert-category ppe \
  --alert-severity high \
  --force
```

**Generate files with the interactive wizard:**

```bash
python3 inspect_model.py \
  --model path/to/model.onnx \
  --interactive \
  --force
```

The wizard prints detected model family, input/output shapes, and suggested decoder, then prompts for labels, decoder, alert settings, preprocessing params, and instance kind.

**Write outputs to a specific directory:**

```bash
python3 inspect_model.py \
  --model path/to/model.onnx \
  --labels helmet \
  --alert-labels all \
  --alert-category ppe \
  --output-dir /tmp/my_model_config \
  --force
```

By default, outputs are written next to the model file (or to the parent directory if the model sits in a numbered version folder like `model_name/1/model.onnx`).

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to model.onnx |
| `--name` | parent folder name | Triton model name |
| `--output-dir` | next to model | Where to write outputs |
| `--labels` | auto-detected | Comma-separated class names |
| `--labels-path` | auto-detected | Path to labels.txt |
| `--decoder` | `auto` | `auto`, `yolo_v8`, `yolo_single_class`, `ssd` |
| `--alert-labels` | none | Comma-separated subset or `all` |
| `--alert-category` | `detection` | e.g. `ppe`, `fire`, `intrusion` |
| `--alert-severity` | `medium` | `low`, `medium`, `high`, `critical` |
| `--layout` | `NCHW` | `NCHW` or `NHWC` (auto-inferred) |
| `--color-order` | `RGB` | `RGB` or `BGR` |
| `--scale` | `1/255` | Pixel scaling factor |
| `--mean` | `0,0,0` | Mean triplet |
| `--std` | `1,1,1` | Std triplet |
| `--confidence-threshold` | `0.45` | Detection confidence cutoff |
| `--nms-threshold` | `0.45` | NMS IoU threshold |
| `--instance-kind` | `KIND_CPU` | `KIND_CPU` or `KIND_GPU` |
| `--interactive` | off | Run the guided wizard |
| `--inspect-only` | off | Print details without writing files |
| `--force` | off | Overwrite existing files |

```bash
python3 inspect_model.py --help
```

---

### `generate_model_package.py` — Full wizard (adapter v2)

Interactive wizard that generates `config.pbtxt` + `adapter.yaml` (schema v2). Produces richer alert configurations including per-label rules with modes (`presence`, `absence`, `absence_unconditional`), guard labels, cooldowns, and custom messages.

**Always interactive — requires a terminal.**

```bash
python3 generate_model_package.py --model path/to/model.onnx
```

**Inspect without writing:**

```bash
python3 generate_model_package.py --model path/to/model.onnx --inspect-only
```

**Override output directory:**

```bash
python3 generate_model_package.py --model path/to/model.onnx --output-dir /tmp/out
```

The wizard walks through:
1. Model identity (Triton model name)
2. Labels (edit auto-detected labels)
3. Decoder selection
4. Preprocessing (layout, color order, scale, mean, std, input size)
5. Postprocessing thresholds
6. Instance group (CPU/GPU)
7. Alert rules — per-label rules or a shared default alert

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to model.onnx |
| `--name` | parent folder name | Triton model name |
| `--output-dir` | next to model | Where to write outputs |
| `--labels` | auto-detected | Comma-separated labels |
| `--labels-path` | auto-detected | Path to labels.txt |
| `--decoder` | `auto` | `auto`, `yolo_v8`, `yolo_single_class`, `ssd` |
| `--instance-kind` | `KIND_CPU` | `KIND_CPU` or `KIND_GPU` |
| `--inspect-only` | off | Print details without writing files |
| `--force` | off | Overwrite existing files |

---

## Output files

| File | Script | Description |
|---|---|---|
| `config.pbtxt` | both | Triton model configuration (inputs, outputs, instance group) |
| `adapter.json` | `inspect_model.py` | Adapter schema v1 — decoder, preprocessing, postprocess, alert metadata |
| `adapter.yaml` | `generate_model_package.py` | Adapter schema v2 — same as v1 plus per-label alert rules with modes and cooldowns |

Once generated, upload `model.onnx` + `config.pbtxt` + adapter file via the Cognecto dashboard under **AI Models → Add**.

---

## Which script should I use?

- Use **`inspect_model.py`** when you want a quick non-interactive run (CI, scripting, or already know all your parameters), or need `adapter.json` (v1 schema).
- Use **`generate_model_package.py`** when you need `adapter.yaml` (v2 schema) with rich per-label alert rules, or want the full guided setup wizard.

---

## Common examples

**Helmet detection model (Triton folder layout):**

```bash
python3 inspect_model.py \
  --model infrastructure/triton/model_archive/helmet-detection-1/1/model.onnx \
  --labels helmet,no_helmet \
  --alert-labels no_helmet \
  --alert-category ppe \
  --alert-severity high \
  --force
```

**Fire detection model with GPU instance:**

```bash
python3 inspect_model.py \
  --model models/fire/1/model.onnx \
  --labels fire,smoke \
  --alert-labels all \
  --alert-category fire \
  --alert-severity critical \
  --instance-kind KIND_GPU \
  --force
```

**Force a specific decoder:**

```bash
python3 inspect_model.py \
  --model /path/to/model.onnx \
  --labels person \
  --decoder ssd \
  --alert-labels all \
  --alert-category intrusion \
  --force
```
