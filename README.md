# WeHMiT-Bench

WeHMiT-Bench evaluates health misinformation detection in Chinese WeChat article titles under a title-only setting.

## Files

- `WeHMiT-Bench_Core.csv`: 460 labeled health titles.
- `WeHMiT-Bench_OOD.csv`: 63 labeled titles outside the health domain.
- `llm_title_detection.py`: LLM title detection.
- `machine_learning_title_detection.py`: machine learning title detection.
- `model.example.json`: API configuration example without keys.

Both benchmark files contain only `title` and the expert-verified final `label` (`Yes` or `No`). Internal annotation and review records are not included.

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy `model.example.json` to `model.json` and provide API keys through the environment variables specified by `api_key_env`. The real `model.json` is excluded by `.gitignore`.

## Usage

```bash
python llm_title_detection.py
python machine_learning_title_detection.py
```

To preserve the fixed partitions without adding a third field, the Core file is ordered as 319 training titles, 69 development titles, and 72 test titles. The machine learning script reads these ranges directly.

Generated outputs are written to `experiment_outputs/` and excluded by `.gitignore`.
