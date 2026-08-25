# WeHMiT-Bench

WeHMiT-Bench evaluates health misinformation detection in Chinese WeChat article titles under a title-only setting.

## Files

- `WeHMiT-Bench.xlsx`: verified annotation workbook.
- `WeHMiT-Bench_Core.csv`: 460 labeled health titles with fixed data splits.
- `WeHMiT-Bench_OOD.csv`: 63 labeled titles outside the health domain.
- `prepare_benchmark_data.py`: data preparation.
- `llm_title_detection.py`: LLM title detection.
- `machine_learning_title_detection.py`: machine learning title detection.
- `model.example.json`: API configuration example without keys.

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy `model.example.json` to `model.json` and provide API keys through the environment variables specified by `api_key_env`. The real `model.json` is excluded by `.gitignore`.

## Usage

```bash
python llm_title_detection.py --core-file WeHMiT-Bench_Core.csv --ood-file WeHMiT-Bench_OOD.csv
python prepare_benchmark_data.py
python machine_learning_title_detection.py
```

Generated outputs are written to `experiment_outputs/` and excluded by `.gitignore`.
