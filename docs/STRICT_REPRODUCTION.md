# Strict reproduction status

Этот файл фиксирует отдельный strict path: реализации, максимально близкие к
архитектурам, напечатанным в таблицах статьи.

Важно: strict path отделен от старых control/ablation runner-ов. Старые runner-ы
могут быть полезнее как инженерные модели, но они не являются буквальной
архитектурой из таблиц.

## Общая правка Hopf

Для strict режима исправлены две вещи:

- `beta` по умолчанию установлен в `-100.0`, как в параметрах Fig. 11
  (`mu=1, beta1=-100, beta2=0, I0=0.2`);
- forcing больше не вычисляет `arg(I)` через `atan2`, потому градиент
  `atan2(0,0)` дает NaN. Используется эквивалентная форма через `I_r`, `I_i`,
  `sin(theta)`, `cos(theta)`.

## Complex static layers

Добавлен `src/paper_donn.py`.

Static layers реализованы как complex dense:

```text
W = W_r + i W_i
Z_out = W Z_in + b
f(Z_out) = f(Re(Z_out)) + i f(Im(Z_out))
```

Это соответствует Eq. (20) из статьи ближе, чем старые real-only Dense слои.

## Table 1 strict

Runner:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/paper_classification_result.py --dataset-source supplement-notebook --epochs 20 --batch-size 32 --learning-rate 0.0001 --clipnorm 1.0 --hopf-input-scale 0.1 --beta -100 --metrics-path artifacts/plots/paper_exact/classification_paper_sequence_supplement_metrics_20e.json --out-path artifacts/plots/paper_exact/classification_paper_sequence_supplement_summary_20e.png
```

Architecture:

```text
Linear(20), Hopf(20), tanh(20), output(2)
```

Result:

- paper claim: `99% accuracy`;
- mean/sum accuracy: `0.605`;
- final-step accuracy: `0.56`;
- template-MSE accuracy: `0.555`;
- test MSE: `0.128399`.

Status: strict architecture implemented, paper score not reproduced.

## Table 2 strict

Runner:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/paper_sequence_result.py --task demodulation --num-samples 400 --epochs 60 --batch-size 32 --learning-rate 0.0001 --clipnorm 1.0 --dt 0.01 --duration 1.0 --hopf-input-scale 0.1 --beta -100 --metrics-path artifacts/plots/paper_exact/demodulation_paper_sequence_metrics_400x60.json --out-path artifacts/plots/paper_exact/demodulation_paper_sequence_summary_400x60.png
```

Architecture:

```text
ReLU(40), Hopf(40), ReLU(40), Hopf(40), tanh(40), output(1)
```

Result:

- paper claim: validation MSE `0.02`;
- local strict validation MSE: `3.509055`;
- local strict test MSE: `3.612422`;
- test correlation: `0.540015`.

Status: strict architecture implemented, paper score not reproduced.

## Table 3 strict smoke

Runners:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/paper_sequence_result.py --task integration --num-samples 80 --epochs 5 --batch-size 8 --learning-rate 0.0001 --clipnorm 1.0 --dt 0.001 --duration 1.0 --hopf-input-scale 0.1 --beta -100 --scale-data --metrics-path artifacts/plots/paper_exact/integration_paper_sequence_metrics_smoke.json --out-path artifacts/plots/paper_exact/integration_paper_sequence_summary_smoke.png

TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/paper_sequence_result.py --task differentiation --num-samples 80 --epochs 5 --batch-size 8 --learning-rate 0.0001 --clipnorm 1.0 --dt 0.001 --duration 1.0 --hopf-input-scale 0.1 --beta -100 --scale-data --metrics-path artifacts/plots/paper_exact/differentiation_paper_sequence_metrics_smoke.json --out-path artifacts/plots/paper_exact/differentiation_paper_sequence_summary_smoke.png
```

Architecture:

```text
ReLU(20), Hopf(20), ReLU(20), Hopf(20), tanh(20), output(1)
```

Smoke results:

- integration test MSE: `0.008609`;
- integration val MSE: `0.012465`;
- differentiation test MSE: `792.721252`;
- differentiation val MSE: `691.720413`.

Status: strict architecture implemented; integration smoke is numerically
stable, differentiation needs longer/tuned strict training and still does not
match the paper claim in the short run.

## Remaining blockers

The paper still does not specify enough details for a true byte-for-byte
reproduction:

- exact train/test splits and dataset sizes for Tables 1-3;
- exact optimizer settings for Tables 1-3;
- gradient clipping or stabilization details;
- exact initialization of complex weights;
- whether output uses real part only or another complex-to-real rule;
- Table 4 embedding parameter accounting;
- local UCF11/UCF50 dataset for Table 5.
