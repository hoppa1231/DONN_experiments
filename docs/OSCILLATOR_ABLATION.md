# Oscillator ablation

Этот файл отвечает на отдельный вопрос: дают ли Hopf/oscillator layers
практический выигрыш в текущих локальных воспроизведениях.

Команда:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/oscillator_ablation_result.py
```

Артефакты:

- `artifacts/plots/ablation/oscillator_vs_baselines_summary.png`
- `artifacts/plots/ablation/oscillator_vs_baselines_metrics.json`

## Сводка

| Task | oscillator result | baseline result | local winner |
| --- | ---: | ---: | --- |
| Table 1 classification | CE `0.58`; ramp/MSE mean `0.38-0.47`; best final-step `0.61` accuracy | `0.995-1.0` FFT accuracy | baseline |
| Table 2 demodulation | `0.0567` test MSE | `0.2677` matched no-Hopf MSE; `0.1897` coherent-demod MSE | oscillator |
| Table 3 operators | `0.0040` integration MSE; `19.6581` differentiation MSE | matched no-Hopf `0.0051` / `37.6981`; numeric `7.5e-12` / `2.56e-4` | numeric baseline |
| Table 4 sentiment | `0.5254` DONN 1k/1e; `0.5244` DONN 2k/2e accuracy | `0.5762` BiLSTM 1k/1e accuracy | baseline |

## Interpretation

На текущем локальном наборе проверок нет широкого эмпирического преимущества
осцилляторных нейронов. Простые неосцилляторные baseline выигрывают Table 1,
Table 3 и Table 4.

Единственный локальный плюс Hopf сейчас виден в Table 2: демодуляция. Там DONN
с Hopf-слоем и temporal readout дает `test_mse = 0.0567`. Matched no-Hopf
вариант с тем же Dense frontend и temporal readout дает `test_mse = 0.2677`, а
известно-несущий coherent demodulation baseline с moving-average low-pass дает
`test_mse = 0.1897`.

Для Table 1 теперь сохраняются несколько метрик ramp-output: mean/sum,
final-step и ближайший ramp-template по MSE. Метрика действительно влияет:
на supplement notebook варианте mean accuracy `0.47`, а final-step accuracy
`0.61`. Но ни одна из проверенных метрик не приближает результат к заявленным
`99%`.

## Ограничения вывода

Это не финальный теоретический приговор осцилляторным сетям вообще. Это вывод
по текущей реализации, текущим данным и тем формулам, которые удалось
восстановить из статьи.

Для более строгого ответа нужны:

- авторский forward pass для Table 1 и Table 4;
- настоящий UCF11/UCF50 dataset для Table 5;
- несколько вариантов matched-capacity baselines для Table 2 и Table 3;
- несколько seed-run вместо одного seed.
