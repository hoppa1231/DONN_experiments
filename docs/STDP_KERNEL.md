# Case study 2: STDP kernel

Этот файл фиксирует проверку второго case study из статьи.

## Что утверждает статья

Авторы рассматривают пару связанных Hopf-осцилляторов, получающих импульсы с
задержкой `tau`. Комплексный вес `W` обновляется по Hebbian-like правилу.

Уравнения из статьи:

```text
zdot1 = (mu + i omega1) z1 - |z1|^2 z1 + W z2 + p(t)
zdot2 = (mu + i omega2) z2 - |z2|^2 z2 + conjugate(W) z1 + p(t + tau)
Wdot  = -W + eta z1 conjugate(z2)
```

Качественное утверждение: действительная часть `W` отражает potentiation, а
мнимая часть `W` кодирует STDP-like kernel как функцию задержки `tau`.

## Что сделано локально

Добавлены файлы:

- `src/stdp_kernel.py`;
- `visual/stdp_kernel_result.py`.

Runner интегрирует уравнения Эйлером для набора задержек `tau` от `-10` до
`10`, как на Fig. 7. По умолчанию эти 20 единиц `tau` сопоставлены одному
периоду осцилляции.

Команда:

```bash
uv run --python 3.12 --with numpy --with matplotlib python visual/stdp_kernel_result.py
```

Диагностический literal-вариант:

```bash
uv run --python 3.12 --with numpy --with matplotlib python visual/stdp_kernel_result.py --hebbian-product literal-control --metrics-path artifacts/plots/case_study/case_study_stdp_kernel_literal_control_metrics.json --out-path artifacts/plots/case_study/case_study_stdp_kernel_literal_control_summary.png
```

## Результаты

Файлы:

- `artifacts/plots/case_study/case_study_stdp_kernel_summary.png`;
- `artifacts/plots/case_study/case_study_stdp_kernel_metrics.json`;
- `artifacts/plots/case_study/case_study_stdp_kernel_literal_control_summary.png`;
- `artifacts/plots/case_study/case_study_stdp_kernel_literal_control_metrics.json`.

Основной paper-conjugate вариант `z1 * conj(z2)`:

- `real_peak_to_peak = 0.032075`;
- `imag_peak_to_peak = 0.029975`;
- `imag_negative_delay_mean = -0.010214`;
- `imag_positive_delay_mean = 0.008860`.

Диагностический literal-control вариант `z1 * z2`:

- `real_peak_to_peak = 0.001808`;
- `imag_peak_to_peak = 0.002245`;
- `imag_negative_delay_mean = -0.000050`;
- `imag_positive_delay_mean = 0.000002`.

Оба варианта дают заметную зависимость комплексного веса от задержки.
Paper-conjugate вариант лучше воспроизводит знак STDP-like формы с
отрицательной областью при `tau < 0` и положительной при `tau > 0`, как на
присланном Fig. 7. При этом это не точное восстановление Fig. 7, потому что
статья не задает численные параметры импульса, начальных условий и
интегрирования.

После присланного фрагмента Fig. 7 исправлена шкала графика: теперь ось
подписана как `tau` и идет от `-10` до `10`, как в статье, а не в секундах.

## Вывод

На уровне уравнений механизм качественно воспроизводим: комплексный вес
становится функцией задержки между импульсами, а мнимая часть дает
delay-dependent kernel.

Но строгий pixel-level или numeric-level повтор Fig. 7 невозможен без
недостающих параметров эксперимента.
