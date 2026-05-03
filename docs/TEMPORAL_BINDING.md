# Case study 1: temporal binding

Этот файл фиксирует следующий этап проверки статьи после Table 4.

## Что утверждает статья

В Case study 1 авторы рассматривают четыре класса видео:

- движущаяся полоса одного цвета и одной ориентации;
- два цвета;
- две ориентации;
- всего четыре комбинации.

После обучения ConvOsc-модели они выбирают осцилляторы второго ConvOsc-слоя,
которые селективны к цвету и ориентации, затем сравнивают синхронность:

- внутри group oscillators;
- внутри residuary oscillators.

Ключевой качественный результат: group oscillators должны быть синхроннее, чем
residuary oscillators.

Figure 5 статьи задает такую архитектуру:

```text
Input video 100 x 32 x 32 x 3
-> ConvOsc 16 @ 3x3
-> ConvOsc 16 @ 3x3
-> ConvOsc 16 @ 3x3
-> Dense ReLU 64
-> Dense 4
```

В тексте сказано, что temporal-binding анализ выполняется по осцилляторам
второго ConvOsc-слоя. Все осцилляторы инициализируются в диапазоне `1-10 Hz`.

## Что сделано локально

Добавлены файлы:

- `src/temporal_binding.py`;
- `visual/temporal_binding_result.py`.

Текущий runner делает не полный обученный ConvOsc, а контроль двух вещей:

1. аудит сохраненного moving-bar датасета;
2. проверку set-selection и synchrony-алгоритма на детерминированном
   oscillatory probe.

Файлы результата:

- `artifacts/plots/case_study/case_study_temporal_binding_summary.png`;
- `artifacts/plots/case_study/case_study_temporal_binding_metrics.json`.

Команда:

```bash
uv run --python 3.12 --with numpy --with matplotlib python visual/temporal_binding_result.py
```

Можно также прогнать генератор напрямую:

```bash
uv run --python 3.12 --with numpy --with matplotlib python visual/temporal_binding_result.py --source generated-buggy --metrics-path artifacts/plots/case_study/case_study_temporal_binding_generated_buggy_metrics.json --out-path artifacts/plots/case_study/case_study_temporal_binding_generated_buggy_summary.png

uv run --python 3.12 --with numpy --with matplotlib python visual/temporal_binding_result.py --source generated-fixed --metrics-path artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_metrics.json --out-path artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_summary.png
```

## Найденная проблема в датасете

В notebook-генераторе `artifacts/case_study/temporal_binding_dataset.ipynb`
классы кодируются как `class = 2 * orientation + color`, поэтому сохраненный
порядок классов читается как `AX, BX, AY, BY`, а не как `AX, AY, BX, BY`.

В том же генераторе ветка обработки wrap-around выглядит ошибочно:

```python
if bar_on > bar_off:
    swap = bar_on
    bar_off = bar_on
    bar_on = swap
```

После такой операции `bar_on == bar_off`, поэтому полоса на этих кадрах не
рисуется. В сохраненном датасете это заметно:

- `blank_frame_fraction = 0.16644`;
- `samples_with_blank_frames = 98 / 100`;
- `max_blank_frames_in_sample = 188`.

То есть примерно 16.6% кадров полностью пустые. Это важно учитывать перед
полным повторением обученной ConvOsc-модели.

Контроль генератора подтвердил, что это именно ошибка wrap-around:

| source | blank_frame_fraction | samples_with_blank_frames | max_blank_frames_in_sample |
|---|---:|---:|---:|
| saved | `0.16644` | `98 / 100` | `188` |
| generated-buggy | `0.16188` | `99 / 100` | `157` |
| generated-fixed | `0.0` | `0 / 100` | `0` |

## Текущий контрольный результат

Локальный control-run показывает ожидаемое направление:

- group synchrony выше residuary synchrony для всех четырех классов;
- на сохраненном датасете средний зазор `group - residuary = 0.034598`;
- на исправленной генерации средний зазор `group - residuary = 0.035809`.

Дополнительно добавлен deterministic classifier audit. Он напрямую считывает
два фактора генератора: активный цветовой канал и ориентацию полосы. Результат:

| source | deterministic classifier accuracy |
|---|---:|
| saved | `1.0` |
| generated-buggy | `1.0` |
| generated-fixed | `1.0` |

Это означает, что сама four-class moving-bar классификация в текущей постановке
очень простая. Поэтому главное проверяемое утверждение Case study 1 не
“можно ли классифицировать видео”, а возникает ли заявленная структура
синхронности именно в скрытом ConvOsc-слое.

Но это пока не доказательство статьи, потому что скрытый слой не получен из
обученной ConvOsc-модели.

## Что осталось для строгой проверки

Следующий строгий этап:

1. восстановить или реализовать ConvOsc-модель для moving-bar задачи;
2. обучить ее на исправленном и/или исходном датасете;
3. извлечь комплексные активности второго ConvOsc-слоя;
4. повторить selection/synchrony алгоритм уже на настоящих hidden activations;
5. сравнить размеры множеств с Table 6 статьи:
   `A_hat = 545`, `X_hat = 435`, `B_hat = 38`, `Y_hat = 48`.
