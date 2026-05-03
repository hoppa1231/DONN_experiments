# Table 5: action recognition

Этот файл фиксирует состояние проверки Table 5 из статьи.

## Что написано в статье

Задача:

- UCF11 YouTube Action dataset;
- `50` кадров на видео;
- кадры resized до `48 x 48 x 3`;
- train/validation: `1290 / 305`;
- `dt = 0.02`;
- optimizer: Adam;
- learning rate: `0.0001`;
- loss: MSE.

Архитектура Table 5:

```text
2 x OCNN (3x3, 40), flatten, output (2)
initial frequency range: 1-15 Hz
input type to oscillator: I(t)
oscillator frequencies: trained
```

Заявленный результат для OCNN в таблице: `98.64%`.

Замечание: в тексте под Figure 4 указано `99.75%`, что совпадает со строкой
Convolutional flip-flops в Table 5, а не со строкой OCNN. Это еще одна
внутренняя нестыковка статьи.

## Что сделано локально

Добавлены файлы:

- `src/action_recognition.py`;
- `visual/action_recognition_result.py`.

Локальный runner проверяет доступность UCF/UCF50-подобных данных в
`/home/user/Projects/test-ai-capabilities/external`. Сейчас таких данных там не
найдено, поэтому строгий Table 5 запуск невозможен без скачивания датасета.

Чтобы не оставлять кодовую часть непроверенной, runner выполняет маленький
synthetic smoke-run:

- два класса движущегося квадрата;
- OCNN-style путь `2 x ConvOsc -> flatten -> output`;
- ramp-target и MSE, как в статье;
- результат сохраняется с явным флагом `is_ucf11_reproduction = false`.

Команда:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/action_recognition_result.py
```

## Текущий smoke-result

Файлы:

- `artifacts/plots/table5/fifth_work_ocnn_smoke_summary.png`;
- `artifacts/plots/table5/fifth_work_ocnn_smoke_metrics.json`.

Текущий результат:

- `val_acc = 0.3333`;
- `val_loss = 0.879913`;
- `train_acc = 0.4722`;
- `train_loss = 0.782569`;
- `total_params = 1394`.

Это не оценка качества статьи, а проверка, что OCNN-style computational path
работает в текущем окружении.

## Что нужно для строгого повтора

Нужен локальный UCF11/UCF50 dataset, подготовленный так же, как в статье:

- `50` кадров на пример;
- `48 x 48 x 3`;
- тот же train/validation split `1290 / 305`;
- точная разметка классов и target-ramp формат.

После этого smoke-run можно заменить настоящим Table 5 runner.
