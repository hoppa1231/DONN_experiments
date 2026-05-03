# Formula audit

Этот файл фиксирует сверку формул статьи с тем, что сейчас используется в коде.

## HopfLayer

Формула статьи для режима `I(t)` после перехода к полярным координатам:

```text
r_dot   = mu*r + beta*r^3 + eps*beta2*r^5/(1 - eps*r^2) + A(t)*cos(psi)
psi_dot = Omega - A(t)/r * sin(psi)
```

Далее статья говорит, что для используемых режимов ограничивается
`beta2 = 0`. Поэтому рабочая формула:

```text
r_dot = mu*r + beta*r^3 + A(t)*cos(psi)
```

В `src/HopfLayer.py` теперь реализовано:

```text
I(t) = input_scale * (x_r + i*x_i)
A(t) = |I(t)|
psi = theta - arg(I(t))
r_dot = mu*r + beta*r^3 + eps*beta2*r^5/(1 - eps*r^2) + A(t)*cos(psi)
theta_dot = omega - A(t)/r * sin(psi)
```

В коде `arg(I(t))` не вычисляется через `atan2`, потому градиент в точке
`I(t)=0` неопределен. Используется алгебраически эквивалентная форма:

```text
A*cos(theta - arg(I)) = I_r*cos(theta) + I_i*sin(theta)
A*sin(theta - arg(I)) = I_r*sin(theta) - I_i*cos(theta)
```

По умолчанию `beta2 = 0`, поэтому эксперименты используют supercritical/critical
упрощение, описанное в статье.

Статус: исправлено.

Важная знаковая конвенция:

- статья использует `beta < 0` для устойчивой supercritical Hopf-динамики;
- в коде теперь `beta = -100.0` по умолчанию, как в параметрах Fig. 11 для
  supercritical Hopf response;
- `beta2 = 0.0` по умолчанию;
- устойчивый радиус без входа: `sqrt(-mu / beta)`.

## Table 1: signal classification

С присланного фрагмента Table 1:

```text
initial frequency range: 0.1-20 Hz
architecture: Linear(20) -> Hopf(20) -> tanh(20) -> output(2)
input type to oscillators: I(t)
oscillator frequencies: not trained
```

Figure caption дополнительно говорит, что Class I сильнее активирует
осцилляторы `0.1-10 Hz`, а Class II - `10-20 Hz`.

Формула статьи для входных сигналов:

```text
sum_i A_i sin(2*pi*f_i*t + phi_i)
```

Основной CE-runner читает сохраненный supplement-style датасет:

- `artifacts/signal_generation/X.npy`;
- `artifacts/signal_generation/Y.npy`.

Также добавлен paper-style runner `visual/classifier_paper_result.py`, который
умеет генерировать два варианта:

- `--dataset-source article`: sine + random phase по тексту статьи;
- `--dataset-source supplement-notebook`: cosine + discrete frequency grid +
  white noise, как в приложенном notebook.

`src/classifier.py` проверяет классификационную постановку поверх сохраненных
target ramp labels. Архитектурный размер `tanh(20)` теперь приведен к Table 1,
но есть две разные objective-проверки:

- `visual/classifier_result.py`: sparse cross-entropy по классу, извлеченному
  из ramp-target;
- `visual/classifier_paper_result.py`: paper-style ramp targets + MSE.

Статус: формула модели Hopf исправлена; размерности Table 1 согласованы ближе
к таблице статьи; генераторы article/supplement восстановлены.

Результаты после исправления Hopf:

- CE-control на сохраненных `.npy`: `test_acc = 0.58`;
- ramp/MSE на supplement-notebook generator:
  - mean/sum accuracy `0.47`;
  - final-step accuracy `0.61`;
  - template-MSE accuracy `0.49`;
- ramp/MSE на article generator:
  - mean/sum accuracy `0.38`;
  - final-step accuracy `0.515`;
  - template-MSE accuracy `0.415`.

Что желательно уточнить: чтобы объяснить разрыв с заявленными `99%`, нужен
точный авторский forward pass/тренировочный код для `Linear -> Hopf -> tanh ->
output`. Дополнительные метрики показывают, что ошибка не сводится только к
выбору mean-vs-final readout.

## Table 2: amplitude demodulation

Формула статьи:

```text
M(t) = (1 + m(t)) * sin(omega_c*t)
m(t) = sum_i sin(2*pi*f_i*t), f_i in U(1, 5) Hz
omega_c / (2*pi) = 8 Hz
```

В `src/demodulation.py`:

```python
message = sum(sin(2*pi*freq*t))
carrier = sin(2*pi*carrier_hz*t)
modulated = (1 + message) * carrier
```

Статус: совпадает.

## Table 3: integration and differentiation

Формулы статьи:

```text
I(t) = sum_i a_i sin(omega_i*t + phi_i)

integration:
O(t) = - sum_i a_i/omega_i * cos(omega_i*t + phi_i)

differentiation:
O(t) = sum_i a_i*omega_i * cos(omega_i*t + phi_i)
```

В `src/operators.py` используются эти же аналитические выражения.

Статус: совпадает.

Замечание: интеграл задан как один представитель первообразной, без
произвольной константы. Это согласуется с тем, что надо обучать конкретную
целевую траекторию.

## Table 4: sentiment analysis

Из статьи:

```text
Embedding(100) -> Hopf(100) -> ReLU(100) -> Hopf(100) -> ReLU(100)
-> tanh(20) -> output(2)
```

Параметры:

- vocabulary length `35000`;
- review length `500`;
- input type `I(t)`;
- oscillator frequencies trained;
- optimizer Adam;
- learning rate `0.001`;
- loss MSE.
- data split reported as `7:3`;
- reported DONN accuracy `85.2%`;
- reported trainable parameters `26,798`.

С присланного фрагмента Table 4 также видно, что baseline-сравнение взято с
моделями из ссылки 23:

- Bidirectional LSTM: `85.19%`;
- Bidirectional flip-flops: `85.07%`.

В `src/sentiment.py` есть два пути:

- старый sequence-ramp вариант;
- более строгий `PaperDONNSentimentClassifier` many-to-one с MSE по one-hot.

Статус: архитектурная формула в paper-style пути близка к статье, но сама
статья внутренне противоречива по числу параметров: один embedding
`35000 x 100` уже дает `3,500,000` параметров, тогда как статья сообщает
`26,798`.

Что остается неясным: реальный способ подсчета/использования embedding и
точная постановка MSE для Table 4. Также в локальном runner используется
стандартный IMDB train/test split и `val_ratio=0.3` внутри train split, а не
буквальное пересоздание одного общего `7:3` split из текста статьи.

## Table 5: action recognition

Из статьи:

```text
dataset: UCF11
frames per video: 50
frame size: 48 x 48 x 3
train/validation: 1290 / 305
dt: 0.02
architecture: 2 x OCNN(3x3, 40) -> flatten -> output(2)
initial frequency range: 1-15 Hz
input type: I(t)
oscillator frequencies: trained
optimizer: Adam
learning rate: 0.0001
loss: MSE
```

В `src/action_recognition.py` реализован OCNN-style smoke path:

```text
2 x ConvOsc(3x3, filters) -> flatten -> output(num_classes)
```

Статус: строгий Table 5 не воспроизведен, потому что локального UCF11/UCF50
датасета нет. Текущий `visual/action_recognition_result.py` делает synthetic
smoke-run и сохраняет `is_ucf11_reproduction=false`.

Замечание: в статье есть внутренняя нестыковка: Table 5 сообщает для OCNN
`98.64%`, а текст под Figure 4 говорит `99.75%`, что совпадает со строкой
Convolutional flip-flops.

## Case study 1: temporal binding

Формула synchrony из статьи:

```text
S = (1/T) * sum_t | (1/N) * sum_i z_i(t)/|z_i(t)| |
```

В `src/temporal_binding.py`:

```python
normalized = selected / (abs(selected) + 1e-8)
S = mean(abs(mean(normalized, axis=1)))
```

Статус: совпадает.

Set algebra для `A_hat`, `B_hat`, `X_hat`, `Y_hat` реализована по тексту
статьи. Дополнительно учтено, что notebook кодирует классы как:

```text
class = 2 * orientation + color
```

Статус: формулы synchrony и set algebra совпадают; полный ConvOsc hidden-layer
повтор еще не сделан.

## Case study 2: STDP kernel

Формулы статьи:

```text
zdot1 = (mu + i*omega1) z1 - |z1|^2 z1 + W*z2 + p(t)
zdot2 = (mu + i*omega2) z2 - |z2|^2 z2 + conjugate(W)*z1 + p(t + tau)
Wdot = -W + eta*z1*conjugate(z2)
```

В `src/stdp_kernel.py` default `paper-conjugate` использует именно эти
уравнения. Literal-вариант `z1*z2` оставлен как диагностический контроль.

Статус: совпадает на уровне напечатанных уравнений.

С присланного фрагмента Fig. 7 уточнено, что ось задержки показана как
`tau = -10..10` за один период осцилляций. В runner это теперь отражено через
`tau_min=-10`, `tau_max=10`, `tau_units_per_period=20`.

Ограничение: статья не задает численные параметры импульса, начальные условия и
шаг интегрирования, поэтому `visual/stdp_kernel_result.py` является
equation-level control, а не точным повтором Fig. 7.

Дополнительная неясность снята после проверки HTML/текста статьи: нужна
сопряженная форма. Старый literal-вариант оставлен как
`--hebbian-product literal-control`.

## Требует дополнительного изображения/уточнения

Если хотим довести аудит до “строгого все формулы 1-в-1”, стоит прислать или
выделить:

1. точное описание генерации Table 1 и правила классов;
2. фрагмент Table 4/методов про embedding и MSE-target;
3. локальный UCF11/UCF50 датасет для Table 5;
4. формулу ConvOsc-слоя, если она есть отдельно от обычного Hopf/OCNN описания;
5. численные параметры STDP Fig. 7, если они есть в исходном коде авторов.
