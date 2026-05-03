# Repro Audit Matrix

Цель этого файла: не смешивать "получили какой-то похожий график" и
"воспроизвели условия статьи". Ниже для каждого эксперимента отдельно
зафиксировано:

- что именно утверждает статья;
- что сейчас совпадает в коде;
- что пока является локальной адаптацией;
- какой runner лучше считать основным для проверки достоверности.

Источники по статье в репозитории:

- `tmp/pdf_text/DONN.txt`
- `tmp/pdf_text/DONN_appendix_1.txt`
- `docs/FORMULA_AUDIT.md`

## Статусы

- `paper-faithful`: ключевые условия статьи воспроизведены достаточно близко.
- `adapted`: эксперимент полезен как локальный контроль, но условия статьи
  изменены.
- `not-yet-reproduced`: строгого воспроизведения статьи пока нет.

## Table 1: Classification

Статья:

- датасет генерируется по формуле
  `I(t) = sum_i A_i sin(2*pi*f_i*t + phi_i)`;
- `A_i ~ U[-3, 3]`;
- `phi_i ~ U[0, 2*pi]`;
- Class 1: все `f_i` в `[0, 10] Hz`;
- Class 2: все `f_i` в `[10, 20] Hz`;
- архитектура: `Linear(20) -> Hopf(20) -> tanh(20) -> output(2)`;
- собственные частоты осцилляторов: `[0.1, 20] Hz`;
- вход в осцилляторы: `I(t)`;
- частоты осцилляторов: `not trained`;
- заявленный результат: `99% accuracy`.

Что сейчас в коде:

- модель по размерностям близка к статье:
  `DONNClassifierCE` в `src/classifier.py`;
- Hopf frequencies нетренируемые, что совпадает со статьёй;
- вход в Hopf идёт как `I(t)` через `x_r/x_i`;
- CE visual runner: `visual/classifier_result.py`;
- paper-style ramp/MSE runner: `visual/classifier_paper_result.py`;
- strict complex-static runner: `visual/paper_classification_result.py`.

Что не совпадает / не подтверждено:

- сохранённые `artifacts/signal_generation/X.npy` и `Y.npy` соответствуют
  supplement notebook, а не текстовой формуле статьи;
- `visual/classifier_paper_result.py` проверяет оба варианта генератора:
  `--dataset-source article` и `--dataset-source supplement-notebook`;
- точный авторский forward pass для `Linear -> Hopf -> tanh -> output` всё ещё
  не подтверждён.

Статус: `adapted`

Основной вывод:

- архитектурный каркас Table 1 проверяется;
- после исправления Hopf локальные результаты остаются около случайного уровня:
  CE-control `test_acc = 0.58`, supplement ramp/MSE mean/final
  `0.47/0.61`, article ramp/MSE mean/final `0.38/0.515`;
- заявленные `99%` пока не воспроизводятся ни по текстовому генератору, ни по
  supplement notebook generator.
- strict complex-static runner на supplement notebook за 20 эпох дает
  mean accuracy `0.605`, final-step accuracy `0.56`.

## Table 2: Amplitude Demodulation

Статья:

- `M(t) = (1 + m(t)) * sin(omega_c t)`;
- `m(t) = sum_i sin(2*pi*f_i*t)`, `f_i in U(1, 5) Hz`;
- carrier fixed at `8 Hz`;
- архитектура: `ReLU(40) -> Hopf(40) -> ReLU(40) -> Hopf(40) -> tanh(40) -> output(1)`;
- oscillator init range: `[0.1, 12] Hz`;
- input type: `I(t)`;
- oscillator frequencies: `not trained`;
- заявленный результат: `validation MSE = 0.02 (p < 0.05, n = 10)`.

Что сейчас в коде:

- формула данных совпадает в `src/demodulation.py`;
- `carrier_hz=8.0`, `msg_fmin=1.0`, `msg_fmax=5.0`;
- Hopf frequencies инициализируются в `[0.1, 12] Hz`;
- frequencies не тренируются;
- вход в Hopf соответствует `I(t)`;
- используется синтетический генератор по статье, а не старые `.npy`;
- strict runner `visual/paper_sequence_result.py --task demodulation`
  реализует напечатанную архитектуру с двумя Hopf-слоями и complex static
  layers.

Что не совпадает / не подтверждено:

- архитектура головы отличается: вместо второго Hopf-блока используется
  temporal Conv1D readout;
- текущий runner проверяет ту же задачу, но не тот же точный head;
- strict runner архитектурно ближе к статье, но текущий strict 400/60 run дает
  `val_mse = 3.509055`, а не заявленные `0.02`;
- статья пишет только `val MSE`, а у нас фиксируются `test_mse` и `val_mse`
  одного конкретного прогона.

Статус: `adapted`

Основной вывод:

- задача и формула демодуляции воспроизведены честно;
- архитектура не paper-exact, поэтому это сильный functional control,
  но не полное буквальное воспроизведение Table 2.

## Table 3: Mathematical Operators

Статья:

- вход:
  `I(t) = sum_i a_i sin(omega_i*t + phi_i)`;
- `a_i ~ N(0, 1)`;
- `phi_i ~ N(0, pi)`;
- `omega_i` sampled from `U(1, 5)`;
- integration target:
  `O(t) = - sum_i a_i/omega_i * cos(omega_i*t + phi_i)`;
- differentiation target:
  `O(t) = sum_i a_i*omega_i * cos(omega_i*t + phi_i)`;
- dataset frequency range: `[0.1, 5] Hz` for input data row in table;
- oscillator init range: `[1, 10] Hz`;
- архитектура: `ReLU(20) -> Hopf(20) -> ReLU(20) -> Hopf(20) -> tanh(20) -> output(1)`;
- input type: `I(t)`;
- oscillator frequencies: `not trained`;
- заявленные результаты:
  integration `val MSE = 0.08`, differentiation `MSE = 0.1`.

Что сейчас в коде:

- аналитические формулы integration/differentiation совпадают со статьёй в
  `src/operators.py`;
- амплитуды, фазы и частоты для генератора заданы по тем же распределениям;
- численный baseline добавлен как независимая sanity-check проверка;
- frequencies не тренируются;
- вход в Hopf соответствует `I(t)`.

Что не совпадает / не подтверждено:

- вместо paper architecture с двумя Hopf-блоками сейчас используется
  `Linear -> Hopf -> Conv1D temporal readout`;
- oscillator init range по умолчанию у `HopfLayer` шире, чем в Table 3;
- raw `MSE` между integration и differentiation нельзя сравнивать напрямую:
  после правильной формулы у differentiation гораздо больше масштаб цели;
- из-за этого для интерпретации нужно смотреть `normalized_mse`, `RMSE/std`
  и `R2`, а не только raw `MSE`.

Статус: `adapted`

Основной вывод:

- формулы и данные Table 3 проверены хорошо;
- строгого paper-exact architecture reproduction пока нет;
- после исправления формулы differentiation не "стала хуже", а просто получила
  более крупный физически корректный масштаб цели.

## Table 4: Sentiment Analysis

Статья:

- IMDB, maximum review length `500`;
- training/validation split `7:3`;
- vocabulary size `35000`;
- embedding dimension `100`;
- architecture:
  `Embedding(100) -> Hopf(100) -> ReLU(100) -> Hopf(100) -> ReLU(100) -> tanh(20) -> output(2)`;
- oscillator frequency init range `[1, 15] Hz`;
- input type `I(t)`;
- oscillator frequencies: `trained`;
- optimizer: `Adam`;
- learning rate: `0.001`;
- objective: `MSE`;
- paper-reported params: `26,798`;
- paper-reported test accuracy: `85.2%`.

Что сейчас в коде:

- есть два разных пути:
  - `visual/sentiment_result.py`: локальная tractable adaptation;
  - `visual/sentiment_paper_result.py`: более строгий paper-style control;
- в `PaperDONNSentimentClassifier`:
  - `Embedding(100)`,
  - `Hopf(100)`,
  - `ReLU(100)`,
  - `Hopf(100)`,
  - `ReLU(100)`,
  - `tanh(20)`,
  - `output(2)`;
- Hopf frequencies trainable;
- optimizer `Adam`, learning rate `0.001`, loss `MSE`;
- vocab size `35000`, max length `500`.

Что не совпадает / не подтверждено:

- статья внутренне противоречива по числу параметров:
  один embedding `35000 x 100` уже даёт `3,500,000` параметров, что несовместимо
  с заявленными `26,798`;
- рабочий `paper_result` использует many-to-one MSE-on-one-hot assumption,
  потому что точная постановка MSE из статьи не раскрыта полностью;
- локальный `visual/sentiment_result.py` не paper-exact и нужен только как
  облегчённый контроль;
- без прояснения параметров embeddings Table 4 нельзя считать строго
  воспроизведённой даже при совпадающей архитектуре верхнего уровня.

Статус:

- `visual/sentiment_result.py`: `adapted`
- `visual/sentiment_paper_result.py`: `paper-faithful` по архитектуре и
  гиперпараметрам, но `not-yet-reproduced` по внутренней непротиворечивости
  статьи

Основной вывод:

- Table 4 сейчас лучше всего исследовать именно через paper-style runner;
- главная проблема уже не в нашем коде, а в несогласованности текста статьи по
  trainable parameters и постановке objective.

## Case Study 1: Temporal Binding

Статья:

- проверяется гипотеза, что group oscillators synchronise stronger than
  residuary oscillators;
- synchrony считается как order parameter;
- использован ConvOsc/OCNN hidden-layer setting на dataset moving bars.

Что сейчас в коде:

- формула synchrony совпадает;
- algebra of `A_hat/B_hat/X_hat/Y_hat` проверена;
- добавлен audit supplied dataset;
- обнаружен generator/wraparound bug в выложенном датасете;
- есть corrected-generator control.

Что не совпадает / не подтверждено:

- это не full trained ConvOsc reproduction;
- текущий код проверяет dataset + synchrony algorithm control, а не весь
  обученный visual model path из статьи.

Статус: `adapted`

Основной вывод:

- qualitative claim про `group > residuary` локально подтверждается;
- но это ещё не строгая проверка всей case-study architecture статьи.

## Case Study 2: STDP Kernel

Статья:

- задаёт уравнения coupled Hopf pair and weight dynamics;
- показывает STDP-like sweep по delay `tau`.

Что сейчас в коде:

- literal equations реализованы в `src/stdp_kernel.py`;
- delay axis приведена к виду `tau = -10..10` per period;
- есть diagnostic branch для conjugate-control, потому что текст статьи и
  напечатанное уравнение не полностью однозначны.

Что не совпадает / не подтверждено:

- статья не даёт полный набор численных параметров импульса и интегрирования;
- текущий код therefore equation-level control, not exact Fig. 7 reproduction.

Статус: `adapted`

Основной вывод:

- уравнения проверяются;
- строгой численной reproduction of Fig. 7 пока нет из-за неполной
  спецификации статьи.

## Приоритеты для дальнейшей строгой проверки

0. Oscillator ablation: текущий локальный вывод сохранен в
   `docs/OSCILLATOR_ABLATION.md`; широкого преимущества Hopf-слоев пока не
   видно, кроме Table 2 demodulation.
1. Table 1: найти авторский forward pass, потому article/supplement генераторы
   уже проверены и остаются около случайного уровня.
2. Table 2: собрать более близкий к статье two-Hopf architecture runner.
3. Table 3: добавить paper-exact two-Hopf architecture control рядом с текущим
   Conv1D control.
4. Table 4: продолжать только через `visual/sentiment_paper_result.py` и
   отдельно документировать внутреннее противоречие статьи по параметрам.
5. Case study 1: если хотим строгую проверку статьи, нужен именно trained ConvOsc
   path, а не только synchrony control.
6. Case study 2: сохранить как equation-level audit до появления недостающих
   численных условий.
