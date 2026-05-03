# Путеводитель по файлам

Этот файл нужен как короткая карта проекта:

- что за файл перед вами,
- зачем он нужен,
- в каком порядке лучше изучать код.

Если хочется быстро войти в проект, не открывайте файлы хаотично. Идите по порядку ниже.

## Порядок изучения

1. [README.md](C:/Users/user/Projects/DONN_experiments/README.md)  
   Сначала понять, какие таблицы вообще реализованы и какими командами они запускаются.

2. [docs/code/README.md](C:/Users/user/Projects/DONN_experiments/docs/code/README.md)  
   Это оглавление русских разборов кода.

3. [src/HopfLayer.py](C:/Users/user/Projects/DONN_experiments/src/HopfLayer.py)  
   Главный общий механизм DONN. Без него дальше читать проект почти бессмысленно.

4. Table 1:
   [src/classifier.py](C:/Users/user/Projects/DONN_experiments/src/classifier.py) -> [visual/classifier_result.py](C:/Users/user/Projects/DONN_experiments/visual/classifier_result.py)

5. Table 2:
   [src/demodulation.py](C:/Users/user/Projects/DONN_experiments/src/demodulation.py) -> [visual/demodulation_result.py](C:/Users/user/Projects/DONN_experiments/visual/demodulation_result.py)

6. Table 3:
   [src/operators.py](C:/Users/user/Projects/DONN_experiments/src/operators.py) -> [visual/operators_result.py](C:/Users/user/Projects/DONN_experiments/visual/operators_result.py)

7. Table 4:
   [src/sentiment.py](C:/Users/user/Projects/DONN_experiments/src/sentiment.py) -> [visual/sentiment_result.py](C:/Users/user/Projects/DONN_experiments/visual/sentiment_result.py)
   [src/sentiment.py](C:/Users/user/Projects/DONN_experiments/src/sentiment.py) -> [visual/sentiment_paper_result.py](C:/Users/user/Projects/DONN_experiments/visual/sentiment_paper_result.py)

8. Table 5:
   [src/action_recognition.py](C:/Users/user/Projects/DONN_experiments/src/action_recognition.py) -> [visual/action_recognition_result.py](C:/Users/user/Projects/DONN_experiments/visual/action_recognition_result.py)

9. Case study 1:
   [src/temporal_binding.py](C:/Users/user/Projects/DONN_experiments/src/temporal_binding.py) -> [visual/temporal_binding_result.py](C:/Users/user/Projects/DONN_experiments/visual/temporal_binding_result.py)

10. Case study 2:
   [src/stdp_kernel.py](C:/Users/user/Projects/DONN_experiments/src/stdp_kernel.py) -> [visual/stdp_kernel_result.py](C:/Users/user/Projects/DONN_experiments/visual/stdp_kernel_result.py)

11. После этого уже смотреть артефакты в [artifacts/plots](C:/Users/user/Projects/DONN_experiments/artifacts/plots), пояснения в [docs/TABLE4_SENTIMENT.md](C:/Users/user/Projects/DONN_experiments/docs/TABLE4_SENTIMENT.md), [docs/TABLE5_ACTION_RECOGNITION.md](C:/Users/user/Projects/DONN_experiments/docs/TABLE5_ACTION_RECOGNITION.md), [docs/TEMPORAL_BINDING.md](C:/Users/user/Projects/DONN_experiments/docs/TEMPORAL_BINDING.md), [docs/STDP_KERNEL.md](C:/Users/user/Projects/DONN_experiments/docs/STDP_KERNEL.md) и [docs/FORMULA_AUDIT.md](C:/Users/user/Projects/DONN_experiments/docs/FORMULA_AUDIT.md).

## Главные файлы

### Корень проекта

- [README.md](C:/Users/user/Projects/DONN_experiments/README.md)  
  Главная точка входа: список таблиц, команды запуска, ссылки на итоговые артефакты.

- [requirements.txt](C:/Users/user/Projects/DONN_experiments/requirements.txt)  
  Список библиотек и версий, на которых всё запускалось.

- [.gitignore](C:/Users/user/Projects/DONN_experiments/.gitignore)  
  Служебный файл Git, чтобы не тащить мусор в репозиторий.

- [.gitattributes](C:/Users/user/Projects/DONN_experiments/.gitattributes)  
  Служебные настройки Git.

### Общая DONN-механика

- [src/HopfLayer.py](C:/Users/user/Projects/DONN_experiments/src/HopfLayer.py)  
  Общий слой Хопфа: осцилляторы, шаг интегрирования, частоты, seed.

- [docs/code/src_HopfLayer.md](C:/Users/user/Projects/DONN_experiments/docs/code/src_HopfLayer.md)  
  Русский подробный разбор `HopfLayer.py`.

### Table 1

- [src/classifier.py](C:/Users/user/Projects/DONN_experiments/src/classifier.py)  
  Модель и обучение для классификации сигналов.

- [visual/classifier_result.py](C:/Users/user/Projects/DONN_experiments/visual/classifier_result.py)  
  Запуск эксперимента и построение финального графика для Table 1.

- [artifacts/plots/table1/first_work_visual_comparison_ce.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table1/first_work_visual_comparison_ce.png)  
  Итоговая картинка.

- [artifacts/plots/table1/first_work_visual_metrics_ce.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table1/first_work_visual_metrics_ce.json)  
  Итоговые метрики.

- [docs/code/src_classifier.md](C:/Users/user/Projects/DONN_experiments/docs/code/src_classifier.md)  
  Русский разбор логики модели.

- [docs/code/visual_classifier_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_classifier_result.md)  
  Русский разбор visual-скрипта.

### Table 2

- [src/demodulation.py](C:/Users/user/Projects/DONN_experiments/src/demodulation.py)  
  Генерация задачи и модель амплитудной демодуляции.

- [visual/demodulation_result.py](C:/Users/user/Projects/DONN_experiments/visual/demodulation_result.py)  
  Запуск эксперимента и сохранение отчёта.

- [artifacts/plots/table2/second_work_visual_comparison_fixed.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table2/second_work_visual_comparison_fixed.png)  
  Итоговая картинка.

- [artifacts/plots/table2/second_work_visual_metrics_fixed.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table2/second_work_visual_metrics_fixed.json)  
  Итоговые метрики.

- [docs/code/src_demodulation.md](C:/Users/user/Projects/DONN_experiments/docs/code/src_demodulation.md)  
  Русский разбор модели.

- [docs/code/visual_demodulation_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_demodulation_result.md)  
  Русский разбор visual-скрипта.

### Table 3

- [src/operators.py](C:/Users/user/Projects/DONN_experiments/src/operators.py)  
  Генерация данных и обучение для интегрирования и дифференцирования.

- [visual/operators_result.py](C:/Users/user/Projects/DONN_experiments/visual/operators_result.py)  
  Общий visual-отчёт сразу для обеих задач.

- [artifacts/plots/table3/third_work_visual_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table3/third_work_visual_summary.png)  
  Итоговая картинка.

- [artifacts/plots/table3/third_work_visual_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table3/third_work_visual_metrics.json)  
  Итоговые метрики.

- [docs/code/src_operators.md](C:/Users/user/Projects/DONN_experiments/docs/code/src_operators.md)  
  Русский разбор вычислительной логики.

- [docs/code/visual_operators_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_operators_result.md)  
  Русский разбор visual-скрипта.

### Table 4

- [src/sentiment.py](C:/Users/user/Projects/DONN_experiments/src/sentiment.py)  
  IMDB, DONN-модель для текста и baseline на Bidirectional LSTM.

- [visual/sentiment_result.py](C:/Users/user/Projects/DONN_experiments/visual/sentiment_result.py)  
  Visual-отчёт по анализу тональности.

- [visual/sentiment_paper_result.py](C:/Users/user/Projects/DONN_experiments/visual/sentiment_paper_result.py)  
  Контрольный paper-style прогон для проверки опубликованной Table 4.

- [artifacts/plots/table4/fourth_work_visual_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table4/fourth_work_visual_summary.png)  
  Итоговая картинка.

- [artifacts/plots/table4/fourth_work_visual_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table4/fourth_work_visual_metrics.json)  
  Итоговые метрики.

- [artifacts/plots/table4/fourth_work_paper_exact_summary_4k3e.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table4/fourth_work_paper_exact_summary_4k3e.png)  
  Итоговая картинка строгого контрольного прогона по статье.

- [artifacts/plots/table4/fourth_work_paper_exact_metrics_4k3e.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table4/fourth_work_paper_exact_metrics_4k3e.json)  
  Метрики строгого paper-style контроля.

- [docs/TABLE4_SENTIMENT.md](C:/Users/user/Projects/DONN_experiments/docs/TABLE4_SENTIMENT.md)  
  Отдельная заметка про состояние Table 4 и почему DONN там пока слабый.

- [src/temporal_binding.py](C:/Users/user/Projects/DONN_experiments/src/temporal_binding.py)  
  Проверочные функции для Case study 1: аудит moving-bar датасета, выбор feature-групп и расчет synchrony.

- [visual/temporal_binding_result.py](C:/Users/user/Projects/DONN_experiments/visual/temporal_binding_result.py)  
  Запуск контрольного temporal-binding анализа и сохранение картинки/метрик.

- [artifacts/plots/case_study/case_study_temporal_binding_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/case_study/case_study_temporal_binding_summary.png)  
  Итоговая картинка для Case study 1 control-run.

- [artifacts/plots/case_study/case_study_temporal_binding_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/case_study/case_study_temporal_binding_metrics.json)  
  Метрики аудита датасета и synchrony-сравнения.

- [artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_summary.png)  
  Такой же control-run, но на исправленной генерации moving-bar видео.

- [artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_metrics.json)  
  Метрики исправленной генерации: пустые кадры должны исчезнуть.

- [docs/TEMPORAL_BINDING.md](C:/Users/user/Projects/DONN_experiments/docs/TEMPORAL_BINDING.md)  
  Отдельная заметка про Case study 1, найденную проблему в генераторе и оставшийся путь до строгого ConvOsc-повтора.

- [src/stdp_kernel.py](C:/Users/user/Projects/DONN_experiments/src/stdp_kernel.py)  
  Уравнения и sweep по задержке для STDP-like kernel на паре Hopf-осцилляторов.

- [visual/stdp_kernel_result.py](C:/Users/user/Projects/DONN_experiments/visual/stdp_kernel_result.py)  
  Запуск Case study 2 и сохранение графика/метрик.

- [artifacts/plots/case_study/case_study_stdp_kernel_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/case_study/case_study_stdp_kernel_summary.png)  
  Итоговая картинка equation-level STDP-контроля.

- [artifacts/plots/case_study/case_study_stdp_kernel_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/case_study/case_study_stdp_kernel_metrics.json)  
  Метрики и delay-sweep для STDP-контроля.

- [docs/STDP_KERNEL.md](C:/Users/user/Projects/DONN_experiments/docs/STDP_KERNEL.md)  
  Отдельная заметка про Case study 2 и ограничения точного повтора Fig. 7.

- [docs/FORMULA_AUDIT.md](C:/Users/user/Projects/DONN_experiments/docs/FORMULA_AUDIT.md)  
  Общая сверка формул статьи с текущими реализациями в `src/`.

- [docs/code/src_sentiment.md](C:/Users/user/Projects/DONN_experiments/docs/code/src_sentiment.md)  
  Русский разбор основного кода.

- [docs/code/visual_sentiment_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_sentiment_result.md)  
  Русский разбор visual-скрипта.

- [docs/code/visual_sentiment_paper_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_sentiment_paper_result.md)  
  Русский разбор контрольного visual-скрипта для Table 4.

### Table 5

- [src/action_recognition.py](C:/Users/user/Projects/DONN_experiments/src/action_recognition.py)  
  OCNN-style слой и synthetic smoke-run для проверки Table 5 code path.

- [visual/action_recognition_result.py](C:/Users/user/Projects/DONN_experiments/visual/action_recognition_result.py)  
  Запуск Table 5 smoke-control и сохранение PNG/JSON.

- [artifacts/plots/table5/fifth_work_ocnn_smoke_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table5/fifth_work_ocnn_smoke_summary.png)  
  Итоговая картинка smoke-control.

- [artifacts/plots/table5/fifth_work_ocnn_smoke_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/table5/fifth_work_ocnn_smoke_metrics.json)  
  Метрики smoke-control и флаг, что это не UCF11 reproduction.

- [docs/TABLE5_ACTION_RECOGNITION.md](C:/Users/user/Projects/DONN_experiments/docs/TABLE5_ACTION_RECOGNITION.md)  
  Отдельная заметка про Table 5 и отсутствующий локальный UCF dataset.

## Документация

- [docs/code/README.md](C:/Users/user/Projects/DONN_experiments/docs/code/README.md)  
  Оглавление русских разборов кода.

- [docs/DONN.pdf](C:/Users/user/Projects/DONN_experiments/docs/DONN.pdf)  
  Основная статья.

- [docs/DONN_appendix_1.pdf](C:/Users/user/Projects/DONN_experiments/docs/DONN_appendix_1.pdf)  
  Дополнительные материалы 1.

- [docs/DONN_appendix_2.pdf](C:/Users/user/Projects/DONN_experiments/docs/DONN_appendix_2.pdf)  
  Дополнительные материалы 2.

## Если нужен самый короткий маршрут

Если задача просто понять проект без глубокого копания, хватит такого пути:

1. [README.md](C:/Users/user/Projects/DONN_experiments/README.md)
2. [src/HopfLayer.py](C:/Users/user/Projects/DONN_experiments/src/HopfLayer.py)
3. [src/classifier.py](C:/Users/user/Projects/DONN_experiments/src/classifier.py)
4. [src/demodulation.py](C:/Users/user/Projects/DONN_experiments/src/demodulation.py)
5. [src/operators.py](C:/Users/user/Projects/DONN_experiments/src/operators.py)
6. [src/sentiment.py](C:/Users/user/Projects/DONN_experiments/src/sentiment.py)

Если задача не только понять, но и разбирать код очень подробно, после каждого файла из `src/`
сразу открывайте его объяснение из [docs/code](C:/Users/user/Projects/DONN_experiments/docs/code).
