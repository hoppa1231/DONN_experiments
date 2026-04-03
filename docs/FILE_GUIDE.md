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

8. После этого уже смотреть артефакты в [artifacts/plots](C:/Users/user/Projects/DONN_experiments/artifacts/plots) и пояснения в [docs/TABLE4_SENTIMENT.md](C:/Users/user/Projects/DONN_experiments/docs/TABLE4_SENTIMENT.md).

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

- [artifacts/plots/first_work_visual_comparison_ce.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/first_work_visual_comparison_ce.png)  
  Итоговая картинка.

- [artifacts/plots/first_work_visual_metrics_ce.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/first_work_visual_metrics_ce.json)  
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

- [artifacts/plots/second_work_visual_comparison_fixed.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/second_work_visual_comparison_fixed.png)  
  Итоговая картинка.

- [artifacts/plots/second_work_visual_metrics_fixed.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/second_work_visual_metrics_fixed.json)  
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

- [artifacts/plots/third_work_visual_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/third_work_visual_summary.png)  
  Итоговая картинка.

- [artifacts/plots/third_work_visual_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/third_work_visual_metrics.json)  
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

- [artifacts/plots/fourth_work_visual_summary.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/fourth_work_visual_summary.png)  
  Итоговая картинка.

- [artifacts/plots/fourth_work_visual_metrics.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/fourth_work_visual_metrics.json)  
  Итоговые метрики.

- [artifacts/plots/fourth_work_paper_exact_summary_4k3e.png](C:/Users/user/Projects/DONN_experiments/artifacts/plots/fourth_work_paper_exact_summary_4k3e.png)  
  Итоговая картинка строгого контрольного прогона по статье.

- [artifacts/plots/fourth_work_paper_exact_metrics_4k3e.json](C:/Users/user/Projects/DONN_experiments/artifacts/plots/fourth_work_paper_exact_metrics_4k3e.json)  
  Метрики строгого paper-style контроля.

- [docs/TABLE4_SENTIMENT.md](C:/Users/user/Projects/DONN_experiments/docs/TABLE4_SENTIMENT.md)  
  Отдельная заметка про состояние Table 4 и почему DONN там пока слабый.

- [docs/code/src_sentiment.md](C:/Users/user/Projects/DONN_experiments/docs/code/src_sentiment.md)  
  Русский разбор основного кода.

- [docs/code/visual_sentiment_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_sentiment_result.md)  
  Русский разбор visual-скрипта.

- [docs/code/visual_sentiment_paper_result.md](C:/Users/user/Projects/DONN_experiments/docs/code/visual_sentiment_paper_result.md)  
  Русский разбор контрольного visual-скрипта для Table 4.

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
