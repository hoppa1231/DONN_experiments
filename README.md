# DONN Experiments

Clean, showable reproductions of selected DONN paper tables.

## Structure

- `src/`
  - reusable model and training code
- `visual/`
  - one-file entrypoints that train and save the final report figure + metrics
- `artifacts/plots/`
  - final outputs to inspect

## Table 1

Code:

- `src/classifier.py`
- `visual/classifier_result.py`

Current result:

- `artifacts/plots/first_work_visual_comparison_ce.png`
- `artifacts/plots/first_work_visual_metrics_ce.json`

Run:

```powershell
.\.venv\Scripts\python visual\classifier_result.py --use-linear-frontend
```

## Table 2

Code:

- `src/demodulation.py`
- `visual/demodulation_result.py`

Current result:

- `artifacts/plots/second_work_visual_comparison_fixed.png`
- `artifacts/plots/second_work_visual_metrics_fixed.json`

Run:

```powershell
.\.venv\Scripts\python visual\demodulation_result.py --use-linear-frontend --use-input-skip
```

Notes:

- Table 2 code generates the synthetic demodulation dataset directly from the
  paper description.
- It does not depend on the misleading old `artifacts/amplitude_demodulation/*.npy`
  files.

## Table 3

Code:

- `src/operators.py`
- `visual/operators_result.py`

Current result:

- `artifacts/plots/third_work_visual_summary.png`
- `artifacts/plots/third_work_visual_metrics.json`

Run:

```powershell
.\.venv\Scripts\python visual\operators_result.py --use-linear-frontend
```

Notes:

- Table 3 uses the article-style formulas for both integration and differentiation.
- The summary figure overlays the DONN prediction and a simple numeric baseline
  to make the remaining gap obvious at a glance.

## Table 4

Code:

- `src/sentiment.py`
- `visual/sentiment_result.py`
- `visual/sentiment_paper_result.py`

Current result:

- `artifacts/plots/fourth_work_visual_summary.png`
- `artifacts/plots/fourth_work_visual_metrics.json`
- `artifacts/plots/fourth_work_paper_exact_summary_4k3e.png`
- `artifacts/plots/fourth_work_paper_exact_metrics_4k3e.json`

Run:

```powershell
.\.venv\Scripts\python visual\sentiment_result.py
```

Paper-style control:

```powershell
.\.venv\Scripts\python visual\sentiment_paper_result.py --train-samples 4096 --test-samples 4096 --epochs 3 --batch-size 256
```

Notes:

- This is a tractable local IMDB reproduction built around the Table 4 paper setup:
  top-35000 vocabulary, review length 500, embedding size 100, trainable Hopf
  frequencies in the 1-15 Hz range.
- The local DONN transfer to text is currently weak: on the saved run it stays
  near random-guess accuracy, while a simple Bidirectional LSTM baseline on the
  same subset reaches a noticeably higher score.
- `visual/sentiment_paper_result.py` is the stricter control path for checking
  the Table 4 architecture from the full-size table, without the older
  sequence-ramp approximation.

## Docs

- `docs/FILE_GUIDE.md` explains every working file in the repository in plain language.
- `docs/TABLE4_SENTIMENT.md` explains the Table 4 sentiment experiment, what is
  paper-faithful, and what is only a local practical adaptation.
- `docs/code/` contains separate Russian explanations for each Python code file.
