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
