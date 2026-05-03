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
- `visual/classifier_paper_result.py`
- `visual/paper_classification_result.py`

Current result:

- `artifacts/plots/table1/first_work_visual_comparison_ce.png`
- `artifacts/plots/table1/first_work_visual_metrics_ce.json`
- `artifacts/plots/table1/first_work_paper_style_supplement_summary.png`
- `artifacts/plots/table1/first_work_paper_style_supplement_metrics.json`
- `artifacts/plots/table1/first_work_paper_style_article_summary.png`
- `artifacts/plots/table1/first_work_paper_style_article_metrics.json`

Run:

```powershell
.\.venv\Scripts\python visual\classifier_result.py --use-linear-frontend
```

Paper-style control:

```powershell
.\.venv\Scripts\python visual\classifier_paper_result.py --dataset-source article
```

Notes:

- `visual/classifier_result.py` is the CE-based local adaptation.
- `visual/classifier_paper_result.py` is the stricter Table 1 ramp/MSE control.
- The article text and the supplementary notebook disagree on how the Table 1
  dataset is generated, so the paper-style runner exposes both paths.
- After the Hopf equation correction, both paper-style paths remain far below
  the paper claim. The runner now records mean/sum, final-step, and template-MSE
  ramp accuracies; the best saved ramp readout is supplement final-step
  `test_acc = 0.61`.
- `visual/paper_classification_result.py` is the stricter complex-static-layer
  path for the printed architecture.

## Table 2

Code:

- `src/demodulation.py`
- `visual/demodulation_result.py`
- `visual/paper_sequence_result.py`

Current result:

- `artifacts/plots/table2/second_work_visual_comparison_fixed.png`
- `artifacts/plots/table2/second_work_visual_metrics_fixed.json`

Run:

```powershell
.\.venv\Scripts\python visual\demodulation_result.py --use-linear-frontend --use-input-skip
```

Notes:

- Table 2 code generates the synthetic demodulation dataset directly from the
  paper description.
- It does not depend on the misleading old `artifacts/amplitude_demodulation/*.npy`
  files.
- `visual/paper_sequence_result.py --task demodulation` is the stricter printed
  architecture path with two Hopf layers.

## Table 3

Code:

- `src/operators.py`
- `visual/operators_result.py`
- `visual/paper_sequence_result.py`

Current result:

- `artifacts/plots/table3/third_work_visual_summary.png`
- `artifacts/plots/table3/third_work_visual_metrics.json`

Run:

```powershell
.\.venv\Scripts\python visual\operators_result.py --use-linear-frontend
```

Notes:

- Table 3 uses the article-style formulas for both integration and differentiation.
- The summary figure overlays the DONN prediction and a simple numeric baseline
  to make the remaining gap obvious at a glance.
- `visual/paper_sequence_result.py --task integration|differentiation` is the
  stricter printed architecture path with two Hopf layers.

## Strict Reproduction

Code:

- `src/paper_donn.py`
- `visual/paper_classification_result.py`
- `visual/paper_sequence_result.py`

Docs:

- `docs/STRICT_REPRODUCTION.md`

Notes:

- This path implements the printed architectures with complex dense/static
  layers and split real/imag activations from Eq. (20).
- Current strict runs do not reproduce the paper scores, but they are the
  closest code path to the tables.

## Table 4

Code:

- `src/sentiment.py`
- `visual/sentiment_result.py`
- `visual/sentiment_paper_result.py`

Current result:

- `artifacts/plots/table4/fourth_work_visual_summary.png`
- `artifacts/plots/table4/fourth_work_visual_metrics.json`
- `artifacts/plots/table4/fourth_work_paper_exact_summary_4k3e.png`
- `artifacts/plots/table4/fourth_work_paper_exact_metrics_4k3e.json`

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

## Table 5

Code:

- `src/action_recognition.py`
- `visual/action_recognition_result.py`

Current result:

- `artifacts/plots/table5/fifth_work_ocnn_smoke_summary.png`
- `artifacts/plots/table5/fifth_work_ocnn_smoke_metrics.json`

Run:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/action_recognition_result.py
```

Notes:

- This is an OCNN architecture smoke control, not a strict UCF11 reproduction.
- The article appendix points to a public Kaggle UCF50 mirror, but no local
  UCF11/UCF50 video dataset is present in `artifacts/` or the checked external
  repositories.
- The saved JSON explicitly records `is_ucf11_reproduction=false`.

## Oscillator Ablation

Code:

- `visual/oscillator_ablation_result.py`

Current result:

- `artifacts/plots/ablation/oscillator_vs_baselines_summary.png`
- `artifacts/plots/ablation/oscillator_vs_baselines_metrics.json`

Run:

```bash
TF_CPP_MIN_LOG_LEVEL=2 uv run --python 3.11 --with tensorflow-cpu --with matplotlib python visual/oscillator_ablation_result.py
```

Notes:

- This compares current Hopf/DONN runs against simple local baselines.
- Current local conclusion: no broad empirical advantage for oscillator layers.
  Baselines win Table 1, Table 3, and Table 4; Table 2 demodulation is the one
  case where the current Hopf model wins against raw Conv1D, matched no-Hopf,
  and known-carrier coherent demodulation baselines.
- `docs/OSCILLATOR_ABLATION.md` gives the short interpretation.

## Case study 1

Code:

- `src/temporal_binding.py`
- `visual/temporal_binding_result.py`

Current result:

- `artifacts/plots/case_study/case_study_temporal_binding_summary.png`
- `artifacts/plots/case_study/case_study_temporal_binding_metrics.json`
- `artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_summary.png`
- `artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_metrics.json`

Run:

```powershell
uv run --python 3.12 --with numpy --with matplotlib python visual/temporal_binding_result.py
```

Corrected-generator control:

```powershell
uv run --python 3.12 --with numpy --with matplotlib python visual/temporal_binding_result.py --source generated-fixed --metrics-path artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_metrics.json --out-path artifacts/plots/case_study/case_study_temporal_binding_generated_fixed_summary.png
```

Notes:

- This is a dataset and temporal-binding-algorithm control for the moving-bar
  case study, not a full trained ConvOsc reproduction.
- The supplied moving-bar dataset has a generator/wraparound issue: the saved
  local audit finds fully blank frames in most samples.
- The fixed generator removes those blank frames while preserving the qualitative
  group-synchrony-above-residuary result in the local control.
- A direct deterministic classifier reads color and orientation from the videos
  with 100% accuracy, so the strict claim to test is hidden-layer synchrony, not
  the separability of the synthetic classes.

## Case study 2

Code:

- `src/stdp_kernel.py`
- `visual/stdp_kernel_result.py`

Current result:

- `artifacts/plots/case_study/case_study_stdp_kernel_summary.png`
- `artifacts/plots/case_study/case_study_stdp_kernel_metrics.json`
- `artifacts/plots/case_study/case_study_stdp_kernel_literal_control_summary.png`
- `artifacts/plots/case_study/case_study_stdp_kernel_literal_control_metrics.json`

Run:

```powershell
uv run --python 3.12 --with numpy --with matplotlib python visual/stdp_kernel_result.py
```

Literal diagnostic:

```powershell
uv run --python 3.12 --with numpy --with matplotlib python visual/stdp_kernel_result.py --hebbian-product literal-control --metrics-path artifacts/plots/case_study/case_study_stdp_kernel_literal_control_metrics.json --out-path artifacts/plots/case_study/case_study_stdp_kernel_literal_control_summary.png
```

Notes:

- This is an equation-level control for the Hopf-pair STDP case study.
- It is not an exact Fig. 7 reproduction because the article does not specify
  pulse shape and numerical integration parameters.

## Docs

- `docs/FILE_GUIDE.md` explains every working file in the repository in plain language.
- `docs/REPRO_AUDIT_MATRIX.md` tracks, experiment by experiment, what is already
  paper-faithful, what is only a local adaptation, and what is still not
  strictly reproduced.
- `docs/TABLE4_SENTIMENT.md` explains the Table 4 sentiment experiment, what is
  paper-faithful, and what is only a local practical adaptation.
- `docs/TABLE5_ACTION_RECOGNITION.md` explains the Table 5 action-recognition
  smoke control and the missing UCF data blocker.
- `docs/TEMPORAL_BINDING.md` explains the Case study 1 control and remaining
  gap to the paper's trained ConvOsc analysis.
- `docs/STDP_KERNEL.md` explains the Case study 2 equation-level control.
- `docs/FORMULA_AUDIT.md` tracks formula-by-formula checks against the paper.
- `docs/code/` contains separate Russian explanations for each Python code file.
