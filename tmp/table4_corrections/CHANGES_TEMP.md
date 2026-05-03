# Temporary Table-4 Corrections

This directory is temporary and used only for correction experiments around
Table 4 paper reproduction.

## Files

- `tmp/table4_corrections/run_table4_variants_temp.py`
  - temporary runner that executes a sweep of plausible corrections
  - saves a sortable JSON with metrics for each variant
- `artifacts/plots/table4/fourth_work_temp_variants_results_1k1e.json`
  - quick screening on 1024/1024, 1 epoch
- `artifacts/plots/table4/fourth_work_temp_variants_results_4k3e_focus.json`
  - deeper comparison on 4096/4096, 3 epochs (focus subset)

## What Was Changed In Variants

Baseline (`baseline_pre_last_mse`) matches the current paper-control logic:

- padding: `pre`
- readout: last time step
- pad token handling: no explicit zeroing
- loss: `MSE` on one-hot labels

Corrections tested:

1. `pre_last_zeroPad_mse`
   - keeps baseline setup
   - multiplies embeddings by `(token != 0)` mask so pad positions feed zero

2. `pre_lastValid_mse`
   - readout changed from final index to final non-pad token index
   - no explicit pad-zero masking

3. `pre_lastValid_zeroPad_mse`
   - combines final non-pad readout + explicit pad-zero masking

4. `pre_meanValid_zeroPad_mse`
   - replaces readout with mean pooling across valid (non-pad) positions
   - keeps explicit pad-zero masking

5. `pre_lastValid_zeroPad_ce`
   - same architecture as (3) but loss switched from `MSE` to cross-entropy

6. `pre_meanValid_zeroPad_ce`
   - same architecture as (4) but loss switched from `MSE` to cross-entropy

7. `post_lastValid_zeroPad_mse`
   - padding switched to `post`
   - readout is final valid token index
   - explicit pad-zero masking enabled

8. `post_meanValid_zeroPad_ce`
   - padding switched to `post`
   - valid-token mean pooling
   - cross-entropy loss

## Commands Used

Quick full sweep:

```powershell
.\.venv\Scripts\python tmp\table4_corrections\run_table4_variants_temp.py --train-samples 1024 --test-samples 1024 --epochs 1 --batch-size 256 --out-path artifacts\plots\table4\fourth_work_temp_variants_results_1k1e.json
```

Deeper focused sweep:

```powershell
.\.venv\Scripts\python tmp\table4_corrections\run_table4_variants_temp.py --train-samples 4096 --test-samples 4096 --epochs 3 --batch-size 256 --variant-names baseline_pre_last_mse,pre_last_zeroPad_mse,pre_lastValid_zeroPad_mse,pre_lastValid_zeroPad_ce,post_lastValid_zeroPad_mse --out-path artifacts\plots\table4\fourth_work_temp_variants_results_4k3e_focus.json
```

## Current Outcome

- No tested correction reached paper-level behavior.
- Best focused result stayed near random:
  - `test_acc = 0.5154` (`baseline_pre_last_mse`)
  - `test_acc = 0.5154` (`pre_last_zeroPad_mse`)
- Other corrections remained in the same range or worse.

This means the tested fixes do not resolve the core Table-4 mismatch yet.

## V2 Variants (Stronger Attempts)

Added file:

- `tmp/table4_corrections/run_table4_variants_temp_v2.py`

What v2 adds beyond the first temp runner:

- explicit control of sequence length (`max_len`),
- independent `padding` and `truncating` modes,
- optional cross-entropy setup with classifier-style targets,
- additional readout mode `meanmax_valid` (concat of valid-token mean + max),
- optional learned real/imag input projections before each Hopf block,
- optional fixed-vs-trainable oscillator frequencies.

### V2 quick sweep command

```powershell
.\.venv\Scripts\python tmp\table4_corrections\run_table4_variants_temp_v2.py --train-samples 2048 --test-samples 2048 --epochs 2 --batch-size 256 --out-path artifacts\plots\table4\fourth_work_temp_variants_v2_results_2k2e.json
```

Top quick result:

- `v2_pre_meanmax_ce_300_proj`: `test_acc = 0.5249`

Saved in:

- `artifacts/plots/table4/fourth_work_temp_variants_v2_results_2k2e.json`

### V2 focused deeper command

```powershell
.\.venv\Scripts\python tmp\table4_corrections\run_table4_variants_temp_v2.py --train-samples 4096 --test-samples 4096 --epochs 3 --batch-size 256 --variant-names v2_baseline_pre_last_mse,v2_post_lastValid_ce_500,v2_post_meanmax_ce_300_proj_lr5e4 --out-path artifacts\plots\table4\fourth_work_temp_variants_v2_results_4k3e_focus.json
```

Focused deeper results:

- `v2_post_meanmax_ce_300_proj_lr5e4`: `test_acc = 0.5054`
- `v2_post_lastValid_ce_500`: `test_acc = 0.4946`
- `v2_baseline_pre_last_mse`: `test_acc = 0.4846`

Saved in:

- `artifacts/plots/table4/fourth_work_temp_variants_v2_results_4k3e_focus.json`

### V2 conclusion

- v2 produced small short-run gains on small data (`~0.525`), but
- under deeper 4096/4096 checks all tested variants still stay around random.
