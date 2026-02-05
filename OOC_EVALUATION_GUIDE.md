# Out-of-Context (OOC) Evaluation Guide

## Overview

The Out-of-Context (OOC) evaluation feature enables testing whether the model truly learns temporal patterns or merely memorizes date-item co-occurrences. This addresses a critical issue in context-aware recommendation: **temporal data leakage**.

## The Problem: Temporal Data Leakage

### Current Issue
When date contexts are prepended to interaction sequences during training:
- The model sees: `[2014-Q3 context] → [item X]` co-occurrences
- At test time: Model may predict item X simply because it memorized "2014-Q3 → item X"
- This is **shortcut learning**, not genuine temporal understanding

### Why This Matters
- **Performance may be inflated** due to memorization
- **Model fails to generalize** to truly unseen temporal contexts
- **Evaluation is compromised** when test dates were seen during training

## OOC Solution

### Temporal Holdout Strategy
1. **Hold out entire quarter** (2014-Q3 by default) from training
2. Model **never sees** date contexts from this period during training
3. At evaluation, test prediction with **unseen date contexts**
4. Compare:
   - **IC (In-Context)**: Dates seen during training ✓
   - **OOC (Out-of-Context)**: Dates never seen during training ✗

### Expected Outcomes

**If memorization dominates:**
```
IC Performance:  High (e.g., NDCG@10 = 0.45)
OOC Performance: Low  (e.g., NDCG@10 = 0.20)
Gap = 0.25 → Strong evidence of memorization
```

**If true generalization exists:**
```
IC Performance:  High (e.g., NDCG@10 = 0.45)
OOC Performance: Similar (e.g., NDCG@10 = 0.42)
Gap = 0.03 → Model learned genuine patterns
```

## Usage

### 1. Enable OOC Evaluation

Add to your method config (e.g., `configs/method/base.yaml`):

```yaml
# Enable OOC evaluation
do_generalize_test: true

# Define temporal holdout period (default: 2014-Q3)
ooc_date_threshold: "2014-07-01"
ooc_date_end: "2014-10-01"
```

Or use the standalone config:
```bash
# Include OOC config in your Hydra command
python run.py +method/ooc_config=default method.do_generalize_test=true
```

### 2. Run Training

```bash
# Example: Enable OOC for Beauty dataset
python run.py \
    dataset=amazon \
    dataset.name=Beauty \
    method.do_generalize_test=true \
    method.ooc_date_threshold="2014-07-01" \
    method.ooc_date_end="2014-10-01"
```

### 3. Interpret Results

The logs will include additional metrics:

**Training/Validation:**
```
genret_in_val/NDCG@10:      0.35  # In-context validation (existing)
ooc_val/NDCG@10:            0.22  # Out-of-context validation (NEW)
```

**Testing:**
```
genret_in_test/NDCG@10:     0.40  # In-context test (existing)
ooc_test/NDCG@10:           0.25  # Out-of-context test (NEW)
ooc_dense_test/NDCG@10:     0.28  # OOC with dense retrieval (NEW)

Performance Gap = 0.40 - 0.25 = 0.15
```

## Data Split Behavior

### Without OOC (`do_generalize_test=false`)
```
Train:  All sequences up to second-to-last item
Val:    Second-to-last item (if seen in training)
Test:   Last item (if seen in training)
```

### With OOC (`do_generalize_test=true`)
```
Train:  Sequences with labels BEFORE 2014-07-01
Val:    Second-to-last item (labels < 2014-07-01)
Test:   Last item (labels < 2014-07-01)

OOC Val:  First 50% of samples with labels in [2014-07-01, 2014-10-01)
OOC Test: Last 50% of samples with labels in [2014-07-01, 2014-10-01)
```

**Key point:** OOC samples are completely separate from train/val/test.

## Quarterly Coverage Analysis

For Beauty dataset (2002-Q2 to 2014-Q3):
- Total interactions: 198,502
- 2014-Q3 interactions: ~6,041 (3.04%)
- Split: ~3,020 OOC Val, ~3,021 OOC Test

Run analysis:
```bash
python analyze_timestamp_coverage.py
```

## Advanced Configuration

### Custom Holdout Period

Test different quarters:
```yaml
# Hold out 2013-Q4 instead
ooc_date_threshold: "2013-10-01"
ooc_date_end: "2014-01-01"
```

### Multiple Dataset Support

Each dataset can have different thresholds in `configs/dataset/`:
```yaml
# configs/dataset/amazon.yaml
do_generalize_test: true
ooc_date_threshold: "2014-07-01"
ooc_date_end: "2014-10-01"

# configs/dataset/steam.yaml  
do_generalize_test: true
ooc_date_threshold: "2015-01-01"
ooc_date_end: "2015-04-01"
```

## Analysis Tools

### Compare IC vs OOC Performance

Use the analysis script to extract and compare metrics:

```bash
python analyze_ooc_results.py --log_dir ./results/tiger/
```

Expected output:
```
=== OOC Analysis Results ===
IC Test NDCG@10:     0.4023
OOC Test NDCG@10:    0.2547
Performance Gap:     0.1476 (36.7% drop)
Conclusion:          Strong evidence of date-item memorization
```

## Implementation Details

### Files Modified
1. **src/load_data.py**: Added OOC filtering and 50/50 split logic
2. **src/training.py**: Added OOC evaluation in `evaluate_helper()`
3. **configs/method/ooc_config.yaml**: Configuration template

### Non-Invasive Design
- All changes wrapped in `if do_generalize_test:` checks
- Existing code paths unchanged when flag is False
- Backward compatible with all existing experiments

## FAQ

**Q: Why 2014-Q3 specifically?**
A: It's the last quarter in Beauty dataset with sufficient samples (~6K interactions) while being recent enough to be representative.

**Q: Can I use this with dense-only models?**
A: Yes! OOC evaluation works with generative, dense, and hybrid (Liger) approaches.

**Q: What if my dataset doesn't have 2014-Q3?**
A: Adjust `ooc_date_threshold` and `ooc_date_end` to match your dataset's temporal range. Use `analyze_timestamp_coverage.py` to find suitable quarters.

**Q: How does this affect training time?**
A: Minimal impact. OOC samples are filtered during data loading and evaluated separately. No additional forward passes during training.

**Q: What's a "good" IC-OOC gap?**
A: 
- Gap < 5%: Excellent generalization
- Gap 5-15%: Moderate memorization  
- Gap > 15%: Strong memorization, revisit context design

## Citation

If you use OOC evaluation in your research, please consider citing:

```bibtex
@misc{liger_ooc_eval_2024,
  title={Out-of-Context Evaluation for Temporal Generalization in Recommender Systems},
  author={Your Name},
  year={2024},
  note={Implemented in Liger framework}
}
```

## Contact

For issues or questions about OOC evaluation:
- Open an issue in the repository
- Check existing discussions on temporal data leakage
