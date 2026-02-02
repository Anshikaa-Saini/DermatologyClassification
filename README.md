# Dermatology Image Classification with Noisy Labels

This project addresses a real-world medical imaging problem where training labels are noisy while validation/test labels are clean. The goal is to build a robust classifier that generalizes well despite label corruption.

## Dataset
- Input: 28×28 RGB images
- Classes: 7
- Training data: Noisy labels
- Validation data: Clean expert-labeled

## Approach

1. **Exploratory Data Analysis (EDA)**
   - Verified image shapes, pixel distributions, and class imbalance.
   - Observed heavy dominance of one class and noise-induced instability in training accuracy.

2. **Baseline Model**
   - Built a lightweight CNN using PyTorch.
   - Observed low training accuracy but decent validation accuracy — typical behavior under noisy supervision.

3. **Noise-Robust Learning Strategy**
   - **Label Smoothing:** Prevents overconfidence on potentially incorrect labels.
   - **Confident Learning–style Sample Filtering:**
     - Warm-up training used to estimate per-sample losses.
     - High-loss samples treated as likely mislabeled and downweighted.
     - Model retrained on trusted samples using label smoothing.

4. **Results**
   - Validation accuracy improved compared to baseline.
   - Training accuracy remained low, indicating reduced memorization of noise.
     

## Running Inference on New Data

```python
from inference.inference import run_inference
run_inference("hidden_test.npz")
