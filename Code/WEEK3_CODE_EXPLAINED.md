# Week 3 Code Explained — Grad-CAM + Confusion Matrix / Fairness Breakdown

Companion to `WEEK2_CODE_EXPLAINED.md`. Covers notebook cells 96–110 line by line,
using the actual outputs from your run (test set = 3,553 images).

---

## Vocabulary you need first (defined once, used everywhere below)

- **Tensor** — just a multi-dimensional array of numbers. A color image is a 3D tensor
  `(height, width, channels)`. A batch of images is a 4D tensor `(batch, height, width, channels)`.
- **Gradient** — "if I nudge this number up slightly, how much does that other number change?"
  For a single input, this is calculus (a derivative). Neural nets compute gradients of the
  *loss* w.r.t. *weights* to learn. Grad-CAM reuses the same gradient machinery for a different
  purpose: explaining a prediction instead of updating weights.
- **`tf.GradientTape()`** — TensorFlow's gradient recorder. Anything computed *inside* the
  `with tf.GradientTape() as tape:` block gets logged, so afterward you can ask
  `tape.gradient(output, input)` — "how sensitive was this output to that input?" Outside the
  block, no history is kept and gradients can't be computed.
- **Batch dimension** — models are built to process many images at once for speed. Even when
  you only have ONE image, the model still expects a leading dimension of size 1:
  shape `(96, 96, 3)` must become `(1, 96, 96, 3)` before you can feed it in.
- **`argmax`** — "which index holds the biggest value?" Your model outputs 9 probabilities
  (one per age band); `argmax` picks the single most-likely class, e.g. index `4` → `"30-39"`.
- **Confusion matrix** — a grid that cross-tabulates *true* label (rows) against *predicted*
  label (columns). The diagonal = correct predictions; anywhere off the diagonal = a specific
  kind of mistake (e.g. "true 30-39, predicted 20-29").
- **Pivot table** — a spreadsheet-style summary: group rows by one column, split into more
  columns by another column, and aggregate (mean/count/sum) a third column into each cell.

---

## Cell 96 — Imports
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
```
`numpy` (arrays/math), `matplotlib` (plotting), `tensorflow` (the model + gradient engine),
`PIL.Image` (opening/resizing image files). Nothing new — same libraries you've used since Week 1.

---

## Cell 97 — Reload the trained model (Recovery cell)
```python
model    = load_model(MODEL_PATH)
train_df = pd.read_csv(...)
val_df   = pd.read_csv(...)
test_df  = pd.read_csv(...)
```
Your kernel restarted (or this is a fresh session), so nothing from Week 2 exists in memory.
`load_model` reads back the exact trained weights you saved to `age_classifier_baseline.keras` —
no retraining. The three `.csv` reads restore the dataframes (filepaths + labels), **not** the
loaded pixel arrays (`X_test`, `y_test` are gone — you'll rebuild those in Cell 106 only when needed).

---

## Cell 98 — Find the last convolutional layer (verification, not assumption)
```python
for layer in model.layers[-10:]:
    print(layer.name, layer.output.shape)
```
**Your actual output:**
```
block_16_project_BN     (None, 3, 3, 320)
Conv_1                  (None, 3, 3, 1280)
Conv_1_bn               (None, 3, 3, 1280)
out_relu                (None, 3, 3, 1280)     ← the one we want
global_average_pooling2d (None, 1280)           ← spatial info destroyed here
dropout                 (None, 1280)
dense                   (None, 128)
dense_1                 (None, 9)               ← final prediction
```
This confirms `out_relu` is the last layer that still has a `3×3` spatial grid (before
`GlobalAveragePooling2D` flattens it away). `None` in the shape just means "batch size —
could be any number of images." `1280` = number of filters/channels at that depth
(see the earlier chat explanation of why MobileNetV2 grows channel counts with depth).

---

## Cell 99 — Build the "tap" model
```python
grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
)
```
A normal `model.predict()` only returns the final 9-number prediction — everything in between
is discarded once computed. `grad_model` is the *same* network, but rigged to hand back two
things from one forward pass: the `(3,3,1280)` feature maps from `out_relu`, **and** the final
prediction. You need both simultaneously for Grad-CAM.

---

## Cell 100 — The core Grad-CAM function (the math)
```python
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    img_array = np.expand_dims(img_array, axis=0)          # (96,96,3) -> (1,96,96,3)
```
Adds the batch dimension every Keras model expects (see vocab above).

```python
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
```
Runs the image through `grad_model` *inside* the tape, so every operation gets recorded for
later differentiation. `training=False` tells `BatchNorm`/`Dropout` layers to behave as they
do at test time (fixed statistics, no random dropping) — matches how `model.evaluate()` behaved
during Week 2.

```python
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
```
`predictions` has shape `(1, 9)` — one row (the single image), 9 probabilities. If you didn't
specify which class to explain, default to the model's own top prediction. `class_channel` pulls
out just that ONE number — e.g. "the probability this is 30-39" — because that's the only
number Grad-CAM explains at a time.

```python
    grads = tape.gradient(class_channel, conv_outputs)
```
**This is the key line.** It asks: "for each of the 1280 feature maps, each 3×3 grid, how much
would nudging that value change `class_channel`?" Result: a gradient tensor the same shape as
`conv_outputs`, `(1, 3, 3, 1280)`.

```python
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
```
Averages away the batch and the two spatial (3×3) axes, leaving exactly `(1280,)` — one number
per channel. This is the **importance weight**: "how much does channel #732 matter for
predicting 30-39?"

```python
    conv_outputs = conv_outputs[0]                             # (3,3,1280)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]     # (3,3,1)
    heatmap = tf.squeeze(heatmap)                               # (3,3)
```
`conv_outputs[0]` drops the batch dim. The `@` (matrix multiply) computes, for every one of the
9 spatial positions, a weighted sum across all 1280 channels — channels that matter a lot
(high importance weight) contribute strongly, irrelevant ones contribute almost nothing.
Collapses `(3,3,1280)` down to a single `(3,3)` grid. `tf.squeeze` just removes the leftover
size-1 dimension from the matmul.

```python
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
```
`tf.maximum(heatmap, 0)` is ReLU — throws away negative values (evidence *against* the class,
which we don't care about here). Dividing by the max rescales everything into 0–1 so it displays
consistently as a heatmap regardless of the raw gradient magnitudes. `+ 1e-8` just prevents a
divide-by-zero crash if the max happened to be exactly 0.

```python
    return heatmap.numpy(), int(pred_index)
```
`.numpy()` converts from a TensorFlow tensor to a plain numpy array (easier for matplotlib).
Returns both the heatmap and which class it explained.

---

## Cell 101 — Single-image display helper
```python
def show_gradcam(filepath, grad_model, true_label=None):
    img = Image.open(filepath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
```
Loads and preprocesses **exactly** like training did — same resize, same `/255.0` scaling
(pixel values 0–255 → 0.0–1.0). Has to match, because the model only knows how to read images
prepared this way.

```python
    heatmap, pred_index = make_gradcam_heatmap(img_array, grad_model)
    heatmap_big = tf.squeeze(tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))).numpy()
```
Calls the Cell 100 function, then stretches the tiny `3×3` heatmap up to `96×96` using
`tf.image.resize` (smooth/bilinear upscaling, not blocky). `heatmap[..., tf.newaxis]` adds a
channel dimension because `tf.image.resize` expects `(H, W, channels)`.

```python
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_array); ...
    axes[1].imshow(img_array)
    axes[1].imshow(heatmap_big, cmap="jet", alpha=0.4)
```
Two side-by-side panels: the plain face, and the face with the heatmap drawn on top at 40%
opacity (`alpha=0.4`) so you can see both the face and the heat at once. `cmap="jet"` is the
blue→red colormap (blue = low importance, red = high).

---

## Cell 102 — Constants (redefine after kernel restart)
```python
AGE_GROUP_LABELS = ["0-2", "3-9", ..., "70+"]
IMG_SIZE = 96
SEED = 42
NUM_CLASSES = len(AGE_GROUP_LABELS)   # 9
```
Plain Python values, not saved by the checkpoint, so they need re-declaring in any fresh session.

---

## Cell 103 — First real test
```python
row = test_df.iloc[0]
show_gradcam(row["filepath"], grad_model, true_label=AGE_GROUP_LABELS[row["label"]])
```
Ran successfully — produced one figure with two panels. `test_df.iloc[0]` just grabs the first
row of your test dataframe as a sanity check.

---

## Cell 104 — Group comparison: Asian vs. Black (the fairness payoff)
```python
def show_gradcam_grid(df_subset, grad_model, group_name):
    ...
N_SAMPLES = 5
asian_sample = test_df[test_df["race"] == "Asian"].sample(N_SAMPLES, random_state=SEED)
black_sample = test_df[test_df["race"] == "Black"].sample(N_SAMPLES, random_state=SEED)
```
`test_df[test_df["race"] == "Asian"]` filters the dataframe down to only rows where the race
column equals `"Asian"`. `.sample(5, random_state=SEED)` randomly picks 5 of those rows, with
`random_state` fixing the randomness so re-running gives the same 5 faces every time (reproducibility).
The loop inside `show_gradcam_grid` runs Grad-CAM on each face and lays them out in one row per
group — this is your visual evidence for Risk R3 (does the model look at different regions for
different races?). Produced 2 figures (5 faces each), both ran without error.

---

## Cell 105 — Imports for the confusion-matrix section
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
```
`confusion_matrix` builds the true-vs-predicted grid. `seaborn` (`sns`) draws nicer heatmaps
than raw matplotlib. `to_categorical` turns an integer label (e.g. `4`) into a one-hot vector
(e.g. `[0,0,0,0,1,0,0,0,0]`) — the format your model was trained to output against.

---

## Cell 106 — Reload test images + regenerate predictions
```python
def load_images(dataframe, img_size):
    ...
X_test, y_test = load_images(test_df, IMG_SIZE)

y_pred_labels = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true_labels = np.argmax(y_test, axis=1)

test_eval = test_df.copy()
test_eval["y_true"] = y_true_labels
test_eval["y_pred"] = y_pred_labels
```
**Your actual output:** `Ready: (3553, 96, 96, 3) test images loaded and predicted` — confirms
all 3,553 test images loaded and matches the Week 2 test-set size exactly.

`load_images` re-does the same PIL-open → resize → `/255.0` steps as training, this time for
every row in `test_df`. `model.predict(X_test)` runs all 3,553 images through the model in one
batched call, returning `(3553, 9)` probabilities. `np.argmax(..., axis=1)` picks the winning
class per row → `y_pred_labels`, shape `(3553,)`. `y_test` (one-hot, from `load_images`) gets
un-one-hotted the same way into `y_true_labels`. `test_eval` is `test_df` with two new columns
bolted on — one row per test image, carrying filepath, race, true label, AND predicted label
together, which is what makes all the group-slicing below possible.

---

## Cell 107 — Overall confusion matrix
```python
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=range(NUM_CLASSES))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=..., yticklabels=...)
```
`confusion_matrix(true, pred)` returns a 9×9 grid of counts: row = true age band, column =
predicted age band, cell value = how many test images had that (true, predicted) combination.
`annot=True` prints the number inside each cell; `fmt="d"` formats it as an integer (not a
decimal). Ran and produced the figure — check whether the bright cells hug the diagonal
(good — confusions are between neighboring age bands) or scatter far off it (bad — random errors).

---

## Cell 108 — Confusion matrix, split by race
```python
for ax, race in zip(axes, races):
    subset = test_eval[test_eval["race"] == race]
    cm_race = confusion_matrix(subset["y_true"], subset["y_pred"], labels=range(NUM_CLASSES))
```
Same idea as Cell 107, but looped once per race, each on its own subplot (`ax`), using only
that race's rows from `test_eval`. Lets you visually compare error *patterns* (not just overall
accuracy) side by side — five 9×9 grids in one row.

---

## Cell 109 — Accuracy by age group × race (pivot table)
```python
test_eval["correct"]   = (test_eval["y_true"] == test_eval["y_pred"]).astype(int)
test_eval["age_group"] = test_eval["y_true"].map(lambda i: AGE_GROUP_LABELS[i])

accuracy_pivot = test_eval.pivot_table(
    index="age_group", columns="race", values="correct", aggfunc="mean"
).reindex(AGE_GROUP_LABELS)
```
`correct` is 1 where prediction matched truth, 0 otherwise. `age_group` converts the true label
back from an integer to its readable string (`4` → `"30-39"`). `pivot_table` then groups rows by
`age_group`, splits into columns by `race`, and takes the **mean** of `correct` in each cell —
since `correct` is only 0s and 1s, the mean of a group *is* its accuracy. `.reindex(...)` just
forces the rows into age order instead of alphabetical.

**Your actual output (partial):**
```
race       Asian  Black  Indian  Other  White
age_group
0-2        0.895  0.889   0.864  0.763  0.891
3-9        0.778  0.500   0.484  0.400  0.534
10-19      0.333  0.469   0.474  0.520  0.625
20-29      0.629  0.561   0.515  0.489  ...
```
**Worth noticing:** in the `10-19` row, Asian accuracy (0.333) is actually *worse* than Black
(0.469) — the opposite of the overall trend (Asian 55.6% > Black 42.5% overall). The race gap
isn't uniform across ages; it flips in at least one band. That's a genuinely useful, specific
finding — but see the count table below before treating it as solid.

---

## Cell 110 — Heatmap + sample-count honesty check
```python
sns.heatmap(accuracy_pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ...)
print(count_pivot)
```
Colors the same pivot table red (weak) to green (strong), `vmin=0, vmax=1` fixes the color scale
so a 0.50 always looks the same shade regardless of what other values are in the table.

**Your actual sample counts (partial):**
```
race       Asian  Black  Indian  Other  White
age_group
0-2         95.0    9.0    44.0   38.0   55.0
3-9         36.0    8.0    31.0   25.0  118.0
10-19        9.0   32.0    19.0   25.0  144.0
```
**This is the critical caveat.** `Black / 0-2` has only **9 test images**, `Black / 3-9` only
**8**. An accuracy computed from 8–9 images is close to a coin flip in reliability — that
`10-19` reversal above (Asian 0.333) is also sitting on just **9 Asian images**. **Don't report
those cells as solid findings without flagging the tiny n right next to them** — this is exactly
the "does the group state the limits of its own evidence" scrutiny a Responsible AI grader will
apply. The `20-29`+ rows, with counts in the 100s+ range, are far more trustworthy.

---

## What's next (not yet built)
- `pytest` test suite, ≥80% coverage
- Pseudo-model-card (this file's real numbers — especially the count-pivot caveat — belong there)
