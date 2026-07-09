# Week 2 — Every Line of Code Explained

> A complete walkthrough of the modelling section of `Responsible_Age_Classifier_Week1_2.ipynb`,
> starting at **Configuration**. For each cell: *what* the code is, *why* it exists, and *how* it
> is used downstream.
>
> Written for someone comfortable with classical ML / NLP but new to computer vision.
> Concepts you already know (softmax, dropout, class weights, train/test split) are noted
> briefly; the **CV-specific** parts are explained in full.

**Real numbers referenced throughout** (from the 6 July 2026 run):
`23,684` clean images → `16,578` train / `3,553` val / `3,553` test ·
`2,423,113` total parameters · `165,129` trainable in Stage 1 ·
final test accuracy **46.16%** · race accuracy-parity gap **13.1pp**

---

## Table of Contents

1. [Configuration — the hyperparameters](#1-configuration)
2. [Load Images — pixels into arrays](#2-load-images)
3. [Class Weights — fighting imbalance](#3-class-weights)
4. [Build the Model — MobileNetV2 + head](#4-build-the-model)
5. [Stage 1 — training with a frozen backbone](#5-stage-1-training)
6. [Training Curves — reading the plots](#6-training-curves)
7. [Stage 1 Evaluation](#7-stage-1-evaluation)
8. [Stage 2 — fine-tuning](#8-stage-2-fine-tuning)
9. [Final Evaluation](#9-final-evaluation)
10. [Per-Race Accuracy — the fairness core](#10-per-race-accuracy)
11. [Per-Race Bar Chart](#11-per-race-bar-chart)
12. [Per-Gender Accuracy](#12-per-gender-accuracy)
13. [FairFace Prediction Sampling](#13-fairface-prediction-sampling)
14. [FairFace Distribution Plot](#14-fairface-distribution-plot)
15. [Save Checkpoint](#15-save-checkpoint)
16. [Verify Checkpoint](#16-verify-checkpoint)

---

## Quick recap: what happened just before Configuration

Two things are already in place:

- **`df_utk_clean`** — a DataFrame with one row per face, carrying `filepath`, `age`,
  `gender`, `race`, `age_group`, and `label` (the integer 0–8 for the age band).
- **`train_df` / `val_df` / `test_df`** — a stratified 70/15/15 split of that DataFrame.
  `temp_df` was only a scratch bucket used to create val + test; it is never used again.

Critically, `test_df` **still carries the `race` and `gender` columns**. That is what makes the
fairness analysis possible later — the model never sees those columns, but the DataFrame
remembers them.

---

## 1. Configuration

```python
IMG_SIZE     = 96      # input resolution (CPU constraint — larger would be slow)
BATCH_SIZE   = 32
EPOCHS_S1    = 50      # stage 1: frozen backbone (EarlyStopping cuts this short)
EPOCHS_S2    = 30      # stage 2: fine-tuning
LR_STAGE1    = 1e-4
LR_STAGE2    = 1e-5    # much lower LR for fine-tuning — protects pretrained weights
SEED         = 42

print(f"Image size {IMG_SIZE} · Batch {BATCH_SIZE} · "
      f"Stage-1 epochs {EPOCHS_S1} · Stage-2 epochs {EPOCHS_S2}")
```

### What each one is

| Name | Value | Meaning |
|---|---|---|
| `IMG_SIZE` | 96 | Every image is resized to **96×96 pixels** before entering the model |
| `BATCH_SIZE` | 32 | The model sees 32 images, then updates its weights **once** |
| `EPOCHS_S1` | 50 | **Maximum** passes over the training data in Stage 1 |
| `EPOCHS_S2` | 30 | **Maximum** passes in Stage 2 |
| `LR_STAGE1` | 1e-4 | How big a step each weight takes per update, Stage 1 |
| `LR_STAGE2` | 1e-5 | Same, Stage 2 — deliberately **10× smaller** |
| `SEED` | 42 | Fixes randomness so results are reproducible |

### Why put them all in one cell?

Every value used to configure training lives here, so changing an experiment means editing
one cell instead of hunting through the notebook. This is standard practice and it makes the
work auditable — an examiner can see every knob at a glance.

### Why 96×96? (the most-asked question)

**The whole image is resized down to 96×96. Nothing is cropped.** `img.resize((96,96))`
shrinks the entire face; it does not cut a 96×96 window out of it.

No distortion occurs because both source datasets are already square:
UTKFace is 200×200, FairFace is 448×448.

| Resolution | Pixels | Relative compute cost |
|---|---|---|
| **96×96** | 9,216 | **1× (chosen)** |
| 128×128 | 16,384 | ~1.8× slower |
| 224×224 (MobileNetV2's native size) | 50,176 | ~5.4× slower |

Training runs on **CPU** (`tf.config.list_physical_devices('GPU')` returns `[]`), so 224×224
would turn hours into a day, and `X_train` would need ~5× the RAM.

**The cost, stated honestly:** downscaling 200×200 → 96×96 discards about **77%** of the pixel
information. The lost detail — fine wrinkles, skin texture — is exactly what separates `50-59`
from `60-69`. This is one real reason accuracy sits near 46%, and raising the resolution is the
first item on the Week-3 improvement list.

### Why two epoch counts and two learning rates?

Because training happens in **two stages with completely different jobs**:

- **Stage 1** — your classification head starts with **random** weights and knows nothing.
  It needs a normal learning rate (`1e-4`) and plenty of epochs (up to 50) to learn.
  The backbone is frozen, so there is nothing precious to damage.

- **Stage 2** — you unfreeze the top of the backbone, which holds **ImageNet-pretrained**
  weights learned from ~1.4M images. A large learning rate here would smash those weights
  (*catastrophic forgetting*). So the rate drops 10× to `1e-5`: tiny, gentle nudges. It is
  refinement, not learning from scratch, so 30 epochs suffice.

> **One-line intuition:** *Stage 1 teaches the new head while the backbone is protected.
> Stage 2 gently lets the backbone specialise, using a much smaller learning rate so it
> doesn't forget what it already knows.*

**Note the coincidence:** `EPOCHS_S2 = 30` and `layers[:-30]` both contain a `30`, but they are
completely unrelated. One counts **epochs (time)**; the other counts **layers (structure)**.

---

## 2. Load Images

```python
def load_images(dataframe, img_size):
    images, labels = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row["filepath"]).convert("RGB")
            img = img.resize((img_size, img_size))
            images.append(np.array(img) / 255.0)
            labels.append(row["label"])
        except Exception:
            continue          # unreadable file — skip
    return np.array(images), to_categorical(labels, num_classes=NUM_CLASSES)

X_train, y_train = load_images(train_df, IMG_SIZE)
X_val,   y_val   = load_images(val_df,   IMG_SIZE)
X_test,  y_test  = load_images(test_df,  IMG_SIZE)
```

This is the **most CV-specific function in the notebook**. It turns a table of file paths into
the numeric arrays a neural network can consume.

### Line by line

```python
images, labels = [], []
```
Two empty Python lists. `images` will collect pixel arrays; `labels` will collect integers 0–8.

```python
for _, row in dataframe.iterrows():
```
Loop over each row (= each face). `_` discards the pandas index, which we don't need.

```python
img = Image.open(row["filepath"]).convert("RGB")
```
- `Image.open(...)` opens the file from disk → a **PIL Image object** (not numbers yet).
- `.convert("RGB")` is a **defensive guard**. Images in the wild come in different modes:

| Mode | Channels | What would break |
|---|---|---|
| `RGB` | 3 | ✅ what we want |
| `L` (grayscale) | 1 | array shape `(96,96)` → model expects 3 channels → crash |
| `RGBA` | 4 | shape `(96,96,4)` → crash |
| `P` (palette) | 1 index | wrong format entirely |

Forcing RGB guarantees **exactly 3 channels every time**. One grayscale image hiding among
23,708 files would otherwise crash training.

```python
img = img.resize((img_size, img_size))
```
Shrinks the **whole** image to 96×96. A model requires a fixed input size — this is the visual
equivalent of padding every sentence to a fixed token length in NLP.

```python
images.append(np.array(img) / 255.0)
```
Two operations in one line:

1. **`np.array(img)`** — converts the PIL image into a NumPy array of shape **`(96, 96, 3)`**
   with integer values `0–255`, dtype `uint8`.
2. **`/ 255.0`** — rescales every value to `0.0–1.0`, producing dtype `float64`.

**What `(96, 96, 3)` means:** 96 rows × 96 columns × **3 colour channels**. Every pixel is three
numbers `[R, G, B]`.

**RGB** = Red, Green, Blue. Each 0–255 (`0` = none, `255` = maximum). Mixing them additively
produces any on-screen colour (all three at max = white).

Real example from `30_0_0_....jpg` in this project:

```
top-left pixel raw        : [163, 147, 134]     ← warm beige (R > G > B, typical skin tone)
top-left pixel normalized : [0.6392, 0.5765, 0.5255]
```

Picture the array as **three stacked 96×96 spreadsheets** — one for red, one for green, one for blue.

**Why divide by 255?** Neural networks train far better on small, uniformly scaled inputs. It is
the same reasoning as standardising tabular features before regression.

```python
labels.append(row["label"])
```
Grabs the integer age-band label from the DataFrame. Its provenance:

```
filename "30_0_0_..." → age = 30 → age_group = "30-39" → label = 4
                       (parsed)   (age_to_group)       (.map(AGE_GROUP_MAP))
```

So `labels` becomes a plain list of integers: `[4, 3, 8, 0, 3, 6, ...]`.

```python
except Exception:
    continue          # unreadable file — skip
```
If a file is corrupt or unreadable, skip that image rather than crash the whole run.

```python
return np.array(images), to_categorical(labels, num_classes=NUM_CLASSES)
```
- **`np.array(images)`** stacks all the individual `(96,96,3)` arrays into one 4-D array:
  **`(N, 96, 96, 3)`**. This is `X_train`.
- **`to_categorical(labels, 9)`** converts each integer into a **one-hot vector**. This is `y_train`.

**One-hot conversion** for `label = 4`:

```
4  →  [0, 0, 0, 0, 1, 0, 0, 0, 0]
       0  1  2  3  4  5  6  7  8
                   ↑
              "30-39"
```

**Why one-hot?** The model's output layer is `Dense(9, softmax)`, producing 9 probabilities. The
loss `categorical_crossentropy` compares those 9 predicted values against 9 true values, so the
target must also be a 9-length vector.

### The result

| Variable | Shape | Contents |
|---|---|---|
| `X_train` | `(16578, 96, 96, 3)` | pixel floats `0.0–1.0` |
| `y_train` | `(16578, 9)` | one-hot age bands |

**Notice what is absent:** no race, no gender, no filename, no exact age. The model only ever
receives pixels and a 9-slot answer. **That is the "race-agnostic by construction" claim,
visible in the data structure itself.**

### The one thing to watch out for

This cell is the **memory bottleneck**. `X_train` alone is roughly
`16,578 × 96 × 96 × 3 × 8 bytes ≈ 3.5 GB` (because `/255.0` produces `float64`). Adding val and
test pushes total RAM usage to several gigabytes. If the notebook dies here, insufficient RAM is
the cause — not a bug.

---

## 3. Class Weights

```python
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)
class_weight_dict = dict(enumerate(class_weights))

print("Class weights:")
for k, v in class_weight_dict.items():
    print(f"  Class {k} ({AGE_GROUP_LABELS[k]:5s}): {v:.2f}")
```

### The problem it solves

The real label distribution in UTKFace is badly skewed:

```
label 3 (20-29): 7,344 images   ← dominant
label 7 (60-69): 1,316 images   ← starved
label 8 (70+)  : 1,372 images   ← starved
```

Without intervention the model learns a lazy shortcut: *"predict 20-29 and be right often."*
Rare age bands get ignored. **This is Risk R5 in the risk analysis.**

### Line by line

```python
class_weight="balanced"
```
Tells scikit-learn to compute weights **inversely proportional to class frequency**. Rare classes
get large weights; common classes get small ones. The formula is roughly
`n_samples / (n_classes × count_of_this_class)`.

```python
classes=np.unique(train_df["label"])
```
The list of distinct labels present: `[0,1,2,3,4,5,6,7,8]`.

```python
y=train_df["label"]
```
The actual training labels, so it can count how many of each exist.

```python
class_weight_dict = dict(enumerate(class_weights))
```
`compute_class_weight` returns a plain array `[w0, w1, ... w8]`. Keras's `fit()` needs a
**dictionary** `{0: w0, 1: w1, ...}`. `enumerate` supplies the indices; `dict` builds the mapping.

### How it is actually used

Passed to `model.fit(..., class_weight=class_weight_dict)`. During training, the loss for each
sample is **multiplied by its class weight**, so mistakes on a 70+ face cost several times more
than mistakes on a 20-29 face. The gradient therefore pushes harder to get rare bands right.

**Honest limitation:** class weights reweight the loss, but they cannot invent detail that isn't
in the data. The 13.1pp fairness gap survived them — which the risk analysis states plainly
rather than hiding.

---

## 4. Build the Model

```python
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False          # Stage 1: freeze ALL base layers

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = Dropout(0.5)(x)
x      = Dense(128, activation="relu")(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary(show_trainable=True)
```

### The backbone

```python
base_model = MobileNetV2(input_shape=(96,96,3), include_top=False, weights="imagenet")
```

| Argument | Meaning |
|---|---|
| `input_shape=(96,96,3)` | Expect 96×96 RGB images |
| **`include_top=False`** | **Discard** MobileNetV2's original 1000-class ImageNet classifier. Keep only the feature extractor. You want 9 age bands, not "dog"/"car". |
| **`weights="imagenet"`** | **This is the transfer learning.** Load weights pretrained on ~1.4M ImageNet images. With `weights=None` you'd get a random network needing vastly more data and compute. |

You can see this in the download log when you first run it:
```
mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
                                                   ↑    ↑   ↑
                                             alpha=1.0  │   include_top=False
                                                   IMG_SIZE=96
```

**Why MobileNetV2 and not something bigger?**

| Model | Parameters | Verdict |
|---|---|---|
| **MobileNetV2** ✅ | ~2.3M | Lightweight, built for phones/edge devices with no GPU. Matches CPU-only training *and* is realistic for actual retail-camera hardware. |
| ResNet-50 | ~25M | ~7× heavier. Better accuracy, far too slow on CPU. |
| VGG16 | ~138M | Enormous, outdated. |
| EfficientNet / ViT | varies | Stronger, but need GPU and more data to shine. |

### Freezing

```python
base_model.trainable = False
```

**What freezing means:** each layer holds **weights** (learnable numbers). Setting
`trainable = False` **locks** them — gradient descent will not update them.

Crucially, **a frozen layer still works.** Data flows through it and it transforms that data
exactly as before. Only its internal numbers are pinned.

> **Analogy:** the network is an assembly line of ~155 stations. Each station has dials (weights)
> tuned by ImageNet training. Freezing = taping over the dials. The station still processes every
> item; you just cannot turn its knobs.

**Why freeze?** Those 2,257,984 backbone weights encode knowledge from 1.4M images — knowledge
you could never learn from 16,578 faces. Your head starts **random**, so its first predictions
are garbage and its gradients are large and meaningless. Letting those gradients into the
backbone would **destroy** the pretrained features in a few steps.

### The classification head

```python
x = base_model.output
```
The backbone's output — for a 96×96 input, a **`3 × 3 × 1280`** feature map (a stack of 1280
small 2-D grids).

```python
x = GlobalAveragePooling2D()(x)
```
**CV-specific.** Collapses each of the 1280 feature maps to a single average value → a flat
**1280-length vector**. This bridges the CNN's 2-D world to the 1-D `Dense` layers.

*Why not `Flatten`?* `Flatten` would give `3×3×1280 = 11,520` values, meaning far more parameters
in the next `Dense` layer and much more overfitting. Pooling is smaller and more robust.
**It has 0 parameters** — it's just an averaging operation.

```python
x = Dropout(0.5)(x)
```
During training, randomly zeros 50% of those 1280 values on every step. Strong regularisation —
the backbone is powerful and the dataset is small, so overfitting is a genuine risk.
**0 parameters** — it's an operation, not a learnable layer.

```python
x = Dense(128, activation="relu")(x)
```
A 128-neuron fully-connected "thinking layer" that recombines backbone features into
age-relevant signals.
**Parameters:** `1280 × 128 + 128 = 163,968`

```python
output = Dense(NUM_CLASSES, activation="softmax")(x)
```
The prediction layer: **9 neurons, one per age band**. `softmax` turns the raw scores into
probabilities that sum to 1.
**Parameters:** `128 × 9 + 9 = 1,161`

`NUM_CLASSES` is `9` because `AGE_GROUP_LABELS` has 9 bands. Those are FairFace's official age
buckets, deliberately **unequal in width** (`0-2` = 3 years, `20-29` = 10 years, `70+` open-ended)
because appearance changes fast in childhood and slowly in adulthood.

### Wiring it together

```python
model = Model(inputs=base_model.input, outputs=output)
```
`Model` is a **container class, not a model**. It says: *"treat the whole graph from the
backbone's input to my softmax as one trainable unit."* It adds no learning capacity — it defines
the wiring.

### Compiling

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```
- **Adam** — adaptive optimiser, the sensible default.
- **`categorical_crossentropy`** — the standard loss for multi-class one-hot classification.
- **`metrics=["accuracy"]`** — report accuracy each epoch (not used for optimisation, just display).

### Reading `model.summary(show_trainable=True)`

```
Total params:         2,423,113
Trainable params:       165,129   ← ONLY the head
Non-trainable params: 2,257,984   ← the FROZEN backbone
```

They add up: `165,129 + 2,257,984 = 2,423,113`. **That split *is* the freezing, expressed
numerically.** 93% of weights are locked; 7% are learning.

And the head arithmetic confirms it: `163,968 + 1,161 = 165,129` ✅

The `Trainable` column shows **`N`** for every backbone layer and **`Y`** for the two `Dense`
layers.

**On layer counts:** Keras counts every operation as a layer object — `Conv2D`,
`BatchNormalization`, `ReLU`, `Add`, `DepthwiseConv2D`. By that count MobileNetV2 is ~154 layers.
The literature's "53 layers deep" counts only layers **with weights**. Both are correct; they
measure different things. Check yours with `len(base_model.layers)`.

---

## 5. Stage 1 Training

```python
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_S1,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)
```

### EarlyStopping

| Argument | Meaning |
|---|---|
| `monitor="val_accuracy"` | Watch accuracy on the **validation** set (never train accuracy — that would encourage overfitting) |
| `patience=10` | If val accuracy doesn't improve for 10 consecutive epochs, stop |
| `restore_best_weights=True` | Roll back to the **best** epoch's weights, not the last one |

Without `restore_best_weights`, you'd keep whatever the model looked like when patience ran out —
which is 10 epochs *past* its peak.

### `model.fit(...)`

| Argument | What it does |
|---|---|
| `X_train, y_train` | The data the model learns from |
| `validation_data=(X_val, y_val)` | Evaluated after each epoch; **never** trained on |
| `epochs=EPOCHS_S1` | Up to 50 passes (EarlyStopping usually stops earlier) |
| `batch_size=BATCH_SIZE` | 32 images per weight update |
| `class_weight=class_weight_dict` | Applies the imbalance weights to the loss |
| `callbacks=[early_stop]` | Plugs in the early-stopping logic |
| `verbose=1` | Print a progress bar per epoch |

### What one "step" actually does

1. Take 32 images from `X_train`.
2. Push them through all ~155 layers → 32 predictions.
3. Compare against truth → average loss over the 32 (weighted by class weight).
4. Compute gradients: *which direction should each trainable weight move?*
5. **Update every trainable weight once**: `weight -= learning_rate × gradient`.

With 16,578 training images:
```
16,578 ÷ 32 ≈ 519 steps per epoch
```
In Stage 1, each epoch nudges those 165,129 head weights **519 times**. The 2,257,984 backbone
weights never move.

### `history`

The returned object stores per-epoch metrics — `history.history["accuracy"]`,
`["val_accuracy"]`, `["loss"]`, `["val_loss"]` — used for the plots next.

---

## 6. Training Curves

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, len(history.history["accuracy"]) + 1)

axes[0].plot(epochs_range, history.history["accuracy"],     label="Train")
axes[0].plot(epochs_range, history.history["val_accuracy"], label="Validation")
axes[0].set_title("Stage 1 — Accuracy per Epoch")

axes[1].plot(epochs_range, history.history["loss"],     label="Train")
axes[1].plot(epochs_range, history.history["val_loss"], label="Validation")
axes[1].set_title("Stage 1 — Loss per Epoch")
```

`epochs_range` is `1…N` where N is how many epochs **actually ran** (EarlyStopping may have cut
it short) — that's why it reads the length of the history rather than using `EPOCHS_S1`.

### How to read these plots (a likely exam question)

| Pattern | Diagnosis |
|---|---|
| Train ↑, Validation ↑ together | ✅ Healthy learning |
| Train ↑, Validation flat/↓ | ⚠️ **Overfitting** — memorising the training set |
| Both flat and low | ⚠️ Underfitting — model too weak, or LR too small |
| Validation **above** train | Normal here — `Dropout(0.5)` is active during training but **off** at validation, so validation looks better |

That last row is exactly what happened in this project, and the notebook says so: validation
accuracy sat slightly *above* training accuracy in Stage 1. It is a dropout artefact, not a bug.

---

## 7. Stage 1 Evaluation

```python
s1_loss, s1_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("=== STAGE 1 (FROZEN BACKBONE) TEST RESULTS ===")
print(f"Test Accuracy : {s1_accuracy:.2%}")     # ≈ 43.06%
print(f"Test Loss     : {s1_loss:.4f}")
```

`model.evaluate` runs the model over the test set **without training** and returns
`[loss, accuracy]`. `verbose=0` suppresses the progress bar.

**Important caveat, which the notebook states:** this is a **reference point only**. Training
continues afterwards (fine-tuning), so the *reported* headline number is the post-fine-tuning
evaluation. Model selection uses only validation data — the test set never influences a
training decision.

---

## 8. Stage 2 Fine-Tuning

```python
base_model.trainable = True                 # 1. unfreeze everything
for layer in base_model.layers[:-30]:       # 2. all layers EXCEPT the last 30
    layer.trainable = False                 #    ...re-freeze them

model.compile(                              # 3. recompile is MANDATORY
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop_ft = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)

history_finetune = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_S2,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop_ft],
    verbose=1
)
```

### The unfreeze, line by line

```python
base_model.trainable = True
```
Unlocks **all** backbone layers.

```python
for layer in base_model.layers[:-30]:
    layer.trainable = False
```
`[:-30]` is Python slicing: *"everything except the last 30 items."* So this re-freezes the first
~125 layers, leaving **only the last 30 trainable**.

### Why the *last* layers, not the first?

| Layers | What they learned from ImageNet | Need retraining for faces? |
|---|---|---|
| **Early** (1–125) | Edges, colours, simple textures | ❌ No — universal. An edge is an edge, cat or face. |
| **Late** (126–155) | High-level concepts: "dog snout", "car wheel" | ✅ Yes — must become "wrinkles", "jawline", "skin texture" |

You keep the generic foundation frozen and let only the top, task-specific layers adapt.

**Note:** `[:-30]` slices Keras **layer objects**, which include `BatchNormalization` and `ReLU`
(0 parameters each). So "the last 30 layers" is really ~10 actual convolutions plus their
BatchNorms and activations — a gentler unfreeze than it sounds.

### Why recompile?

```python
model.compile(...)
```
**This is not optional.** Keras builds its training graph at compile time. Changing `trainable`
flags after compiling has **no effect** until you recompile. Forgetting this is a classic bug —
the fine-tuning would silently do nothing.

The recompile also installs the new, 10× smaller learning rate `LR_STAGE2 = 1e-5`.

### Why a smaller learning rate?

The backbone's weights are precious. With `1e-4`, gradient updates would be large enough to
overwrite ImageNet knowledge (*catastrophic forgetting*). At `1e-5` the updates are small enough
to **specialise** the features toward faces without erasing them.

`patience=8` (down from 10) — Stage 2 converges faster, so less waiting is needed.

**After running this cell**, re-run `model.summary()` and you will see **Trainable params jump
well above 165,129** — visual proof the unfreeze took effect.

**Result:** test accuracy rises from ~43.06% → **46.16%**.

---

## 9. Final Evaluation

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("=== FINAL TEST SET RESULTS (after fine-tuning) ===")
print(f"Test Accuracy : {test_accuracy:.2%}")   # 46.16%
print(f"Test Loss     : {test_loss:.4f}")
```

**This is the headline baseline number.** It is trustworthy because the test set was used **zero**
times for any decision — not for early stopping, not for hyperparameter choice.

### Is 46% good?

Yes, for a baseline, and here is the defence:

- **9 classes → random chance is 11.1%.** The model is ~4× better than random.
- **Adjacent-class confusion dominates.** Predicting `30-39` for a 41-year-old scores as
  *completely wrong*, though it is off by one year. Many "errors" are near-misses.
- **CPU constrained** image size (96×96) and fine-tuning depth.
- Published UTKFace results with **GPU training and larger inputs** reach 60–70% on comparable
  band schemes — a realistic Week-3 target.

**And the point that matters most for this course:** overall accuracy is *not* the objective.
Whether accuracy is *consistent across race and gender* is. That is measured next.

---

## 10. Per-Race Accuracy

**This is the heart of the fairness analysis.**

```python
y_pred_labels = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true_labels = np.argmax(y_test, axis=1)

test_eval           = test_df.copy()
test_eval["y_true"] = y_true_labels
test_eval["y_pred"] = y_pred_labels

print("=== ACCURACY PER RACE (UTKFace Test Set) ===\n")
race_accuracies = {}
for race in sorted(test_eval["race"].unique()):
    subset = test_eval[test_eval["race"] == race]
    acc    = accuracy_score(subset["y_true"], subset["y_pred"])
    race_accuracies[race] = acc
    print(f"  {race:20s}: {acc:.2%}  ({len(subset)} images)")

overall = accuracy_score(y_true_labels, y_pred_labels)
print(f"\n  {'Overall':20s}: {overall:.2%}")

gap = max(race_accuracies.values()) - min(race_accuracies.values())
print(f"\n  Accuracy parity gap : {gap:.2%} "
      f"(best: {max(race_accuracies, key=race_accuracies.get)}, "
      f"worst: {min(race_accuracies, key=race_accuracies.get)})")
```

### Line by line

```python
y_pred_labels = np.argmax(model.predict(X_test, verbose=0), axis=1)
```
- `model.predict(X_test)` → shape `(3553, 9)`: nine probabilities per image.
- `np.argmax(..., axis=1)` → the **index of the largest probability** per row → shape `(3553,)`.

That index *is* the predicted age band. Example:
```
[0.01, 0.02, 0.08, 0.62, 0.19, 0.05, 0.02, 0.01, 0.00]  →  argmax = 3  →  "20-29"
```

```python
y_true_labels = np.argmax(y_test, axis=1)
```
Same trick in reverse: converts the one-hot truth back to integers.

```python
test_eval           = test_df.copy()
test_eval["y_true"] = y_true_labels
test_eval["y_pred"] = y_pred_labels
```

**This is the crucial move.** `X_test`/`y_test` are just pixels and labels — they lost all
metadata. But `test_df` is a **DataFrame that still carries `race` and `gender`**. By copying it
and attaching predictions, you can now slice accuracy by any demographic column.

```python
for race in sorted(test_eval["race"].unique()):
    subset = test_eval[test_eval["race"] == race]
    acc    = accuracy_score(subset["y_true"], subset["y_pred"])
```
Filter to one race → compute accuracy on just those rows → store it. Repeat for each race.

```python
gap = max(race_accuracies.values()) - min(race_accuracies.values())
```
**The named fairness metric: accuracy parity gap** — the difference between the best- and
worst-performing group. A perfectly fair model has a gap of `0`.

`max(dict, key=dict.get)` returns the *key* (the race name) with the highest value, not the value
itself — that's how the print statement names the best/worst group.

### The result, and why it is interesting

| Race | Accuracy | Test images |
|---|---|---|
| Asian | 55.62% | 507 |
| Indian | 48.00% | 575 |
| Other | 48.12% | 239 |
| White | 43.68% | 1,550 |
| Black | 42.52% | 682 |
| **Overall** | **46.16%** | 3,553 |

**Accuracy parity gap: 13.10 percentage points** (Asian best, Black worst).

**The bias direction was not what naive intuition predicts.** White is the *majority* training
class (~44% of images) yet lands *below* average, while Asian — a smaller class — performs best.
Candidate explanations for Week 3:

1. **Age-diversity effect** — White faces span the widest age range in UTKFace, making age
   prediction intrinsically harder for that group. Groups clustered in narrow age ranges are
   easier.
2. **Class-weight interaction** — class weights push toward rare age bands (0-2, 70+), which are
   disproportionately White, possibly trading majority-band accuracy for rare-band coverage.
3. **Sample-size reliability** — White's test subset is largest (1,550), so its estimate is most
   stable; small subsets (Other: 239) carry wide confidence intervals.

**The takeaway to say out loud:** per-group measurement revealed structure that could not have
been predicted from the training distribution alone. Assumptions about bias direction can be
wrong in *both* directions — which is precisely why Responsible AI demands **measurement over
assumption**.

### The key conceptual point

The model **never received race as an input**. Race lives only in the DataFrame, used *after*
prediction to slice results. Yet a 13.1pp gap exists — because a CNN can **reconstruct**
race-correlated signal from pixels. That is **Risk R3**, and it is why:

> *"Race-agnostic by **input**" ≠ "race-agnostic by **behaviour**."*

---

## 11. Per-Race Bar Chart

```python
races  = list(race_accuracies.keys())
accs   = list(race_accuracies.values())
avg    = np.mean(accs)
colors = ["coral" if a < avg else "steelblue" for a in accs]

plt.bar(races, accs, color=colors, edgecolor="black")
plt.axhline(y=avg, color="red", linestyle="--", label=f"Average: {avg:.2%}")
```

A list comprehension colours each bar **coral if below the average** (potential bias) and
steelblue otherwise. `axhline` draws the dashed average line. Under-performing groups become
visually obvious at a glance — the point of the chart.

---

## 12. Per-Gender Accuracy

```python
gender_accuracies = {}
for gender in sorted(test_eval["gender"].unique()):
    subset = test_eval[test_eval["gender"] == gender]
    acc    = accuracy_score(subset["y_true"], subset["y_pred"])
    gender_accuracies[gender] = acc

g_gap = max(gender_accuracies.values()) - min(gender_accuracies.values())
```

**Identical slicing logic**, grouped by `gender` instead of `race`.

**Why bother, given gender is balanced in the training data?** Because the project goal is
*gender-neutral*, and balance in the input does **not** guarantee parity in the output. Fairness
must be **measured, not assumed** — the same principle that made the race result surprising.

---

## 13. FairFace Prediction Sampling

```python
N_PER_RACE = 200

sample_rows = []
for race in sorted(df_fairface_clean["race"].unique()):
    race_sample = df_fairface_clean[df_fairface_clean["race"] == race] \
                      .sample(n=N_PER_RACE, random_state=SEED)
    sample_rows.append(race_sample)

df_ff_sample = pd.concat(sample_rows).reset_index(drop=True)

ff_images = []
for fp in df_ff_sample["filepath"]:
    img = Image.open(fp).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    ff_images.append(np.array(img) / 255.0)
ff_images = np.array(ff_images)

ff_preds = np.argmax(model.predict(ff_images, batch_size=BATCH_SIZE, verbose=0), axis=1)
df_ff_sample["prediction"] = ff_preds
```

### Why FairFace at all?

FairFace is **race-balanced across 7 groups**, making it an ideal bias-audit set. But — and this
is **Risk R8** — the Kaggle mirror used here has **no age labels**, only race folders.

**Consequence:** you cannot compute accuracy on FairFace. There is no ground truth to compare
against. So instead you measure **prediction-distribution consistency**:

- If the model is fair → all races show a **similar** predicted-age distribution.
- If the model is biased → one race is systematically predicted **older or younger**.

That is a **bias signal, not a correctness measure** — and the notebook says so explicitly.

### Line by line

```python
.sample(n=N_PER_RACE, random_state=SEED)
```
Takes exactly 200 random images **per race** — equal counts, so no race dominates the plot.
`random_state=SEED` makes the sample reproducible.

```python
df_ff_sample = pd.concat(sample_rows).reset_index(drop=True)
```
Stacks the 7 per-race samples into one DataFrame (7 × 200 = 1,400 rows) and renumbers the index.

```python
img = Image.open(fp).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
ff_images.append(np.array(img) / 255.0)
```
**Exactly the same preprocessing as training** — resize to 96×96, normalise to 0–1. This is
essential: a model must see test data preprocessed identically to training data, or predictions
are meaningless.

```python
ff_preds = np.argmax(model.predict(ff_images, batch_size=BATCH_SIZE, verbose=0), axis=1)
```
**One batched `predict` call** for all 1,400 images, rather than 1,400 separate calls. Far faster
on CPU. `argmax` again converts probabilities → the predicted band index.

Note there is **no `y_true`** here, and no `accuracy_score`. There cannot be.

---

## 14. FairFace Distribution Plot

```python
for race in sorted(df_ff_sample["race"].unique()):
    race_preds = df_ff_sample[df_ff_sample["race"] == race]["prediction"] \
                     .value_counts().sort_index()
    plt.plot(race_preds.index, race_preds.values, marker="o", label=race)

plt.xticks(range(NUM_CLASSES), AGE_GROUP_LABELS, rotation=30)
```

For each race: count how many of its 200 images landed in each predicted band
(`value_counts()`), sort by band index (`sort_index()`), and draw a line.

`plt.xticks(range(9), AGE_GROUP_LABELS)` replaces the numeric x-axis `0…8` with readable band
names.

### How to read it

- **Consistent (fair):** all 7 lines peak at the same age band and follow similar shapes.
- **Biased:** one race's line peaks noticeably earlier (predicted younger) or later (predicted
  older) than the others — systematic directional bias.

### Stated limitations (which the notebook does not hide)

1. **No ground truth.** This measures *consistency*, not *correctness*. If the true age
   distributions genuinely differ between race folders, some divergence is legitimate.
2. **Sample of 200 per race** keeps CPU runtime manageable, so confidence in small differences
   is limited.
3. Exact per-race correctness comes from the **UTKFace test set** analysis, not from here.

---

## 15. Save Checkpoint

```python
model.save(MODEL_PATH)
train_df.to_csv(str(WORK_DIR / "train_df.csv"), index=False)
val_df.to_csv(str(WORK_DIR / "val_df.csv"),   index=False)
test_df.to_csv(str(WORK_DIR / "test_df.csv"),  index=False)
print("All saved to", WORK_DIR)
```

- **`model.save(MODEL_PATH)`** — writes the full model (architecture + all 2,423,113 weights) to
  a `.keras` file. Training this takes hours on CPU; reloading takes seconds.
- **Saving the three split CSVs** guarantees **reproducibility**: reloading gives you the *exact*
  same train/val/test rows. Without this, re-running `train_test_split` on different data (or a
  different seed) could put a training image into the test set — silently invalidating every
  number in the notebook.

`index=False` omits pandas' index column so the CSV stays clean.

`MODEL_PATH` and `WORK_DIR` come from the environment-agnostic Path Configuration cell, so this
works identically in the cloud and locally.

---

## 16. Verify Checkpoint

```python
files_to_check = {
    "Model"     : MODEL_PATH,
    "Train CSV" : str(WORK_DIR / "train_df.csv"),
    "Val CSV"   : str(WORK_DIR / "val_df.csv"),
    "Test CSV"  : str(WORK_DIR / "test_df.csv"),
}

print("=== FINAL CHECKPOINT ===\n")
for name, path in files_to_check.items():
    status = "✅ saved" if os.path.exists(path) else "❌ MISSING"
    print(f"{name:12s}: {status}")
```

A simple `os.path.exists` check on each file. Cheap insurance — a silent save failure would only
surface later when the recovery cell can't find the model.

---

## The single most important idea in Week 2

Trace what the model actually receives:

```
X_train.shape = (16578, 96, 96, 3)   ← pixels only
y_train.shape = (16578, 9)           ← one-hot age band only
```

**No race. No gender. No age number. No filename.**

The model is therefore **race-agnostic by construction** — it is structurally incapable of using
race as an input, because race never enters `model.fit()`.

Race and gender live only in the pandas DataFrame (`test_df`), and are used **after** prediction
to *slice* the results (cells 10 and 12 above). That is how you audit fairness without ever
feeding protected attributes to the model.

And yet the measured gap is **13.1 percentage points** — because a CNN can reconstruct
race-correlated signal from pixels alone. Naming that honestly, measuring it, and planning
Grad-CAM to investigate it (Week 3) is what makes this a *Responsible AI* project rather than
just an age classifier.

---

*Companion document to `Responsible_Age_Classifier_Week1_2.ipynb`.
For the regulatory analysis see `regulatory_analysis.md`; for the risk matrix see the
notebook's Week 2 Risk Analysis section.*
