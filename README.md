# UCB ML Capstone — Profile Photo Classifier (Human vs. Avatar vs. Animal/Other)

## Problem Statement
Many Office 365 profile photos are not real human faces (e.g., avatars, pets, objects, landscapes). For both the course capstone and an internal work use case, this project builds a supervised image classifier that assigns each profile image to one of three classes:

- `human` — a real human face
- `avatar` — a drawing/cartoon/emoji or otherwise synthetic depiction of a human face
- `animal` — animals or other non-human content (vehicles, landscapes, objects, etc.)

The goal is to automatically identify whether a profile picture is a real human face or not, with high precision/recall on the `human` class, while also separating avatars from other non-human images.

---

## Repository Overview
```
.
├── data/
│   └── final/
│       ├── train/
│       │   ├── avatar/
│       │   ├── human/
│       │   └── animal/
│       ├── val/
│       │   ├── avatar/
│       │   ├── human/
│       │   └── animal/
│       └── test/
│           ├── avatar/
│           ├── human/
│           └── animal/
├── LoadDataset.ipynb
├── UCB_ML_Capstone.ipynb
└── test_predictions_gallery.png  (generated artifact; gallery of test predictions)
```

### The Two Playbooks (Notebooks)
- **`LoadDataset.ipynb`** — Focuses on data loading and preparation.
  - Expects the above folder structure in `data/final/`.
  - Builds TensorFlow `tf.data` pipelines (shuffle → map preprocess → batch → prefetch).
  - Uses safe, deterministic shuffling with a bounded buffer and seeds for reproducibility.
  - Can optionally cache to disk for faster subsequent epochs without exhausting RAM.
  - Includes lightweight visualization utilities to preview samples by split and sanity-check labels.
  - Optionally checks for duplicate files across splits (via hashing) to reduce leakage risk.

- **`UCB_ML_Capstone.ipynb`** — Model training, evaluation, and reporting.
  - Initializes ResNet50 as a **frozen backbone** and attaches a small classification head (`GAP` → `Dropout` → `Dense(softmax)`).
  - **No fine-tuning**: the ResNet50 base remains frozen; only the classification head is trained.
  - Trains with callbacks (EarlyStopping, ReduceLROnPlateau) and saves best weights.
  - Records timing (start/end/elapsed) and training history.
  - Produces evaluation artifacts, including a predictions gallery and standard metrics.

Note: Data ingestion and model training are separated for clarity and reusability. The end-to-end pipeline can be re-run by opening the notebooks in order.

---

## Why ResNet50?
ResNet-50 strikes a strong balance between accuracy and efficiency for transfer learning on medium-scale image datasets. Its residual (“skip”) connections ease optimization in deep networks and consistently deliver robust features across diverse visual domains.

- Residual networks mitigate vanishing gradients and enable very deep models to train effectively.
- Pretraining on ImageNet provides general-purpose low- and mid-level features that transfer well to new tasks like profile photo classification.
- The architecture is widely supported in Keras/TensorFlow with mature tooling and documentation.

**Key references**
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385. https://arxiv.org/abs/1512.03385  
- Keras Applications — ResNet50 docs: https://keras.io/api/applications/resnet/#resnet50-function  
- Keras `preprocess_input` for ResNet: https://keras.io/api/applications/resnet/#preprocess_input-function  

Additional implementation references:
- TensorFlow `tf.data` guide: https://www.tensorflow.org/guide/data  
- `tf.data` performance & caching: https://www.tensorflow.org/guide/data_performance  
- Mixed precision training (Keras): https://keras.io/guides/mixed_precision/  
- Keras model training and callbacks: https://keras.io/api/models/model_training_apis/  

---

## Data
Images are organized in class-labeled folders (standard ImageNet-style directory layout), with pre-split train/val/test sets under `data/final/`. If you need to regenerate or verify splits, see `LoadDataset.ipynb` for the dataset creation and inspection utilities.

- **Classes:** `human`, `avatar`, `animal`
- **Input size:** 224×224 RGB (resized on the fly in the input pipeline)
- **Normalization:** Per the ResNet preprocessing function (`preprocess_input`) or an equivalent `Rescaling` layer (see the notebook for the exact setup).

---

## Model
The classifier is a ResNet-50 backbone (optionally initialized with ImageNet weights) plus a small head:

- `GlobalAveragePooling2D`  
- `Dropout` (regularization)  
- `Dense(3, activation="softmax")`

Training details:
- **Backbone is frozen — no fine-tuning in this repository.**
- Optimizer: Adam (default settings unless otherwise noted)
- Loss: Categorical or Sparse Categorical Crossentropy (depending on label encoding)
- Metrics: Accuracy; evaluation includes precision/recall/F1 per class
- Callbacks: EarlyStopping (on `val_loss`), ReduceLROnPlateau, best-weight saving

The notebooks enable mixed precision (if supported) and GPU memory growth for stability and speed.

---

## Results
Below is a sample of 24 test images with ground-truth label, model prediction, and class probabilities. This artifact is generated at the end of `UCB_ML_Capstone.ipynb`.

![Test Predictions Gallery](./test_predictions_gallery.png)

Model performance is summarized using accuracy and per-class metrics (precision/recall/F1) from `classification_report`, plus a confusion matrix for error analysis. Exact values will depend on your data snapshot and training seed; see the “Evaluation” section inside `UCB_ML_Capstone.ipynb` to reproduce.

Tip: For auditability, keep the saved training history and evaluation JSON/CSVs (if enabled in the notebook) under a versioned `artifacts/` directory in your repo.

---

## How to Run
1. Create and activate an environment with TensorFlow and Jupyter (GPU recommended).
2. Place data under `data/final/` following the structure shown above.
3. Open and run:
   - `LoadDataset.ipynb` — verify splits, inspect samples.
   - `UCB_ML_Capstone.ipynb` — train/evaluate the **frozen-backbone** ResNet-50 classifier and generate artifacts.

If you encounter out-of-memory errors, reduce `BATCH_SIZE` in the notebook and/or disable in-RAM caching (the notebooks default to disk caching when enabled).

---

## Reproducibility & Fairness Notes
- Seeds are set for Python/NumPy/TensorFlow to improve reproducibility, but non-determinism can still arise from GPU kernels and parallelism.
- Because profile photos reflect people from many cultures and contexts, be cautious about bias and dataset imbalance. Review per-class performance and consider augmentations or rebalancing as needed.

---

## Acknowledgments / Citations
- He et al., 2015. *Deep Residual Learning for Image Recognition.* https://arxiv.org/abs/1512.03385  
- Keras Applications — ResNet50: https://keras.io/api/applications/resnet/#resnet50-function  
- Keras `preprocess_input` for ResNet: https://keras.io/api/applications/resnet/#preprocess_input-function  
- TensorFlow `tf.data`: https://www.tensorflow.org/guide/data  
- `tf.data` performance & caching: https://www.tensorflow.org/guide/data_performance  
- Mixed precision (Keras): https://keras.io/guides/mixed_precision/  
- Keras Model Training & Callbacks: https://keras.io/api/models/model_training_apis/  
- scikit-learn `classification_report` / `confusion_matrix`: https://scikit-learn.org/stable/modules/model_evaluation.html
