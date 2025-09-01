# Profile Photo Classifier

## Background and Problem Statement

Companies are encouraging their people to add their pictures into their Office365 profiles. Since many companis are large and geographically spread out, the primary mode of communication is Teams chats. Having a picture in your profile is a way to increase the familiarity and connection between people. Often, less than 50% have their picture up. There pictures of pets, cartoon faces, cars, landscapes, and many other creative expressions but not helpful in achieving the goal. Ultimately using a model to determine if a person has a human picture or a picture of something will create a cost-effective way to track if profiles have pictures of the employees.

### Expected Results

By utilizing a pretrained model and fine-tuning it with the dataset described above, I expect to be able to a multi-classifier that can detect when a picture is a human face. I also expect the results to be consistent across race, gender, and age. 

## Data Sources
We built the working dataset by combining **three sources referenced directly in `LoadDataset.ipynb`**:

- **Human faces — FairFace (Balanced Adults)**  
  Repository: <https://github.com/joojs/fairface>  
  Paper: <https://arxiv.org/pdf/2009.03224>  
  *(The notebook also includes direct Google Drive links for the FairFace image zips.)*

- **Avatars of human faces — Kaggle: Google Cartoon Set (rehost)**  
  Kaggle dataset: <https://www.kaggle.com/datasets/brendanartley/cartoon-faces-googles-cartoon-set>

- **Animals / “other” — Kaggle: Dogs vs Cats**  
  Kaggle dataset: <https://www.kaggle.com/datasets/salader/dogs-vs-cats>

Three classes will be used in this model:
- `human` — a real human face is present,
- `avatar` — stylized/cartoon/AI-generated depiction of a human face,
- `animal` — any non‑human class (we also map “other/objects/landscapes” into this bucket for enforcement simplicity).

```
.
├── data/
│   └── final/
│       ├── train/
│       │   ├── human/   ├── avatar/   └── animal/
│       ├── val/
│       │   ├── human/   ├── avatar/   └── animal/
│       └── test/
│           ├── human/   ├── avatar/   └── animal/
```

## Notebooks

- **LoadDataset.ipynb** (<https://github.com/lobral2728/ucb_ml_capstone/blob/main/LoadDataset.ipynb>)— Locates images in `data/final/`, builds deterministic stratified splits, and constructs an efficient `tf.data` input pipeline (decode -> resize with padding -> normalize -> optional augmentation). Includes quick EDA and visualization of class balance, sample images, and integrity checks (e.g., dedup across splits).
- **UCB_ML_Capstone.ipynb** (<https://github.com/lobral2728/ucb_ml_capstone/blob/main/UCB_ML_Capstone.ipynb>)— Defines, trains, and evaluates the model (ResNet50 backbone frozen; classification head trained). Exports metrics, a confusion matrix, classification report, and a 24‑image **test predictions gallery** with labels, predictions, and probabilities.

## Why ResNet50?
ResNet‑50 is a strong, widely validated backbone for image classification, with skip‑connections that enable training deeper models reliably. Features learned on large natural‑image corpora (e.g., ImageNet) transfer well to downstream tasks, especially in the early convolutional blocks. We use ResNet‑50 as a frozen **feature extractor** and train a lightweight head on top.
- He et al., “Deep Residual Learning for Image Recognition.” CVPR 2016 / arXiv:1512.03385.  
- Yosinski et al., “How transferable are features in deep neural networks?” NeurIPS 2014.  
- Keras Applications: ResNet / input preprocessing and expected shapes.

> Note: In this project we **do not perform fine‑tuning** of the backbone (all ResNet layers remain non‑trainable). This choice keeps training fast and stable and simplifies reproducibility. You can enable fine‑tuning later as an extension.

## Data

We combined **three labeled image sources** into a single, de‑duplicated dataset and then used the on‑disk stratified splits under `data/final/`.

> Replace the placeholders with your actual dataset names/links.
- **[Dataset A — Human faces]**: real human portraits/headshots from diverse demographics and capture conditions.
- **[Dataset B — Avatars / Synthetic faces]**: cartoons, drawings, emojis, stylized or AI‑generated depictions of human faces.
- **[Dataset C — Non‑human images]**: animals and “other” (landscapes, objects, vehicles, etc.).

**Consolidation workflow**
1. **Ingest & manifest.** Built a manifest CSV with `source, original_path, rel_path, class, split, sha1, phash`.
2. **Quality filters.** Removed unreadable/corrupt/tiny images; standardized to RGB.
3. **Label normalization.** Mapped source labels into `{human, avatar, animal}` via an explicit mapping table kept with the code.
4. **De‑duplication (exact and near‑duplicate).**  
   - Exact duplicates removed using **sha1**.  
   - Near‑duplicates flagged with **perceptual hash (pHash)** and culled preferentially from `val`/`test` to prevent leakage.
5. **Stratified splits.** Ensured class proportions are similar across `train` / `val` / `test` and wrote files to the directory structure above.
6. **Imbalance handling.** Preserved natural frequencies but used **class weights** during training.
7. **Resizing policy.** Resizing is **on‑the‑fly** in the input pipeline via `tf.image.resize_with_pad` to avoid distortion or content cropping; the stored images on disk are not destructively resized.

**Why these steps?**
- **De‑duplication** avoids inflation of metrics due to train/val/test leakage (near‑duplicates are known to bias benchmarks).  
- **Stratified splits** yield more stable evaluation across classes.  
- **On‑the‑fly resize with padding** preserves aspect ratio and matches the model’s expected input size (224×224 for ResNet‑50).  
- **Class weights** counter class imbalance without resampling the files on disk.

## Model and Training
- **Backbone:** `ResNet50` (`include_top=False`, pretrained on ImageNet), **frozen**.
- **Head:** GlobalAveragePooling → Dropout → Dense(3, softmax).  
- **Loss/Labels:** `SparseCategoricalCrossentropy` with `from_logits=False` (softmax outputs).  
- **Optimizer:** `Adam(learning_rate=1e-4)` (fast, robust convergence for the small head).  
- **Augmentation:** Keras preprocessing layers (random flips/rotations/color jitter) applied only on training batches.  
- **Mixed precision:** Enabled when supported (speeds up training and reduces memory use on modern GPUs).  
- **Checkpoints:** Save‑best‑only weights on the minimum validation loss; reload best weights before evaluation.  
- **Epochs:** 15 (early stopping on `val_loss` with patience, so actual epochs may be lower).  
- **Batch size:** 16 (fits a typical 8–12 GB GPU with 224×224 inputs and augmentation).

## Running
1. Consolidated data under `data/final/{train,val,test}/{human,avatar,animal}/`.
2. Run **LoadDataset.ipynb** to verify splits, view sample images, and build the `tf.data` pipeline.
3. Run **UCB_ML_Capstone.ipynb** to train/evaluate. The notebook will:
   - compute class weights from the train distribution,
   - train the classification head (backbone frozen),
   - save `resnet50_best.weights.h5` and `history_frozen.json`,
   - render `test_predictions_gallery.png` (24 test images with labels, predictions, and probabilities).

## Results (example)
Below is a sample of the exported gallery from the test set. Your run will reproduce a similar artifact.
![Test predictions gallery](test_predictions_gallery.png)

Key reported metrics include accuracy, per‑class precision/recall/F1, and a confusion matrix. See the notebook output for exact numbers for your data snapshot.

## Key References
- **ResNet** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR / arXiv:1512.03385. https://arxiv.org/abs/1512.03385
- **Transferability of features** — Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS. https://arxiv.org/abs/1411.1792
- **Keras Applications: ResNet** — https://keras.io/api/applications/resnet/
- **Transfer learning & fine‑tuning (Keras guide)** — https://keras.io/guides/transfer_learning/
- **TensorFlow tutorial: Transfer learning** — https://www.tensorflow.org/tutorials/images/transfer_learning
- **Mixed precision (TensorFlow Core guide)** — https://www.tensorflow.org/guide/mixed_precision
- **Data augmentation layers (Keras)** — https://keras.io/api/layers/preprocessing_layers/
- **Global average pooling (Keras)** — https://keras.io/api/layers/pooling_layers/global_average_pooling2d/
- **Dropout (Srivastava et al., 2014)** — https://jmlr.org/papers/v15/srivastava14a.html
- **Softmax + categorical cross‑entropy (Keras)** — https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class
- **Class weights (scikit‑learn)** — https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
- **De‑duplication & leakage** — Barz & Denzler (2020), *Do We Train on Test Data? Purging CIFAR of Near‑Duplicate Images.* https://arxiv.org/abs/1902.00423
- **Resize with padding (TensorFlow)** — https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad

## Notes
- We intentionally **do not fine‑tune** the backbone in this version to keep training fast and reproducible. Future work can unfreeze upper ResNet blocks and fine‑tune at a lower learning rate.
- Be mindful of privacy and compliance when handling user images. Ensure your usage conforms to policy and consent requirements.
