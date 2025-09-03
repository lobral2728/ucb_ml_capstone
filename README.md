### Photo Classification with ResNet50

**Author** 
Allen Long

#### Executive summary
This project performs **exploratory data analysis (EDA)** and builds a **baseline image-classification model** to determine whether a Microsoft 365 profile picture contains a **real human face** vs. **avatar** vs. **animal** imagery.  
In this module, the focus is on:
- Cleaning and organizing the dataset(s)
- Feature engineering where appropriate
- EDA visualizations to understand variables and relationships
- A single baseline model to serve as a comparison point for Module 24

While this project does not attempt to create the application that will read profile pictures, it does
provide the foundation model that could be used for that effort.

#### Rationale
Accurate identification of real human faces in corporate profile photos improves directory quality, compliance with internal policies, and downstream people-search experiences. Automating this classification reduces manual review burden and increases consistency across a large tenant.

#### Research Question
**Can we reliably distinguish real human-face profile images from non-human images using a lightweight, production-friendly baseline model?**  

Sub-question explored in EDA:
- Does the dataset represent many cultures and ages, and both sexes in a balanced way.

#### Data Sources
Three separate datasets were used to provide input to create a unique dataset for this projects. 

Planned/used sources (documented in notebooks):
- **Human faces:** - FairFace (diverse, labeled faces)
    - Repository: <https://github.com/joojs/fairface>
- **Avatar:** - Google Cartoon Set / “cartoon faces”
    - Kaggle dataset: <https://www.kaggle.com/datasets/brendanartley/cartoon-faces-googles-cartoon-set>
- **Animals** - Dogs vs Cats
    - Kaggle dataset: <https://www.kaggle.com/datasets/salader/dogs-vs-cats>

The final dataset looks like:<br>
```
    Train: 24,000 total -> {'human': 8000, 'avatar': 8000, 'animal': 8000}
    Val: 3,000 total -> {'human': 1000, 'avatar': 1000, 'animal': 1000}
    Test: 3,000 total -> {'human': 1000, 'avatar': 1000, 'animal': 1000}
```

#### Methodology
1. **Data loading & cleaning** (see [LoadDataset.ipynb](LoadDataset.ipynb)):
   - Ingest datasets into a common folder structure with `train/val/test` splits.
   - Deduplicate and remove unreadable or tiny images.
2. **EDA** (see [UCB_ML_Capstone.ipynb](UCB_ML_Capstone.ipynb)):
   - Class distribution and split verification.
   - Sample grids of each class.
3. **Feature engineering** (see [UCB_ML_Capstone.ipynb](UCB_ML_Capstone.ipynb)):
   - Basic augmentations using a Keras data_augmentation layer with RandomFlip, RandomRotation, RandomBrightness, and RandomContrast during training.
4. **Baseline model** (trained/evaluated in [UCB_ML_Capstone.ipynb](UCB_ML_Capstone.ipynb)):
   - **Approach:** Pretrained ResNet50 (frozen) as a feature extractor, with a small classification head (softmax) for three-way classification — human, avatar, and animal.
   - **Why:** Strong off-the-shelf features, quick to train, easy to deploy; serves as a fair, reproducible starting point for Module 24 comparisons.
5. **Evaluation (baseline):**
   - Accuracy, precision/recall/F1 (macro), confusion matrix.
   - Per-class recall to surface asymmetries.

> Note: In this project we **do not perform fine‑tuning** of the backbone (all ResNet layers remain non‑trainable). The accuracy was very good even without it.

#### Results
The training and validation accuracy for the frozen base model are extremely high, roughly 99.8–100% after the first couple of epochs. Early stopping happened at epoch 8 when 15 total epochs were set.
![Accuracy](images/accuracy.png)

- Accuracy: 0.9997 on 3,000 images (2,999/3,000 correct).
- Macro/weighted F1: 0.9997 — performance is uniformly high across classes.
![Confusion matrix](images/confusion_matrix.png)

The sample output demonstrates the ability of the model to classify humans, avatars (of human faces), and animals (cats, dogs).
![Sample Test Predictions](images/test_predictions_gallery.png)

#### Next steps
There are many opportunities to do additional analysis and modeling in the next round.
- The images were limited to humans, avatars, and pets (cats, dogs) for this assignment, but a profile picture could be anything. There is a need to test against a more diverse set of images and improve the dataset and model to handle those.
- There are other models to try, including light-weight CNNs like MobileNetV2, vision transformers, and other approaches.
- Look at hyperameters and tuning, including batch size, image size, unfrozen layers, regularization, and more.
- Finetuning
- Packaging for Azure (independent of this assignment). [Repo](https://github.com/lobral2728/azureprofileapp)

#### Outline of project
- [LoadDataset.ipynb with no output](LoadDataset.ipynb) — dataset ingestion, cleaning, splits, and data quality checks. The notebook created in Google Colab does not render in GitHub when there is output in it. This notebook shows the code.
- [LoadDataset.ipynb with output](output/loadDataset.ipynb) - This notebook shows output. You will need to clone the repo and look at the notebook with VS Code or a similar tool.
- [UCB_ML_Capstone.ipynb](UCB_ML_Capstone.ipynb) — EDA visuals, baseline model training/evaluation, and error analysis.

##### Contact and Further Information
For questions or collaboration, please contact **Allen Long** by filing an issue in the GitHub repo.
