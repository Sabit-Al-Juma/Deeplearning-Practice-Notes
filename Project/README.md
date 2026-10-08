# 🐶🐱 Dog vs Cat Image Classifier

Binary image classification with TensorFlow/Keras. This project compares three approaches on the same dataset:

1. A **custom CNN** trained from scratch
2. **Transfer learning (feature extraction)** with a frozen MobileNetV2 backbone
3. **Full fine-tuning** of EfficientNetB0

Everything lives in one notebook, written for Google Colab.

---

## Dataset

- Source: [`laxmimerit/dog-cat-full-dataset`](https://github.com/laxmimerit/dog-cat-full-dataset) (cloned inside the notebook)
- Train folder: 19,989 images, 2 classes (cat, dog), split 80/20 into **15,992 train / 3,997 validation**
- Test folder: **5,000 images**
- Image size 224×224, batch size 32, seed 42

## Pipeline

- Loading with `tf.keras.utils.image_dataset_from_directory`
- Pixel rescaling to `[0, 1]` and `prefetch(AUTOTUNE)`
- Data augmentation: random horizontal flip, rotation (0.3), zoom (0.1), contrast (0.2)
- Early stopping on `val_loss` (patience 3, best weights restored)

## Models

| Model | Approach | Total params |
|---|---|---|
| Custom CNN | 3× Conv/MaxPool blocks → Dense(128) → sigmoid | 11,169,089 |
| MobileNetV2 | Frozen ImageNet backbone + new head (GAP → Dense 128 → Dropout 0.3 → sigmoid), Adam 1e-3 | 2,422,081 |
| EfficientNetB0 | All layers unfrozen, GAP → Dropout 0.3 → sigmoid, Adam 1e-5 | 4,050,852 |

## Results (validation set)

| Model | Train acc | Val acc | Train loss | Val loss |
|---|---|---|---|---|
| Custom CNN | 0.8720 | 0.8529 | 0.2895 | 0.3458 |
| Feature extraction (MobileNetV2) | 0.9910 | 0.9860 | 0.0251 | 0.0359 |
| Full fine-tuning (EfficientNetB0) | 0.9949 | **0.9905** | 0.0154 | 0.0266 |

Both pretrained models clearly beat the custom CNN. The train/validation gaps are small for all three, so there is no heavy overfitting.

> ⚠️ **Test-set metrics are not reported here.** The test set is currently fed to the models without the `Rescaling(1/255)` step that train/validation get. The models therefore see 0–255 pixel values and score about 50% (near chance). Fix this before quoting test numbers; see [Known issues](#known-issues).

## Project structure

```
.
├── Project_Dog_Cat_Image_Classifier_.ipynb   # full notebook
└── README.md
```

## Getting started

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (a GPU runtime is recommended).
2. Run the cells top to bottom. The first cell clones the dataset.

Or run locally:

```bash
pip install tensorflow matplotlib numpy pandas scikit-learn
jupyter notebook Project_Dog_Cat_Image_Classifier_.ipynb
```

## Known issues

- **Test set isn't normalized.** Apply the same rescaling as the other splits before evaluating, then re-run the comparison table:
  ```python
  test_ds = test_ds.map(lambda x, y: (normalize(x), y)).prefetch(AUTOTUNE)
  ```
- Train/val use the default integer labels, while the test set uses `label_mode="binary"`. Consider using `label_mode="binary"` for all three for consistency.
- The written analysis cell ("Which model performed best / is most efficient / would I deploy") still has `[placeholders]`. Fill them in once the corrected test numbers are available.

## Tech stack

Python · TensorFlow / Keras · scikit-learn · NumPy · pandas · Matplotlib · Google Colab

## Possible next steps

- Fix test normalization and report accuracy, precision, recall and F1 on the test set
- Add a confusion matrix
- Unfreeze only the top layers of the backbone (partial fine-tuning)
- Export the best model (`.keras`) and build a small Gradio demo
