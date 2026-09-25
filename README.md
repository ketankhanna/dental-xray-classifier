# 🦷 Dental X-ray Classifier

Can a neural network spot problems in dental X-rays? A deep learning **team** term project (University of Toronto, SCS, 2024) comparing several approaches on the same data.

## The data
[Dental Radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) on Kaggle (CC BY-SA 4.0): panoramic X-rays with annotated regions in **4 classes**. Split used: 1,925 train · 204 validation · 117 test. The images are **not** included in this repo; download them from Kaggle (the notebooks show how with the Kaggle API; you need your own key).

## Notebooks

| Notebook | What it does | Test accuracy |
|---|---|---|
| [`full_project.ipynb`](full_project.ipynb) | The whole pipeline: cleaning the annotations, outlier removal, EDA, augmentation, VGG16 and ResNet50 transfer learning, and an improved custom model | 37.8% (improved model, 7 epochs) |
| [`custom_cnn_baseline.ipynb`](custom_cnn_baseline.ipynb) | Small CNN built from scratch (Keras Sequential) | **48.7%** |
| [`resnet50_transfer_learning.ipynb`](resnet50_transfer_learning.ipynb) | ResNet50 pre-trained on ImageNet, fine-tuned | 44.4% |
| [`vgg16_transfer_learning.ipynb`](vgg16_transfer_learning.ipynb) | VGG16 as a frozen feature extractor with a custom head | not recorded |

Random guessing across 4 classes would be about 25%.

## The honest takeaway
The big pre-trained models did **not** beat the small one. That result is the interesting part:
- **Too little data for a big model.** About 2,000 images isn't enough to fine-tune ResNet50's 23M+ parameters well.
- **Domain gap.** ImageNet features (dogs, cars) don't transfer cleanly to grayscale X-rays.
- **Low resolution and short training.** Shrinking images and training for only a few epochs throws away detail the models need.

Next steps I'd try: higher resolution, longer training with early stopping, class weighting, and a model pre-trained on medical images.

## Tech
Python · TensorFlow/Keras · ResNet50 · VGG16 · Kaggle API · Google Colab
