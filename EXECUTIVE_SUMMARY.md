Executive Summary: Strep Throat Classification

Problem and Approach

This project addresses binary classification of throat images into strep-positive and strep-negative cases. Given a small dataset of 100 images (50 per class) and optional clinical symptom features, the goal was to design and evaluate a deep learning solution that demonstrates sound methodology under data constraints.

The solution uses transfer learning with a ResNet18 backbone pretrained on ImageNet, extracting 512-dimensional image embeddings. Two model variants were evaluated: (1) image-only classification, and (2) multimodal fusion combining image features with 7 binary clinical symptoms (hoarseness, rhinorrhea, sore throat, congestion, known recent contact, headache, fever) via concatenation before classification.

Rationale for Design Choices

Transfer learning was chosen because pretrained ResNet18 provides robust visual feature representations learned from large-scale natural image datasets. With only 100 training samples, training a CNN from scratch would be impractical. The ImageNet-pretrained features serve as a strong initialization that can be fine-tuned for the medical imaging domain.

Multimodal fusion was tested because clinical symptoms provide complementary information to visual evidence. In medical diagnosis, combining multiple data modalities often improves accuracy. The fusion approach uses simple concatenation of image embeddings (512-dim) and clinical features (7-dim), creating a 519-dimensional input to a two-layer fully connected classifier with dropout regularization.

Training Configuration

Both models were trained with identical hyperparameters to ensure fair comparison:
- Train/validation split: 80/20 (stratified, maintaining class balance)
- Epochs: 15
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Data augmentation: Random horizontal flips, rotations (±10°), color jitter
- Regularization: Dropout (0.5 and 0.3) in classifier layers

Hyperparameters were intentionally conservative to mitigate overfitting given the small dataset size.

Results

Image-only model:
- Accuracy: 60.0%
- Precision: 60.0%
- Recall: 81.82%
- F1: 69.23%
- ROC-AUC: 53.54%

Image + clinical features model:
- Accuracy: 55.0%
- Precision: 58.33%
- Recall: 63.64%
- F1: 60.87%
- ROC-AUC: 51.52%

Analysis

The image-only model outperformed the multimodal variant across all metrics. The high recall (81.82%) indicates good sensitivity for detecting positive cases, which is clinically important. However, the modest ROC-AUC values (0.51-0.54) reflect the challenge of learning from such limited data.

The lack of improvement from clinical features is likely due to several factors: (1) the extremely small dataset size (100 samples) creates high variance that masks potential benefits, (2) the binary symptom features are coarse and may not capture nuanced diagnostic information, and (3) adding parameters increases model complexity without sufficient data to learn effective feature interactions. This result demonstrates that additional features do not automatically improve performance, especially under severe data constraints.

Limitations

The primary limitation is dataset size. With only 100 samples, performance metrics have high variance and the model may not generalize beyond this specific dataset. The single train/validation split provides limited insight into true model performance. Clinical features are binary and lack granularity (e.g., severity, duration). There is no external validation cohort from a different institution or time period. The modest ROC-AUC values indicate the model performs only slightly better than random chance, which is expected given the data constraints.

Interpretability and Next Steps

Current interpretability is limited to confusion matrices and per-sample predictions with confidence scores. For production use, Grad-CAM visualizations would help identify image regions driving predictions, and feature importance analysis would clarify the contribution of clinical symptoms.

With more data or time, the following improvements would be valuable:
- Stratified k-fold cross-validation to better estimate performance variance
- Larger, more diverse dataset (1,000+ samples per class) from multiple sources
- Richer clinical features (severity scales, duration, patient demographics)
- Model interpretability tools (Grad-CAM, attention mechanisms)
- Calibration analysis to ensure predicted probabilities are well-calibrated
- External validation on independent datasets

Conclusion

This project demonstrates a methodical approach to medical image classification under small-data constraints. Transfer learning enabled meaningful learning from limited samples, and the comparison of image-only versus multimodal models provides insights into feature utility. The results highlight that feature engineering and model complexity must be balanced against available data, and that additional features are not always beneficial. While the current performance is modest due to data limitations, the framework provides a solid foundation for improvement with additional resources.
