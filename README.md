# Pytorch_Mri_Tumor_Detection


Pytorch workflow to train (+ evaluation and inference) a CCN to classify brain tumors based on MRI with a few lines of python code (98% accuracy on never seen test data).
Configuration can be found at conf/config.yml

Classes and datasets:
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/classes_and_datasets.png)

Model summary:
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/model_summary.png)

Training metrics plotted in real time during training (Slightly raining loss inferior to val loss because validation dataset is not augmented)
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/training_metrics.png)
Confusion matrix
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/ConfusionMatrix.png)

Evaluation on test data (>98% accuracy)
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/evaluation.png)



Here are the results when applying the same transform/augment as the training dataset to the val dataset:

Training metrics plotted in real time during training
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/_with_val_augment/training_metrics.png)
Confusion matrix
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/_with_val_augment/ConfusionMatrix.png)

Evaluation on test data (>96.5% accuracy)
![alt text](https://github.com/ValentinOzeel/Pytorch_Mri_Tumors_Classification/blob/main/_for_readme/_with_val_augment/evaluation.png)