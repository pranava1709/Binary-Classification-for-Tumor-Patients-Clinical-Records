# Binary-Classification-for-Tumor-Patients-Clinical-Records

The dataset employed in this study comprises the clinical records of patients diagnosed with brain tumors. The primary parameters utilized in this dataset include age, MGMT index, WHO CNS grade, and whether a biopsy was performed prior to imaging. 
The target column of interest in this analysis is the survival outcome. The dataset was partitioned into a ratio of 80:20, with 80% of the data allocated to the training set and the remaining 20% assigned to the validation set.
![image](https://github.com/pranava1709/Binary-Classification-for-Tumor-Patients-Clinical-Records/assets/60814171/9e9f8e4f-8de2-4792-89d9-e376ce9bfd21)


The primary hypothesis of Logistic Regression is founded on an equation that incorporates both a slope and an intercept term. The equation can be expressed as follows: z equals the dot product of the weight vector w and the feature vector x, plus the bias term b.

The process of updating weights and biases occurs by utilizing the learning rate, a crucial hyper-parameter, together with the computed gradients as indicated in the code. The model underwent training for a total of 100 epochs, employing a threshold of 0.5. Predicted values exceeding this threshold were classified as positive, while those falling below were classified as negative.

The model underwent inference using the validation data, with the F1 score serving as the evaluation metric. The F1 score is a measure of performance that combines precision and recall, both of which are derived from the true positive, false positive, true negative, and false negative values shown in the confusion matrix.
![image](https://github.com/pranava1709/Binary-Classification-for-Tumor-Patients-Clinical-Records/assets/60814171/a03241f5-cc2d-41e2-90eb-3f12f8472aef)


The F1 score was around 77 percent.
