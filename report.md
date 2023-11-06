
# Machine Learning Analysis for Loan Status Prediction: Model Comparison and Recommendations

## Overview of the Analysis

### Analysis Purpose:
The purpose of this analysis is to create a Logistic Regression Model that can effectively predict loan-related outcomes for future unseen data. The goal is to develop a model that can assist in making informed decisions about loan applications.

### Data Overview:
The data used for this analysis is about financial information related to loans and loan applicants situation. The variables we utilize include the loan amount, the interst rate, the borrower's income, the borrower's debt to income, the borrower's total number of accounts, and the borrower's derogatory marks. These variables serve as independent features, and our objective is to predict the loan status, which is the dependent variable.

### Understanding the Predictive Target:
The variable we were trying to predict in this analysis is the loan status, and it is a binary column where it takes the value 1 if the loan is considered a high-risk loan and 0 if the loan is considered a healthy loan. In our dataset, which consists of a total of 77,536 rows, we have 75,036 rows corresponding to healthy loan cases and 2,500 rows corresponding to high-risk loan cases. This indicates that our data is imbalanced, with a larger number of healthy loan cases compared to high-risk loan cases.

### Machine Learning Process Stages:
The stages of the machine learning process we went through as part of this analysis:

  1- Preprocess:

    Step 1: Read the dataset into a Pandas DataFrame.

    Step 2: Create the labels set and the features set.

    Step 3: Check the balance of the labels variable.

    Step 4: Split the data into training and testing datasets.

  2- Create a Logistic Regression Model with the Original Data:
    
    Step 1: Fit a logistic regression model by using the training data.

    Step 2: Save the predictions on the testing data labels by using the testing feature data and the fitted model.

    Step 3: Evaluate the model’s performance by calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

  3- Predict a Logistic Regression Model with Resampled Training Data to adjusts the number of majority and minority instances:

    Step 1: Resample the training data. 

    Step 2: Fit a Logistic Regression Model using the resampled data.
  
    Step 3: Save the predictions on the testing data labels by using the testing feature data and the fitted model.

    Step 4: Evaluate the model’s performance by calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

### Methods Employed:
In this analysis, we utilized several methods to facilitate the machine learning process:

- LogisticRegression: This method was employed to instantiate and implement a logistic regression model, which plays a pivotal role in predicting loan statuses.

- RandomOverSampler: The utilization of this method was essential for resampling the data, effectively addressing class imbalance, and ensuring the model's performance accuracy.

- fit method: This method enabled the adjustment of the model's parameters, allowing it to make accurate predictions based on the provided data.

- predict method: Finally, this method allowed the process of making predictions with the model, serving as the final step in the analysis, which ultimately assists in informed decision-making regarding loan applications.


## Results

Description of the balanced Accuracy, precision, and recall scores of all used Machine Learning Models:

* Machine Learning Model 1: Logistic Regression Model with the Original Data

  - Balanced Accuracy: This model reached a balanced accuracy score of 94%, indicating that the model maintains a strong accuracy for both High-risk Loan and Healthy Loan classes

  - Precision: The model correctly predicted 87% of actual positive cases (High-risk Loan), demonstrating its effectiveness in making accurate positive predictions. Furthermore, the model correctly predicted 100% of actual negative cases (Healthy Loan), indicating its perfect performance in making negative predictions.

  - Recall scores: The model correctly identified 89% of actual positive cases (High-risk Loan), demonstrating its effectiveness in capturing positive cases. Furthermore, the model correctly identified 100% of actual negative cases, indicating its perfect performance in identifying negative cases. 

* Machine Learning Model 2: Logistic Regression Model with Resampled Training Data

  - Balanced Accuracy: This model reached a balanced accuracy score of 99%, indicating that the model maintains a remarkable accuracy for both High-risk Loan and Healthy Loan classes

  - Precision: The model correctly predicted 87% of actual positive cases (High-risk Loan), demonstrating its ability in making accurate positive predictions. Furthermore, the model correctly predicted 100% of actual negative cases (Healthy Loan), indicating its perfect performance in making negative predictions. 

  - Recall scores: The model correctly identified 100% of actual positive cases (High-risk Loan), demonstrating its perfect performance in capturing positive cases for the High-risk Loan class. Furthermore, the model correctly identified 100% of actual negative cases (Healthy Loan), indicating its perfect performance in identifying negative cases for the Healthy Loan class. 

These scores offer valuable insights into the performance of each model, allowing the evaluation of their ability to predict loan outcomes accurately.

## Summary: Machine Learning Model Evaluation and Recommendations

### Performance Metrics Comparison: Model 1 vs. Model 2
* Performance Metrics for Model 1: Logistic Regression Model with the Original Data:

  - Balanced accuracy: 94%

  - Precision (High-risk Loan): 87%

  - Precision (Healthy Loan): 100%

  - Recall (High-risk Loan): 89%

  - Recall (Healthy Loan): 100%

* Performance Metrics for Model 2: Logistic Regression Model with Resampled Training Data:

  - Balanced accuracy: 99%

  - Precision (High-risk Loan): 87%

  - Precision (Healthy Loan): 100%

  - Recall (High-risk Loan): 100%

  - Recall (Healthy Loan): 100%

### Recommendation: Choosing the Best Model:
Based on the results provided, Model 2 is the superior choice, offering a higher balanced accuracy and perfect recall for both classes. 

### Consideration of Model Choice:
Model 2 is recommended for its superior performance. However, it's important to consider the specific objectives and priorities of the lending institution.
If the institution's primary goal is to minimize the risk of lending to high-risk applicants (predicting '1's), then Model 2 excels in capturing these cases. On the other hand, if maintaining an overall high level of accuracy for both high-risk and low-risk loans (predicting '0's) is a top priority, Model 2 also performs exceptionally well.
In summary, the choice of model should align with the institution's specific objectives, whether it's focused on predicting '1's or '0's, and it should be made while being mindful of potential overfitting risks.
