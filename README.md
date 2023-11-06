# credit-risk-classification

In this Challenge, we will use various techniques to train and evaluate a model based on loan risk. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## What we will learn from this project:

- How to split the data into training and testing sets

- How to create a Logistic Regression Model with the Original Data

- How to predict a Logistic Regression Model with Resampled Training Data

- How to write a credit risk analysis report

## Instructions:

* Split the Data into Training and Testing Sets:
  
  - Read the data into a Pandas DataFrame.

  - Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

* Create a Logistic Regression Model with the Original Data:
  
  - Fit a Logistic Regression model by using the Training Tata (X_train and y_train).

  - Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

  - Evaluate the model’s performance.

* Predict a Logistic Regression Model with Resampled Training Data:

   - Resample the training data. 

   - Fit a Logistic Regression Model using the resampled data.
   
   - Save the predictions on the testing data labels by using the testing feature data and the fitted model.
   
   - Evaluate the model’s performance.

* Write a Credit Risk Analysis Report

  The Anakysis Report contains: 
  
  - An overview of the analysis

  - The results

  - A summary

## Program:

### Tools:

- Visual Studio Code (VSCode): is a free, open-source code editor developed by Microsoft.

- Python: is a high-level programming language known for its simplicity, readability, and versatility. 

- Pandas: is a Python library for data manipulation and analysis.

- sklearn: is a Python library providing a wide range of tools for machine learning.

- imblearn: is a Python library used for handling imbalanced datasets in machine learning. 

### Code:

#### Functions Defined:

##### Instantiates a Logistic Regression model and fits the model 
```
def logic_reg_model(X, y):
    """
    Create and train a Logistic Regression model.

   Input:
        X (pd.Dataframe): The feature data for training the model.
        y (pd.Series): The target labels for training the model.

    Output:
        sklearn.linear_model.LogisticRegression: A trained Logistic Regression model.
         
    """
    # Instantiate the Logistic Regression model
    model = LogisticRegression(random_state=rs)
    # Fit the model 
    model.fit(X, y)
    return model
```

##### Counts the distinct values of the label
```
def count_distinct_values(series):
    """
    Count the distinct values in a Pandas Series and display the results.

    Input:
        series (array): The data to count distinct values.

    Output:
        value_counts (array): The distinct values and their counts.
        
    """
    # Count the distinct values
    value_counts = series.value_counts()
    
    # Display the distinct values
    print("Distinct values:")
    print(value_counts)
    
    #Return the count
    return value_counts
```

##### Calculate the balanced accuracy score
```
def balanced_accuracy(y_true, y_pred):
    """
    Calculate and return the balanced accuracy score for a classification model.

    Input:
        y_true (array): True target values.
        y_pred (array): Predicted target values from a classification model.

    Output:
        balanced_accuracy (float): The balanced accuracy score.

    """
    # Calculate the balanced accuracy score of the model
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    
    # Return the balanced accuracy score
    return balanced_accuracy
```

#####  Generate a confusion matrix
```
def conf_matrix(y_true, y_pred, model_name):
    """
    Generate and display a confusion matrix for a classification model.

    Input:
        y_true (array): True target values.
        y_pred (array): Predicted target values from a classification model.
        model_name (string): Name or identifier of the classification model.

    Output:
        confusion (array): The confusion matrix as a 2D array.

    """
    # Generate the confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    
    # Print the result
    print(f"Confusion Matrix for {model_name}:")
    
    # Retutn the confusion matrix
    return confusion
```

##### Generate a classification report
```
def class_report(y_true, y_pred, model_name, target_names):
    """
    Generate and display a classification report for a classification model.

    Input:
        y_true (array): True target values.
        y_pred (array): Predicted target values from a classification model.
        model_name (string): Name or identifier of the classification model.
        target_names (list): List of class labels.

    Output:
    Print the Classification Report for Logistic Regression
    
    """
    
    # Generate a classification report
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    # Display the result
    print(f"Classification Report for {model_name}:")
    print(report)
```

#### Main Code

##### Split the data into training and testing datasets:
```
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=rs,
                                                    stratify=y 
                                                    )
X_train.shape
```

##### Save the predictions
```
y_pred_train = model_training.predict(X_test)
```

##### Resample the data
```
# Instantiate the random oversampler model
ros = RandomOverSampler(random_state=rs)

# Fit the original training data to the random_oversampler model
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
len(X_resampled)
```

## credit risk analysis report
### "Machine Learning Analysis for Loan Status Prediction: Model Comparison and Recommendations

#### Overview of the Analysis

##### Analysis Purpose:
The purpose of this analysis is to create a Logistic Regression Model that can effectively predict loan-related outcomes for future unseen data. The goal is to develop a model that can assist in making informed decisions about loan applications.

##### Data Overview:
The data used for this analysis is about financial information related to loans and loan applicants situation. The variables we utilize include the loan amount, the interst rate, the borrower's income, the borrower's debt to income, the borrower's total number of accounts, and the borrower's derogatory marks. These variables serve as independent features, and our objective is to predict the loan status, which is the dependent variable.

##### Understanding the Predictive Target:
The variable we were trying to predict in this analysis is the loan status, and it is a binary column where it takes the value 1 if the loan is considered a high-risk loan and 0 if the loan is considered a healthy loan. In our dataset, which consists of a total of 77,536 rows, we have 75,036 rows corresponding to healthy loan cases and 2,500 rows corresponding to high-risk loan cases. This indicates that our data is imbalanced, with a larger number of healthy loan cases compared to high-risk loan cases.

##### Machine Learning Process Stages:
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

##### Methods Employed:
In this analysis, we utilized several methods to facilitate the machine learning process:

- LogisticRegression: This method was employed to instantiate and implement a logistic regression model, which plays a pivotal role in predicting loan statuses.

- RandomOverSampler: The utilization of this method was essential for resampling the data, effectively addressing class imbalance, and ensuring the model's performance accuracy.

- fit method: This method enabled the adjustment of the model's parameters, allowing it to make accurate predictions based on the provided data.

- predict method: Finally, this method allowed the process of making predictions with the model, serving as the final step in the analysis, which ultimately assists in informed decision-making regarding loan applications.


#### Results

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

#### Summary: Machine Learning Model Evaluation and Recommendations

##### Performance Metrics Comparison: Model 1 vs. Model 2
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

##### Recommendation: Choosing the Best Model:
Based on the results provided, Model 2 is the superior choice, offering a higher balanced accuracy and perfect recall for both classes. 

##### Consideration of Model Choice:
Model 2 is recommended for its superior performance. However, it's important to consider the specific objectives and priorities of the lending institution.
If the institution's primary goal is to minimize the risk of lending to high-risk applicants (predicting '1's), then Model 2 excels in capturing these cases. On the other hand, if maintaining an overall high level of accuracy for both high-risk and low-risk loans (predicting '0's) is a top priority, Model 2 also performs exceptionally well.
In summary, the choice of model should align with the institution's specific objectives, whether it's focused on predicting '1's or '0's, and it should be made while being mindful of potential overfitting risks.

