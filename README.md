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

