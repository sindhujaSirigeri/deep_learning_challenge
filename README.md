# deep_learning_challenge

**Deep Learning - Neural Network**

**Project Overview:** This project involved developing a binary classifier to predict whether applicants would succeed based on their chances of success in Alphabet Soup's (a non-profit foundation) ventures if funded by Alphabet. The historical data collected from Alphabet, stored in a CSV file, contained information on over 34,000 organizations that had received funding from Alphabet Soup over the years.

**Dataset columns:**

-   **EIN and NAME:** Identification columns
-   **APPLICATION_TYPE:** Alphabet Soup application type
-   **AFFILIATION:** Affiliated sector of industry
-   **CLASSIFICATION:** Government organization classification
-   **USE_CASE:** Use case for funding
-   **ORGANISATION:** Organisation type
-   **STATUS:** Active status
-   **INCOME_AMT:** Income classification
-   **SPECIAL_CONSIDERATIONS:** Special considerations for application
-   **ASK_AMT:** Funding amount requested
-   **IS_SUCCESSFUL:** Whether the money was used effectively

**Project Results:** This project included two deep-learning models:

**Data Preprocessing:** The target variable was the 'IS_SUCCESSFUL' column, which indicated the outcome of historical applications in Alphabet Soup. The features of the model were:

-   **APPLICATION_TYPE:** Alphabet Soup application type
-   **AFFILIATION:** Affiliated sector of industry
-   **CLASSIFICATION:** Government organization classification
-   **USE_CASE:** Use case for funding
-   **ORGANISATION:** Organisation type
-   **STATUS:** Active status
-   **INCOME_AMT:** Income classification
-   **SPECIAL_CONSIDERATIONS:** Special considerations for application
-   **ASK_AMT:** Funding amount requested

The 'EIN' and 'NAME' columns were removed from the input data, as they were neither targets nor relevant features and could potentially impact the model's performance. For the model, the applicant's name or ID was not considered important.

**Compiling, Training, and Evaluating the Model:** Five models were evaluated:

1.  **The Original Model ('AlphabetSoupCharity.h5').**
    -   Included 43 features and inputs, potentially introducing high variance and low bias for the dataset.
      ![model 1](https://github.com/user-attachments/assets/6e78dc28-6374-4bf3-81c8-bab98e72d7d0)

2.  **Model 2:**
    -   Attempted to optimize the accuracy score by increasing the number of values for each bin and removing columns.
      ![model 2](https://github.com/user-attachments/assets/87d6d7a7-9073-4f77-b230-3d070cfce8c8)

3.  **Model 3:**
    -   Attempted to optimize the accuracy score by increasing the number of epochs through which the model iterated.
    ![model 3](https://github.com/user-attachments/assets/c13b78c6-2d54-4ddf-9144-55ffafc3e7db)

4.  **Model 4:**
    -   Attempted to optimize the accuracy score by increasing the number of hidden layers and neurons.
    ![model 4](https://github.com/user-attachments/assets/35a8ea78-2711-4d6b-84e0-a00901b06a07)

5.  **Model 5:**
    -   Attempted to optimize the accuracy score by modifying the activation functions.

**Questions:**
**What variable(s) were the target(s) for your model?**
  -   The target variable for the model was **'IS_SUCCESSFUL'**. This column indicated whether the funding was used effectively, representing the outcome of historical applications in Alphabet Soup.

**What variable(s) were the features for your model?**
    
    -   The features for the model were:
        -   **'APPLICATION_TYPE'**: Alphabet Soup application type.
        -   **'AFFILIATION'**: Affiliated sector of industry.
        -   **'CLASSIFICATION'**: Government organization classification.
        -   **'USE_CASE'**: Use case for funding.
        -   **'ORGANISATION'**: Organisation type.
        -   **'STATUS'**: Active status.
        -   **'INCOME_AMT'**: Income classification.
        -   **'SPECIAL_CONSIDERATIONS'**: Special considerations for application.
        -   **'ASK_AMT'**: Funding amount requested.
        
**What variable(s) were removed from the input data because they were neither targets nor features?**
    -   The variables that were removed from the input data were:
    -   **'EIN'**: An identification number that was not relevant to the prediction.
    -   **'NAME'**: The organization's name, which did not contribute to predicting success.

**How many neurons, layers, and activation functions were selected for the neural network model, and why?**

-   **Models 1, 2, 3, and 5:** 2 hidden layers, 80 neurons for the first hidden layer, 30 neurons for the second hidden layer, and 1 neuron for the output layer. Activation functions used = relu, tanh, and sigmoid.
    -   The number of hidden layers was dependent on the complexity of the model. In this case, one hidden layer wouldn't have sufficed for the model to learn complex relationships and patterns, so two layers were a good starting point.
    -   The number of neurons for each layer was generally twice the number of features/inputs the model received, making 80 a good starting point. The number of neurons was reduced to 30 in subsequent layers to allow the network to distill and focus on essential features.
    -   The output layer used a sigmoid function as the model was designed to predict a binary result (true/false).
    -   Relu and tanh were used for the first and second hidden layers, respectively. Relu was employed for faster learning and simplified output, while tanh was used to classify the data into two distinguished classes (e.g., good vs. bad or successful vs. unsuccessful).
-   **Model 4:** 5 hidden layers, 100 neurons for the first hidden layer, 40 neurons for the second hidden layer, 10 neurons for the third layer, 3 for the fourth layer, and 1 neuron for the output layer. Activation functions used = relu, tanh, and sigmoid.
    
    -   This model increased the number of neurons and hidden layers for similar reasons, assuming the problem to be predicted was more complex.

**Was the target model performance achieved?**

-   None of the models (1-5) achieved a target score higher than 75%. The third model achieved the highest score, 0.7481, and was saved as 'AlphabetSoupCharity_Optimisation.h5'.

**What steps were taken to increase the model performance?**

-   The following steps were taken to optimize the accuracy score:
    -   Increasing the number of values for each bin and removing columns.
    -   Increasing the number of epochs.
    -   Increasing the number of hidden layers and neurons.
    -   Modifying the activation functions.

**Recommendation:** Given that the model aimed to predict a binary result, a logistic regression model might have been more effective in solving Alphabet Soup's problem statement, as it estimates the probability of an event occurring (such as success or failure) based on a given dataset of independent variables. Logistic regression works well with features that have linear relationships and can also perform adequately with features that do not.

**Skills Learned:**

-   Deep learning model development and optimization
-   Data preprocessing and feature engineering
-   Neural network design and hyperparameter tuning
-   Use of TensorFlow and Keras for deep learning
-   Application of logistic regression for binary classification problems
