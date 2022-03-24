## This repository contains the code for the regression project completed as part of the Codeup Data Science curriculum. 

## Repo contents:
### 1. This Readme:
    - Project description with goals
    - Inital hypothesis/questions on data, ideas
    - Data dictionary
    - Project planning
    - Instructions for reproducing this project and findings
    - Key findings and recommendations for this project
### 2. Final report (final_report.ipynb)
### 3. Wrangle module (wrangle.py)
### 4. Exploration & modeling notebooks (model_testing.ipynb, explore.ipynb)
### 5. Functions to support modeling work (model.py)

### Project Goals

The goal of this project was to identify key drivers of property value for single family properties. I used statistical testing and machine learning models to provide insight into what affects home prices.

### Initial Questions and Hypotheses

1. Does the number of bathrooms affect the home's tax value?
2. Does the number of bedrooms affect the home's tax value?
3. Does the square footage of the home affect tax value?
4. Does the ratio of bedrooms to bathroooms have any effect on tax value?


### Data Dictionary

| Variable    | Meaning     |
| ----------- | ----------- |
| bedroom    |  number of bedrooms in home         |
| bathroom           |  number of bathrooms in home          |
| square_feet    |  total living area of home    |
| tax_value           | total tax assessed value of the parcel (target) |
| fips    |  Federal Information Processing Standard code (location)       |
| bed_to_bath    |  ratio of bedrooms to bathrooms      |
| device_protection   |  whether the customer has device protection option         |
| tech_support    |  whether the customer has tech support option         |


### Project Plan

For this project I followed the data science pipeline:

Planning: I established the goals for this project and the relevant questions I wanted to answer. I developed a Trello board (https://trello.com/b/yCRSVyiw/telco-churn-classification-project) to help keep track of open and completed work.

Acquire: The data for this project is from a SQL Database called 'zillow'. The acquire.py script is used to query the database for the required data tables and returns the data in a Pandas DataFrame. This script also saves the DataFrame to a .csv file for faster subsequent loads. The script will check if the telco_churn.csv file exists in the current directory and if so will load it into memory, skipping the SQL query.

Prepare: The prepare.py script has a prep_telco function that takes as an argument the Pandas DataFrame acquired from acquire.py and prepares the data for exploration and modeling. Steps here include removing null values (NaN), converting data types, and encoding categorical variables. The function also drops unnecessary columns. This script also contains a train_validate_test_split function to split the dataset into train, validate, and test sets cleanly.

Explore: The questions established in planning were analyzed using statistical tests including chi-squared and t-test to test hypotheses about the data. This work was completed in the explore.ipynb file and relevant portions were moved to the churn_report.ipynb final deliverable. A visualization illustrating the results of the tests and answering the question is included. 

Model: Four different classification algorithms were investigated to determine if churn could be predicted using features identified during exploration. A select set of hyperparameters were tested against train and validate data to determine which demonstrated the best performance. The final model was selected and used to make predictions on the withheld test data. These predictions are included in the predictions.csv in this repository.

Delivery: This is in the form of this github repository as well as a presentation of my final notebook to the stakeholders.

### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow database. Store that env file locally in the repository. 
2. Clone my repository (including the wrangle.py and explore.py modules). Confirm .gitignore is hiding your env.py file.
3. Libraries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run final_report.ipynb.

### Key Findings and Recommendations

- 

### Future work

- Explore other factors that may affect home values
- Create additional features for modeling
- Perform additional feature engineering work 
- Test additional hyperparameters for the models
