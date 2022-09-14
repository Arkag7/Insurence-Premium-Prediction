Insurance internship Project

Problem Statement:
The goal of this project is to give people an estimate of how much they need based on their individual health situation. After that, customers can work with any health insurance carrier and its plans and perks while keeping the projected cost from our study in mind. This can assist a person in concentrating on the health side of an insurance policy rather han the ineffective part.

Approach:
The classical machine learning tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and Model Testing. Try out different machine learning 
algorithms that’s best fit for the above case. Some Famous 

Algorithms: - Multiple Linear Regression, Decision tree Regression and Gradient Boosting, Adaboost Regressor, RandomForestRegression and Elasticnet.

##Result:
We have build a solution that should able to predict the amount of health insurance.
ApplicationLink:https://insurencepr.herokuapp.com/

Follow the steps after making github repository and cloning in the working folder.

STEPS ARE MENTIONED BELOW FOR MAKING THE ENTIRE PIPELINE
Step 1:- Create env

conda create -n insurance python=3.7 -y

Step 2:- Activate env

conda activate insurance
Step 3:- Download dataset :- [insurance](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction)

Step 4:- Creat template for project

code present in template.py

Step 5:- Add all the file to github

git add -A
git commit -m "first commit"
git push -u origin main

Step 6:- After finish model building now time to create webapp:-

Step 7:- app.py on root dir for creating flask api Now make routes like \ for rendering home page and /predict for rendering predictions.

Author: Arkadev Chakrabarti

Thank You
