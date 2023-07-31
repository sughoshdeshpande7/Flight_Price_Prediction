# Flight Price Prediction: Project Overview
![ML Project](https://img.shields.io/badge/Project-Machine%20Learning-orange?logo=python)<br>
* Built a model to predict flight prices depending on various user inputs and deployed it on flask
* Trained 10 different models to get the best performing model and optimized it for even better results
* R2 score of the final trained model is 80.97%

![Model Web App](https://github.com/sughoshdeshpande7/Flight_Price_Prediction/blob/0f23b5eee4840158a4d8da0f79945c9fb3148d50/Dataset%20and%20Images/model.png)

## Table of Contents
  * [Code and Resources Used](#code-and-resources-used)
  * [Directory Tree](#directory-tree)
  * [Data Preprocessing](#data-preprocessing)
  * [EDA](#eda)
  * [Feature Engineering](#feature-engineering)
  * [Data Cleaning](#data-cleaning)
  * [Feature Selection](#feature-selection)
  * [Model Building](#model-building)
  * [Hyper-parameter Tuning](#hyper-parameter-tuning)
  * [Deployment](#deployment)


## Code and Resources Used
![Python Badge](https://img.shields.io/badge/Python-3.9-blue?logo=python) 
[![GitHub Notebook](https://img.shields.io/badge/GitHub-Notebook-181717?logo=github)]([https://github.com/Inyrkz/breast_cancer/blob/main/k_fold_cv_article_guide.ipynb](https://github.com/Mandal-21/Flight-Price-Prediction/blob/master/flight_price.ipynb)) 
[![Article](https://img.shields.io/badge/Article-Read%20Here-green)](https://machinelearningprojects.net/flight-price-prediction/)
<br>
![Libraries Badge](https://img.shields.io/badge/Libraries-NumPy|Pandas|Matplotlib|Sklearn|seaborn|flask|datetime-brown?logo=python) <br>

To install the required packages and libraries for this project, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```  
**Dataset:**  Download the entire dataset from ![Data.xls](https://github.com/sughoshdeshpande7/Flight_Price_Prediction/blob/0f23b5eee4840158a4d8da0f79945c9fb3148d50/Dataset%20and%20Images/Data_Train.xlsx)

## Directory Tree 
```
├── Dataset and Images 
│   ├── Data_Train.xlsx
│   ├── model.png
│   ├── plane.jpeg
├── Templates
│   ├── final_trained_model.pkl
│   ├── home.html
│   ├── plane.jpeg
├── Flight Price Prediction.ipynb
├── README.md
├── app.py
├── final_trained_model.pkl
```

## Data Preprocessing
Following changes were made to the data to make it usable for a model:
*	Column with Null Values was removed.
*	Data values for 'Delhi' and 'New Delhi' were combined.
*	Date and Duration which was present as string values was converted into timestamp format.

## EDA
Following analysis were made related to dataset:
* What time of the day most flights take off
* If the duration of flights affect its price
* If total number of stops affect the flight price
* Ticket Fare Distribution by Airline
* Median ticket fare by Airline

## Feature Engineering
* One-Hot Encoding and Label Encoding was done to convert categorical dataset into vector
* Target Guided Encoding was done to avoid curse of dimensionality during feature encoding
* A one hot encoder library was used whereas a label encoder was made manually

## Data Cleaning
* The unwanted columns for the model were removed
* Outlier range and the outliers were detected using IQR method
* The outliers were replaced with the median of the remaining data values

## Feature Selection
* Mutual Information Regression was used to identify dependency between the variables to select the best features for the model 
* As all the features showed a good dependency with the target variable no specific feature was selected

## Model Building 

The data was split into 75% training and 25 % test set. An automated ML model was made so that mutiple models can be evaluated in a single code

10 different models were tried and evaluated based on their metrics:
*	**Random Forest Regression** : R2 score = 79.69%
*	**Decision Tree Regressor**  : R2 score = 64.87%
*	**Linear Regression**        : R2 score = 60.77%
*	**Ridge Method**             : R2 score = 60.77%
*	**Lasso Method**             : R2 score = 60.77%
*	**ElasticNet Method**        : R2 score = 57.27%
*	**Support Regression**       : R2 score = 2.62%
*	**K-NN**                     : R2 score = 64.77%
* **MLP Regressor**            : R2 score = 56.71%
* **Huber Regressor**          : R2 score = 59.4%

## Hyper-parameter Tuning
Clearly Random Forest outperforms the other methods but its performance can be still improved. RandomizedSearchCV was used to find the hyper-parameters and optimize the model upto an R2 score of 80.97% 

## Deployment
A Final Trained model was built on Random Forest regression and deployed on flask as a web app. The Final model can be downloaded from ```final_trained_model.pkl```
