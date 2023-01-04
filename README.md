# ML-Project_9-Used-car-price-prediction
### Data Details 
   
   Kaggle - https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv  
   Colab Notebook - https://colab.research.google.com/drive/1RZm65akz-VqAvdZyKpFehEUbmz6cNKB7?usp=sharing
       
#### Please use the Ipynb file in the repository for a detailed explanation of this project. This is because the project has been completed and the steps have been written in the notebook referenced in the repository.
Link - https://github.com/SudeepSinha09/ML-Project_9-Used-car-price-prediction/blob/main/Used%20car%20price%20prediction.ipynb

## Description

This dataset contains information about used cars.
This data can be used for a lot of purposes such as price prediction to exemplify the use of linear regression in Machine Learning.

## An Overview of EDA

![image](https://user-images.githubusercontent.com/93086122/210528420-2c7a8937-88e8-45e5-a58d-ae78aa46297a.png)

![image](https://user-images.githubusercontent.com/93086122/210528463-de13c980-1082-417d-b92a-285214821390.png)

## Attribute Information

The columns in the given dataset are as follows:

name  
year  
selling_price  
km_driven  
fuel  
seller_type  
transmission  
Owner  

For used motorcycle datasets please go to https://www.kaggle.com/nehalbirla/motorcycle-dataset

## Project Brief

In this project, we aimed to build a machine learning model to predict the prices of used cars based on various features such as make, model, year, mileage, and condition. Predicting the prices of used cars is important for both buyers and sellers as it helps them determine a fair value for the vehicle.

To build the prediction model, we collected data on used cars from various sources and pre-processed it to prepare it for modeling. We then split the data into a training set and a test set to evaluate the models.

We evaluated several different machine learning models for the task, including random forest regression and multiple linear regression. We trained each model using the training data and evaluated their performance on the test data using various evaluation metrics.

To further improve the performance of the models, we performed hyperparameter tuning on the random forest regression model. This involved adjusting the hyperparameters of the model to optimize its performance on the test data.

The results showed that the random forest regression model outperformed the multiple linear regression model and was the most accurate in predicting the prices of used cars. Based on this, we selected the random forest regression model as the final prediction model for used car prices.

Overall, this project helped us develop a model that can accurately predict the prices of used cars based on various features, enabling buyers and sellers to make informed decisions about the value of a used car.

##### The model selected - Random forest regression
