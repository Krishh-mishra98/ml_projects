# ml_projects
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:18:02 2018
@author: km
"""
from sklearn.svm import SVR        #support vector regressor
import csv
import numpy as np
import matplotlib.pyplot as plt
dates = [] # initialising empty list

prices = []

from sklearn.cross_validation import train_test_split
dates_train,dates_test,prices_train,prices_test=train_test_split(
        dates,prices,test_size=0.3,random_state=0) #sklearn.cross_validation import train_test_split 
                                                  # - >Split arrays or matrices into random train and test subsets


def get_data(filename):

         with open(filename, 'r') as csvfile:

                  csvFileReader = csv.reader(csvfile) # loading data

                  next(csvFileReader)  # skipping column names

                  for row in csvFileReader:

                          dates.append(int(row[0].split('-')[0])) # delimiter '-' to seperate values

                          prices.append(float(row[1]))

         return

def predict_price(dates, prices, x):

         dates = np.reshape(dates,(len(dates), 1)) # converting Dates into matrix of n X 1

 


# defining the support vector regression models
         
         svr_rbf = SVR(kernel= 'rbf', C= 1000, gamma= 0.1) 

         svr_lin = SVR(kernel= 'linear', C= 1000)

         svr_poly = SVR(kernel= 'poly', C= 1000, degree= 2)
         
 # fitting the data points in the models
         svr_rbf.fit(dates, prices)

         svr_lin.fit(dates, prices)

         svr_poly.fit(dates, prices)

 
# plotting the initial datapoints
         plt.scatter(dates, prices, color= 'black', label= 'Data') # scatter() is built-in function to create scatter plots

         plt.plot(dates, svr_rbf.predict(dates), color= 'orange', label= 'RBF model') # to plot the line made by the RBF kernel

         plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # to plot the line made by linear kernel

         plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # to plot the line made by polynomial kernel

         plt.xlabel('Date')# label on the x-axis

         plt.ylabel('Price')# label on the y-axis

         plt.title('Support Vector Regression for the Stock of Next Day') # title

         plt.legend() 

         plt.show()

 

         return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]



get_data(r'./stock.csv') # calling get_data method by passing the csv file to it
print('printing Dates and Corresponding Prices ::' )
print ("Dates- ", dates) # to print the dates

print ("Prices- ", prices) # to print the prices


predicted_price = predict_price(dates, prices, 24)  #calling the function predict_price() , 24 is next date

print( "\nThe stock open price for next day is:")

print( "Using RBF kernel: $", str(round(predicted_price[0],2))) # rounding-off the output upto 2 decimal digits

print( "Using Linear kernel: $", str(round(predicted_price[1],2)))

print( "Using Polynomial kernel: $", str(round(predicted_price[2],2)))
