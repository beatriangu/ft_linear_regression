# ft_linear_regression

This project will be your first steps into AI and Machine Learning. You're going to start with a simple, basic machine learning algorithm. You will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

Mandatory part
Implement a linear regression algorithm on a single element, the mileage of a car. To do this you must implement 2 programs:

The first program will be used to predict the price of a car based on its mileage. When you launch the program, it will ask you for mileage and should give you an approximate price of the car.
The second program will be used to train your model. It will read the data set and make a linear regression on this data.
Bonuses
View the data on a graph.
Display the line resulting from your linear regression on this same graph and see if it works!
Display the curve resulting from your cost history.
A program that checks the accuracy of your algorithm.
Install
This project uses Homebrew and Python. Go check them out if you don't have them locally installed.

Use the package manager pip3 to install all needed packages.
pip3 install numpy
pip3 insatll matplotlib
pip3 install sklearn

Usage
python3 ft_linear_regression.py[flags]
python3 priceEstimation.py

Flags
-p, --prediction            - show the prediction curve
-ch, --cost_history         - show the cost history curve
-cd, --coef_determination   - show the coefficient of determination

Rate
125/100
