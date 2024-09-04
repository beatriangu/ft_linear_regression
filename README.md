<!-- ft_linear_regression -->
<a href="https://www.42.fr/">
    <img src="https://www.universfreebox.com/UserFiles/image/site_logo.gif" alt="42 logo" title="42" align="right" />
</a>

<p align="center">
    <img src="https://img.shields.io/badge/Skill%201-Algorithm&AI-9cf" alt="Skill 1: Algorithm & AI">
    <img src="https://img.shields.io/badge/Skill%202-DB%20%26%20Data-blue" alt="Skill 2: DB & Data">
    <img src="https://img.shields.io/badge/Objectives-Machine%20Learning-brightgreen" alt="Objectives: Machine Learning">
</p>

<p>
    An introduction to AI and Machine Learning. Write your own gradient descent algorithm to train a linear regression model that will be used to predict the price of a car.
</p>

<p>
    The first program will be used to predict the price of a car based on its mileage. When you launch the program, it will ask you for the mileage and should give you an approximate price of the car. The second program will be used to train your model. It will read the data set and perform a linear regression on this data.
</p>

<h3>Bonuses</h3>
<ul>
    <li>View the data on a graph: <a href="<ul>https://github.com/beatriangu/ft_linear_regression/blob/main/without%20training.png">see the graph</a></li>
    <li>Display the line resulting from your linear regression on this same graph and see if it works!</li><a href="https://github.com/beatriangu/ft_linear_regression/blob/main/predict.png">
    <li>Display the curve resulting from your cost history.</li>
    <li>A program that checks the accuracy of your algorithm.</li>
</ul>

<pre>
<code>
(django_venv) c4r4s6% python predict.py
Please enter Kilometrage: 60000
Estimated price for 60000 kms: 7184 €
Is it what you expected? (yes/no): yes
Great! I'll buy it!
(django_venv) c4r4s6% python predict.py
Please enter Kilometrage: python severaltrain.py
Error: Cannot cast input to float. Please enter a valid number.
Please enter Kilometrage: 240000
Estimated price for 240000 kms: 3436 €
Is it what you expected? (yes/no): no
Sorry to hear that
