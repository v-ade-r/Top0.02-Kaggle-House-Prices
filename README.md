# Top2% Kaggle House Prices

Code for the "House Prices - Advanced Regression Techniques" competition on Kaggle.

This is not the the code from A to Z, ready to copy/paste. It shows most important things, but omits some parts of manual testing and changing the hypertuning search spaces. It also doesn't contain EDA (Exploratory data analysis),
except one general idea. I did a lot of feature visualization, comparisons, engineering on the fly, however adding this would have lengthen the code and reduced its clarity, so I opted against it. Finally I could have added more
functions to reduce unnecessary code, but practicing data manipulation syntax was one of my goals, so again I decided not to.

What can you learn from this code?

1. How to prepare data for Machine Learning models (different methods of semi-manual filling NaNs, removing skewness, one-hot-encoding with pandas).
2. How to engineer new features and map existing ones to the better spaces for ML models.
3. How to tune hyperparameters of the most efficient ML models and Neural Nets for regression (with Optuna).
4. How to stack them together, creating ensembles and get even better results.
