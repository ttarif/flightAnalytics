from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Residuals plot for Lin Reg. graphs
def plot_residuals(evaluation):
    residuals = evaluation.residuals.toPandas()
    sns.histplot(data=residuals, x='residuals', kde=True, bins = 50)
    plt.xlabel('Residuals')
    plt.show()

# Actual vs Predicted values graph (target = price)
def plot_actual_vs_predicted(predictions):
    predictions_pd = predictions.select('price', 'prediction').toPandas()
    sns.scatterplot(data=predictions_pd, x='price', y='prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs. Predicted Linear Regression Plot')
    plt.show()

# Learning Curve plot 
def plot_learning_curve(training_data, testing_data):
    train_sizes = np.linspace(0.1, 1.0, 10, 100)
    train_scores = []
    test_scores = []
    for train_size in train_sizes:
        train_size_data, _ = training_data.randomSplit([train_size, 1 - train_size], seed=42)
        lr = LinearRegression(featuresCol='features', labelCol='price')
        model = lr.fit(train_size_data)
        train_score = model.evaluate(train_size_data).r2
        test_score = model.evaluate(testing_data).r2
        train_scores.append(train_score)
        test_scores.append(test_score)
    plt.plot(train_sizes, train_scores, label='Training score')
    plt.plot(train_sizes, test_scores, label='Testing score')
    plt.xlabel('Training set size')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.title('Learning Curve based on train_sizes - Model Performance')
    plt.show()

# Residual Models plot w/r2 score
def plot_residual_models(lr_predictions, dt_predictions, rf_predictions, gbt_predictions, 
                        lr_r2_score, dt_r2_score, rf_r2_score, gbt_r2_score):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Linear Regression
    price = lr_predictions.select('price').toPandas().to_numpy()
    prediction = lr_predictions.select('prediction').toPandas().to_numpy()
    residuals = price - prediction
    slope, intercept, r_value, p_value, std_err = linregress(prediction.flatten(), residuals.flatten())
    axs[0, 0].scatter(prediction.flatten(), residuals.flatten())
    axs[0, 0].plot(prediction.flatten(), intercept + slope*prediction.flatten(), 'r', label=f'R2={lr_r2_score:.3f}')
    axs[0, 0].legend()
    axs[0, 0].set_title('Linear Regression')

    # Decision Tree Regression
    price = dt_predictions.select('price').toPandas().to_numpy()
    prediction = dt_predictions.select('prediction').toPandas().to_numpy()
    residuals = price - prediction
    slope, intercept, r_value, p_value, std_err = linregress(prediction.flatten(), residuals.flatten())
    axs[0, 1].scatter(prediction.flatten(), residuals.flatten())
    axs[0, 1].plot(prediction.flatten(), intercept + slope*prediction.flatten(), 'r', label=f'R2={dt_r2_score:.3f}')
    axs[0, 1].legend()
    axs[0, 1].set_title('Decision Tree Regression')
    
    # Random Forest Regression
    price = rf_predictions.select('price').toPandas().to_numpy()
    prediction = rf_predictions.select('prediction').toPandas().to_numpy()
    residuals = price - prediction
    slope, intercept, r_value, p_value, std_err = linregress(prediction.flatten(), residuals.flatten())
    axs[1, 0].scatter(prediction, residuals)
    axs[1, 0].plot(prediction, intercept + slope*prediction, 'r', label=f'R2={rf_r2_score:.3f}')
    axs[1, 0].legend()
    axs[1, 0].set_title('Random Forest Regression')

    # Gradient-Boosted Tree Regression
    price = gbt_predictions.select('price').toPandas().to_numpy()
    prediction = gbt_predictions.select('prediction').toPandas().to_numpy()
    residuals = price - prediction
    slope, intercept, r_value, p_value, std_err = linregress(prediction.flatten(), residuals.flatten())
    axs[1, 1].scatter(prediction.flatten(), residuals.flatten())
    axs[1, 1].plot(prediction.flatten(), intercept + slope*prediction.flatten(), 'r', label=f'R2={gbt_r2_score:.3f}')
    axs[1, 1].legend()
    axs[1, 1].set_title('Gradient-Boosted Tree Regression')

    for ax in axs.flat:
        ax.set(xlabel='Predicted Price', ylabel='Residual')

    plt.tight_layout()
    plt.show()

# Scatter plots for predicted values(all models)
def plot_scatter_plots(lr_predictions, dt_predictions, rf_predictions, gbt_predictions):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Linear Regression
    axs[0, 0].scatter(lr_predictions.select('price').toPandas(), lr_predictions.select('prediction').toPandas())
    axs[0, 0].set_title('Linear Regression')
    axs[0, 0].set_xlabel('Observed Price')
    axs[0, 0].set_ylabel('Predicted Price')

    # Decision Tree Regression
    axs[0, 1].scatter(dt_predictions.select('price').toPandas(), dt_predictions.select('prediction').toPandas())
    axs[0, 1].set_title('Decision Tree Regression')
    axs[0, 1].set_xlabel('Observed Price')
    axs[0, 1].set_ylabel('Predicted Price')

    # Random Forest Regression
    axs[1, 0].scatter(rf_predictions.select('price').toPandas(), rf_predictions.select('prediction').toPandas())
    axs[1, 0].set_title('Random Forest Regression')
    axs[1, 0].set_xlabel('Observed Price')
    axs[1, 0].set_ylabel('Predicted Price')

    # Gradient-Boosted Tree Regression
    axs[1, 1].scatter(gbt_predictions.select('price').toPandas(), gbt_predictions.select('prediction').toPandas())
    axs[1, 1].set_title('Gradient-Boosted Tree Regression')
    axs[1, 1].set_xlabel('Observed Price')
    axs[1, 1].set_ylabel('Predicted Price')

    plt.tight_layout()
    plt.show()

# Histogram plot for model residual histograms
def plot_residual_histograms(lr_predictions, dt_predictions, rf_predictions, gbt_predictions):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Linear Regression
    lr_residuals = lr_predictions.select('price', 'prediction').toPandas()['price'] - lr_predictions.select('price', 'prediction').toPandas()['prediction']
    axs[0, 0].hist(lr_residuals, bins=20)
    axs[0, 0].set_title('Linear Regression')

    # Decision Tree Regression
    dt_residuals = dt_predictions.select('price', 'prediction').toPandas()['price'] - dt_predictions.select('price', 'prediction').toPandas()['prediction']
    axs[0, 1].hist(dt_residuals, bins=20)
    axs[0, 1].set_title('Decision Tree Regression')

    # Random Forest Regression
    rf_residuals = rf_predictions.select('price', 'prediction').toPandas()['price'] - rf_predictions.select('price', 'prediction').toPandas()['prediction']
    axs[1, 0].hist(rf_residuals, bins=20)
    axs[1, 0].set_title('Random Forest Regression')

    # Gradient-Boosted Tree Regression
    gbt_residuals = gbt_predictions.select('price', 'prediction').toPandas()['price'] - gbt_predictions.select('price', 'prediction').toPandas()['prediction']
    axs[1, 1].hist(gbt_residuals, bins=20)
    axs[1, 1].set_title('Gradient-Boosted Tree Regression')

    for ax in axs.flat:
        ax.set(xlabel='Residual', ylabel='Frequency')
    plt.tight_layout()
    plt.show()

