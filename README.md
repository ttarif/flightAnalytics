# Flight Data Analysis - Insights to Reduce Carbon Footprint and Optimize Ticket Pricing

This project analyzes flight data to identify correlations between features in regards to ticket price and origin country based on the level of CO2 emission.

## Problem

The goal of this project is to identify if there exists correlations between features in regards to ticket price and origin country based on the level of CO2 emission.

## Interests

- Utilizing the feature set to minimize ticket costs.
- Finding locations with a greater carbon footprint.

## Data

The required data for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/polartech/flight-data-with-1-million-or-more-records). After downloading the data, rename the data file to `flight_data.csv` and placed in the same directory as the project files. Be sure to use 'copyFromLocal' to copy the file from your local file system to the HDFS.

## Files

This project includes the following files:

- `data_functions.py`: This Python file contains functions that clean and generate data required to plot analysis. It includes functions such as `fit_and_evaluate_model` and `prepare_data`.
- `plot_functions.py`: This Python file contains all plot functions generated in the notebook.
- `Final_Project.ipynb`: This Python file contains all analysis items including charts, plots, and summary statistics calls from PySpark.

## Usage

To run this project, first make sure that you have downloaded the required data and placed it in the same directory as the project files. Then, open the `Final_Project.ipynb` file in a Jupyter Notebook and run it to generate the analysis and visualizations.

## Requirements

This project requires PySpark and its dependencies to be installed. It also requires several Python libraries, including matplotlib, numpy, seaborn, and pandas.
