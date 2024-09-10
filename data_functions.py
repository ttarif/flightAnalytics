import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from scipy.stats import linregress

# Function that prepares our data for model evaluation, returns train/test datasets
def prepare_data(df):
    # Select relevant columns for model
    selected_columns = ['airline_name', 'duration', 'stops', 'price']
    regression_data = df.select(selected_columns)

    # Cast price and duration columns from string variables into floats
    regression_data = regression_data.withColumn('price', regression_data['price'].cast('float'))
    regression_data = regression_data.withColumn('duration', regression_data['duration'].cast('float'))

    # Create indices for categorical columns
    indexers = [StringIndexer(inputCol=column, outputCol=column+'_index').fit(regression_data) 
                for column in ['airline_name']]
    indexers_pipeline = Pipeline(stages=indexers)
    regression_data_indexed = indexers_pipeline.fit(regression_data).transform(regression_data)
    # Perform one-hot encoding
    encoder = OneHotEncoder(inputCols=['airline_name_index'], outputCols=['airline_name_encoded'])
    regression_data_encoded = encoder.fit(regression_data_indexed).transform(regression_data_indexed)
    regression_data_encoded = regression_data_encoded.withColumn('stops_int', regression_data_encoded['stops'].cast('int'))
    
    # Populate Vector Column
    assembler = VectorAssembler(inputCols=['airline_name_encoded', 'duration', 'stops_int'], outputCol='features')
    regression_data_assembled = assembler.transform(regression_data_encoded)

    # train/test split
    (training_data, testing_data) = regression_data_assembled.randomSplit([0.7, 0.3], seed=42)
    
    return training_data, testing_data

# Function that takes model type and train/test data and outputs the model, prediction model, r2_score
def fit_and_evaluate_model(model_class, training_data, testing_data):
    model = model_class(featuresCol='features', labelCol='price').fit(training_data)
    predictions = model.transform(testing_data)  
    if model_class == LinearRegression:
        r2_score = model.evaluate(testing_data).r2
    else:
        r2_score = predictions.stat.corr('price', 'prediction') ** 2
    
    return model, predictions, r2_score

# Display R2 scores of our models
def display_performance(lr_r2_score, dt_r2_score, rf_r2_score, gbt_r2_score):
    performance_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Gradient-Boosted Tree Regression'],
        'R2 Score': [lr_r2_score, dt_r2_score, rf_r2_score, gbt_r2_score]
    })
    display(performance_df)

def display_predictions(predictions):
    predictions.select('features', 'price', 'prediction').show(truncate = False)

# Prepare data for multiclass classification + returns heatmap
def multiClass_generator(df):
    # Correcting data entries for proper processing
    fil_df = df.withColumn("co2_percentage", regexp_replace(col("co2_percentage"), "%", "")) \
               .withColumn("co2_percentage", col("co2_percentage").cast("float")) \
               .drop("from_index") \
               .withColumn("avg_co2_emission_for_this_route", col("avg_co2_emission_for_this_route").cast("float"))

    # 3 stage pipeline process
    pipeline = Pipeline(stages=[
        StringIndexer(inputCol="dest_country", outputCol="dest_index"),
        VectorAssembler(inputCols=["co2_percentage", "dest_index", "avg_co2_emission_for_this_route"], outputCol="features"),
        LogisticRegression(labelCol="from_index", featuresCol="features")
    ])

    # Country index - 65/35 data split
    fil_df = StringIndexer(inputCol="from_country", outputCol="from_index").fit(fil_df).transform(fil_df)
    train, test = fil_df.randomSplit([0.65, 0.35], seed=10)

    # Evaluate performance of the model
    predictions = pipeline.fit(train).transform(test)
    accuracy = MulticlassClassificationEvaluator(labelCol="from_index", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
    print(f"Test accuracy: {accuracy:.2f}")

    # Generate confusion matrix plot
    confusion_matrix = pd.crosstab(predictions.select("from_index", "prediction").toPandas()["from_index"],
                                   predictions.select("from_index", "prediction").toPandas()["prediction"],
                                   rownames=["Actual"], colnames=["Predicted"])
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()

