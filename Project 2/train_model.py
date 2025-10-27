# DSA5208 Project 2
# Machine Learning on Weather Data
# Admon Lee Wen Xuan (A0294691N)

# Pyspark script to train ML models on weather data stored in Parquet format.

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import pandas as pd, matplotlib.pyplot as plt
from google.cloud import storage

spark = SparkSession.builder.appName('Train Model').getOrCreate()
bucket_name = "dsa5208-project-2"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Load training and test data from parquet
train_df = spark.read.parquet('gs://dsa5208-project-2/parquet/train_data.parquet')
test_df = spark.read.parquet('gs://dsa5208-project-2/parquet/test_data.parquet')

# Drop HOUR and MONTH
train_df = train_df.drop('HOUR', 'MONTH')
test_df = test_df.drop('HOUR', 'MONTH')

# Combine features into a single vector
va = VectorAssembler(outputCol = 'features', handleInvalid='skip').\
    setInputCols([col for col in train_df.columns if col != 'TMP_0'])

# Compute range of TMP_0 for NRMSE calculation

train_tmp_stats = train_df.agg(
    F.max("TMP_0").alias("max_TMP_0"),
    F.min("TMP_0").alias("min_TMP_0")
).collect()[0]

train_tmp_range = train_tmp_stats["max_TMP_0"] - train_tmp_stats["min_TMP_0"]

test_tmp_stats = test_df.agg(
    F.max("TMP_0").alias("max_TMP_0"),
    F.min("TMP_0").alias("min_TMP_0")
).collect()[0]
test_tmp_range = test_tmp_stats["max_TMP_0"] - test_tmp_stats["min_TMP_0"]

# RIDGE REGRESSION MODEL
# Define Ridge Regression model
lr = LinearRegression().\
    setFeaturesCol('features').\
    setLabelCol('TMP_0')

# Create parameter grid for hyperparameter tuning
params = ParamGridBuilder().\
    addGrid(lr.regParam, [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]).\
    build()

# Create pipeline
lr_pipeline = Pipeline().setStages([va, lr])

# Define evaluator
evaluator = RegressionEvaluator().\
    setLabelCol('TMP_0').\
    setPredictionCol('prediction').\
    setMetricName('rmse')

mae_evaluator = RegressionEvaluator().\
    setLabelCol('TMP_0').\
    setPredictionCol('prediction').\
    setMetricName('mae')

r2_evaluator = RegressionEvaluator().\
    setLabelCol('TMP_0').\
    setPredictionCol('prediction').\
    setMetricName('r2')

# Set up cross validation
lr_cv = CrossValidator(seed = 42).\
    setNumFolds(5).\
    setEstimatorParamMaps(params).\
    setEstimator(lr_pipeline).\
    setEvaluator(evaluator)

# Train model
lr_model = lr_cv.fit(train_df)

# Evaluate model and extract metrics
lr_train_rmse = evaluator.evaluate(lr_model.transform(train_df))
lr_train_mae = mae_evaluator.evaluate(lr_model.transform(train_df))
lr_train_r2 = r2_evaluator.evaluate(lr_model.transform(train_df))
lr_train_nrmse = lr_train_rmse / train_tmp_range

lr_test_rmse = evaluator.evaluate(lr_model.transform(test_df))
lr_test_mae = mae_evaluator.evaluate(lr_model.transform(test_df))
lr_test_r2 = r2_evaluator.evaluate(lr_model.transform(test_df))
lr_test_nrmse = lr_test_rmse / test_tmp_range


# Extract parameters of best model
lr_best_model = lr_model.bestModel.stages[-1]
coefficients = lr_best_model.coefficients
feature_names = va.getInputCols()
lr_param = lr_best_model.getRegParam()

# Plot residuals for a subset of test data
test_predictions = lr_model.transform(test_df).select('TMP_0', 'prediction').sample(fraction=0.001, seed=42).toPandas()
test_predictions_sample = test_predictions.sample(n=1000, random_state=42)
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions_sample['TMP_0'], test_predictions_sample['TMP_0'] - test_predictions_sample['prediction'], alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual TMP_0')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual TMP_0')
plt.savefig('/tmp/ridge_regression_residuals.png', bbox_inches='tight')

destination_blob_name = "ridge_regression_residuals.png"
source_file_name = "/tmp/ridge_regression_residuals.png"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# Plot predicted vs actual values for a subset of test data
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions_sample['TMP_0'], test_predictions_sample['prediction'], alpha=0.5)
plt.plot([test_predictions_sample['TMP_0'].min(), test_predictions_sample['TMP_0'].max()],
         [test_predictions_sample['TMP_0'].min(), test_predictions_sample['TMP_0'].max()],
         color='red', linestyle='--')
plt.xlabel('Actual TMP_0')
plt.ylabel('Predicted TMP_0')
plt.title('Predicted vs Actual TMP_0')
plt.savefig('/tmp/ridge_regression_predicted_vs_actual.png', bbox_inches='tight')

destination_blob_name = "ridge_regression_predicted_vs_actual.png"
source_file_name = "/tmp/ridge_regression_predicted_vs_actual.png"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# Plot relative importance of features using ridge regression coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['AbsCoefficient'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance from Ridge Regression Coefficients')
plt.gca().invert_yaxis()
plt.savefig('/tmp/ridge_regression_feature_importance.png', bbox_inches='tight')

destination_blob_name = "ridge_regression_feature_importance.png"
source_file_name = "/tmp/ridge_regression_feature_importance.png"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# Export model
lr_best_model.write().overwrite().save('gs://dsa5208-project-2/ridge_regression_model')

# Write results to text file
with open('/tmp/ridge_regression_results.txt', 'w') as f:
    f.write(f'Best RegParam: {lr_param}\n')
    f.write('Feature Coefficients:\n')
    for feature, coef in zip(feature_names, coefficients):
        f.write(f'  {feature}: {coef}\n')
    f.write(f'Training RMSE: {lr_train_rmse}\n')
    f.write(f'Training MAE: {lr_train_mae}\n')
    f.write(f'Training R2: {lr_train_r2}\n')
    f.write(f'Training NRMSE: {lr_train_nrmse}\n')
    f.write(f'Test RMSE: {lr_test_rmse}\n')
    f.write(f'Test MAE: {lr_test_mae}\n')
    f.write(f'Test R2: {lr_test_r2}\n') 
    f.write(f'Test NRMSE: {lr_test_nrmse}\n')

destination_blob_name = "ridge_regression_results.txt"
source_file_name = "/tmp/ridge_regression_results.txt"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)


# GRADIENT BOOSTED TREES MODEL
# Define GBT model
gbt = GBTRegressor(seed = 42).setFeaturesCol('features').setLabelCol('TMP_0')

# Create parameter grid for hyperparameter tuning
params = ParamGridBuilder().\
    addGrid(gbt.maxDepth, [2, 5, 10]).\
    addGrid(gbt.maxIter, [20, 100]).\
    addGrid(gbt.stepSize, [0.05, 0.1, 0.2]).\
    addGrid(gbt.subsamplingRate, [0.8, 1.0]).\
    addGrid(gbt.minInstancesPerNode, [2, 5]).\
    build()

# Create pipeline
gbt_pipeline = Pipeline().setStages([va, gbt])

# Set up cross validation (cross validating on a subset, then full data for final model)
cv_df = train_df.sample(0.01, seed=42)

gbt_cv = CrossValidator(seed = 42).\
    setNumFolds(3).\
    setEstimatorParamMaps(params).\
    setEstimator(gbt_pipeline).\
    setEvaluator(evaluator)

# Train model on subset to find best hyperparameters
gbt_cv_model = gbt_cv.fit(cv_df)

# Train final model on full training data with best hyperparameters
best_hyperparams = {p.name: v for p, v in gbt_cv_model.bestModel.stages[-1].extractParamMap().items()}
gbt_pipeline_final = Pipeline().setStages([va, gbt.setParams(**best_hyperparams)])
gbt_best_model = gbt_pipeline_final.fit(train_df)

# Evaluate model
gbt_train_rmse = evaluator.evaluate(gbt_best_model.transform(train_df))
gbt_train_mae = mae_evaluator.evaluate(gbt_best_model.transform(train_df))
gbt_train_r2 = r2_evaluator.evaluate(gbt_best_model.transform(train_df))
gbt_train_nrmse = gbt_train_rmse / train_tmp_range

gbt_test_rmse = evaluator.evaluate(gbt_best_model.transform(test_df))
gbt_test_mae = mae_evaluator.evaluate(gbt_best_model.transform(test_df))
gbt_test_r2 = r2_evaluator.evaluate(gbt_best_model.transform(test_df))
gbt_test_nrmse = gbt_test_rmse / test_tmp_range

# Extract parameters of best model
gbt_param = {
    'maxDepth': gbt_best_model.stages[-1].getMaxDepth(),
    'maxIter': gbt_best_model.stages[-1].getMaxIter(),
    'stepSize': gbt_best_model.stages[-1].getStepSize(),
    'subsamplingRate': gbt_best_model.stages[-1].getSubsamplingRate(),
    'minInstancesPerNode': gbt_best_model.stages[-1].getMinInstancesPerNode(),
    'numTrees': gbt_best_model.stages[-1].getNumTrees,
    'totalNodes': gbt_best_model.stages[-1].totalNumNodes
}

# Plot residuals for a subset of test data
test_predictions = gbt_best_model.transform(test_df).select('TMP_0', 'prediction').sample(0.001, seed=42).toPandas()
test_predictions_sample = test_predictions.sample(n=1000, random_state=42)
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions_sample['TMP_0'], 
            test_predictions_sample['TMP_0'] - test_predictions_sample['prediction'], alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual TMP_0')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual TMP_0')
plt.savefig('/tmp/gbt_residuals.png', bbox_inches='tight')

destination_blob_name = "gbt_residuals.png"
source_file_name = "/tmp/gbt_residuals.png"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# Plot predicted vs actual values for a subset of test data
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions_sample['TMP_0'], test_predictions_sample['prediction'], alpha=0.5)
plt.plot([test_predictions_sample['TMP_0'].min(), test_predictions_sample['TMP_0'].max()],
         [test_predictions_sample['TMP_0'].min(), test_predictions_sample['TMP_0'].max()],
         color='red', linestyle='--')
plt.xlabel('Actual TMP_0')
plt.ylabel('Predicted TMP_0')
plt.title('Predicted vs Actual TMP_0')
plt.savefig('/tmp/gbt_predicted_vs_actual.png', bbox_inches='tight')

destination_blob_name = "gbt_predicted_vs_actual.png"
source_file_name = "/tmp/gbt_predicted_vs_actual.png"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# Plot feature importances
importances = gbt_best_model.stages[-1].featureImportances.toArray()
feature_names = va.getInputCols()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importances from GBT Model')
plt.gca().invert_yaxis()
plt.savefig('/tmp/gbt_feature_importance.png', bbox_inches='tight')

destination_blob_name = "gbt_feature_importance.png"
source_file_name = "/tmp/gbt_feature_importance.png"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)

# Export model
gbt_best_model.stages[-1].write().overwrite().save('gs://dsa5208-project-2/gbt_model')

# Write results to text file
with open('/tmp/gbt_results.txt', 'w') as f:
    f.write('Best Hyperparameters:\n')
    for param, value in gbt_param.items():
        f.write(f'  {param}: {value}\n')
    f.write(f'Training RMSE: {gbt_train_rmse}\n')
    f.write(f'Training MAE: {gbt_train_mae}\n')
    f.write(f'Training R2: {gbt_train_r2}\n')
    f.write(f'Training NRMSE: {gbt_train_nrmse}\n')
    f.write(f'Test RMSE: {gbt_test_rmse}\n')
    f.write(f'Test MAE: {gbt_test_mae}\n')
    f.write(f'Test R2: {gbt_test_r2}\n')
    f.write(f'Test NRMSE: {gbt_test_nrmse}\n')

destination_blob_name = "gbt_results.txt"
source_file_name = "/tmp/gbt_results.txt"
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file_name)