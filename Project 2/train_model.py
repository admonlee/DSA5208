from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName('Train Model').getOrCreate()

# Load training and test data from parquet
#train_df = spark.read.parquet('gs://dsa5208-project-2/parquet/train_data.parquet')
#test_df = spark.read.parquet('gs://dsa5208-project-2/parquet/test_data.parquet')

train_df = spark.read.parquet('Project 2/data/parquet/train_data.parquet')
test_df = spark.read.parquet('Project 2/data/parquet/test_data.parquet')

# Combine features into a single vector
va = VectorAssembler(outputCol = 'features').\
    setInputCols([col for col in train_df.columns if col != 'TMP_0'])


# RIDGE REGRESSION MODEL
# Define Ridge Regression model
lr = LinearRegression().setFeaturesCol('features').setLabelCol('TMP_0')

# Create parameter grid for hyperparameter tuning
params = ParamGridBuilder().\
    addGrid(lr.regParam, [0.0, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]).\
    build()

# Create pipeline
lr_pipeline = Pipeline().setStages([va, lr])

# Define evaluator
evaluator = RegressionEvaluator().\
    setLabelCol('TMP_0').\
    setPredictionCol('prediction').\
    setMetricName('rmse')

# Set up cross validation
lr_cv = CrossValidator(seed = 42).\
    setNumFolds(5).\
    setEstimatorParamMaps(params).\
    setEstimator(lr_pipeline).\
    setEvaluator(evaluator)

# Train model
lr_model = lr_cv.fit(train_df)

# Evaluate model
train_rmse = evaluator.evaluate(lr_model.transform(train_df))
print(f'Ridge Regression Train RMSE: {train_rmse:.4f}')
test_rmse = evaluator.evaluate(lr_model.transform(test_df))
print(f'Ridge Regression Test RMSE: {test_rmse:.4f}')

# GRADIENT BOOSTED TREES MODEL
# Define GBT model
gbt = GBTRegressor(seed = 42).setFeaturesCol('features').setLabelCol('TMP_0')

# Create parameter grid for hyperparameter tuning
params = ParamGridBuilder().\
    addGrid(gbt.maxDepth, [3, 5, 7]).\
    addGrid(gbt.maxIter, [20, 50, 100]).\
    addGrid(gbt.stepSize, [0.05, 0.1, 0.2]).\
    addGrid(gbt.subsamplingRate, [0.8, 1.0]).\
    build()

# For testing REMOVE FOR FINAL RUN
params = ParamGridBuilder().\
    addGrid(gbt.maxDepth, [3, 5, 7]).\
    addGrid(gbt.maxIter, [20, 30]).\
    build()

# Create pipeline
gbt_pipeline = Pipeline().setStages([va, gbt])

# Set up cross validation
gbt_cv = CrossValidator(seed = 42).\
    setNumFolds(5).\
    setEstimatorParamMaps(params).\
    setEstimator(gbt_pipeline).\
    setEvaluator(evaluator)

# Train model
gbt_model = gbt_cv.fit(train_df)

# Evaluate model
train_rmse = evaluator.evaluate(gbt_model.transform(train_df))
print(f'Gradient Boosted Trees Train RMSE: {train_rmse:.4f}')
test_rmse = evaluator.evaluate(gbt_model.transform(test_df))
print(f'Gradient Boosted Trees Test RMSE: {test_rmse:.4f}')