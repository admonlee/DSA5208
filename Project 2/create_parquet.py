# DSA5208 Project 2
# Machine Learning on Weather Data
# Admon Lee Wen Xuan (A0294691N)

# Pyspark script to read extracted CSV files, clean and standardise data,
# and write to Parquet format for ML model training.

from pyspark.sql import SparkSession, functions as F
import numpy as np, time

spark = SparkSession.builder.appName('Create Parquet').getOrCreate()

# Path to folder with extracted CSVs
folder = 'gs://dsa5208-project-2/data'

required_columns = ['DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 
                    'WND', 'CIG', 'VIS', 'DEW', 'TMP']

# Read all CSVs in folder
df_raw = spark.read.option('header', True).csv(f'{folder}/*.csv')

# Add missing required columns as null
for col in required_columns:
    if col not in df_raw.columns:
        df_raw = df_raw.withColumn(col, F.lit(None))

# Drop extra columns (those not in required_columns)
extra_columns = [col for col in df_raw.columns if col not in required_columns]
if extra_columns:
    df_raw = df_raw.drop(*extra_columns)

# Select columns in desired order
df = df_raw.select(required_columns)

# Split columns that have multiple fields separated by commas
columns_to_split = ['WND', 'CIG', 'VIS', 'DEW', 'TMP']

for col in columns_to_split:
    num_splits = len(df.select(F.split(col, ',').alias(col)).first()[0])
    for i in range(num_splits):
        df = df.withColumn(f'{col}_{i}', F.split(F.col(col), ',').getItem(i))

# Drop original columns that were split
df = df.drop(*columns_to_split, 'WND_2', 'CIG_0', 'CIG_1', 'CIG_2', 'VIS_0', 'VIS_1')

# Cast numerical columns
float_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION']
int_columns = ['WND_0', 'WND_3', 'DEW_0', 'TMP_0']

for col in float_columns:
    df = df.withColumn(col, F.col(col).cast('float'))
for col in int_columns:
    df = df.withColumn(col, F.col(col).cast('int'))

# Drop rows with erroneous data
mask = (
    (F.col('TMP_0') <= -932) | (F.col('TMP_0') >= 618) |
    (F.col('TMP_1').isin(['2', '3', '6', '7'])) |
    ((F.col('WND_0') > 360) & (F.col('WND_0') != 999)) |
    (F.col('WND_0') < 0) | F.col('WND_1').isin(['3', '7']) |
    ((F.col('WND_3') > 900) & (F.col('WND_3') != 9999)) |
    (F.col('WND_3') < 0) | (F.col('WND_4').isin(['3', '7'])) |
    ~(F.col('CIG_3').isin(['N', 'Y', '9'])) |
    ~(F.col('VIS_2').isin(['N', 'V', '9'])) |
    (F.col('VIS_3').isin(['3', '7'])) |
    (F.col('DEW_0') <= -932) | ((F.col('DEW_0') >= 368) & (F.col('DEW_0') != 9999))
)

df = df.filter(~mask)

# Drop quality flag columns
df = df.drop('TMP_1', 'WND_1', 'WND_4', 'VIS_3', 'DEW_1')

# Extract hour and month from date column
df = df.withColumn('HOUR', F.hour(F.col('DATE').cast('timestamp')))
df = df.withColumn('MONTH', F.month(F.col('DATE').cast('timestamp')))
df = df.drop('DATE')

# Perform one hot encoding and cyclic transformations
df = df.select(
    '*',

    # One hot encoding
    F.when(F.col('CIG_3')=='Y', 1).otherwise(0).alias('CIG_3_Y'),
    F.when(F.col('CIG_3')=='9', 1).otherwise(0).alias('CIG_3_9'),
    F.when(F.col('VIS_2')=='V', 1).otherwise(0).alias('VIS_2_V'),
    F.when(F.col('VIS_2')=='9', 1).otherwise(0).alias('VIS_2_9'),
    F.when(F.col('WND_0') == 999, 1).otherwise(0).alias('WND_0_UNDETERMINED'),
    
    # Cyclic hour and month
    F.sin(2*np.pi*F.col('HOUR')/24).alias('HOUR_SIN'),
    F.cos(2*np.pi*F.col('HOUR')/24).alias('HOUR_COS'),
    F.sin(2*np.pi*F.col('MONTH')/12).alias('MONTH_SIN'),
    F.cos(2*np.pi*F.col('MONTH')/12).alias('MONTH_COS'),
    
    # Cyclic WND_0 with 999 -> 0
    F.when(F.col('WND_0') == 999, 0).otherwise(F.sin(F.col('WND_0') * np.pi / 180)).alias('WND_0_SIN'),
    F.when(F.col('WND_0') == 999, 0).otherwise(F.cos(F.col('WND_0') * np.pi / 180)).alias('WND_0_COS')
).drop('CIG_3', 'VIS_2', 'WND_0')

# Split to train and test sets
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Impute missing values of wind speed and dew point temperature for training set
wnd3_means = train_df.groupBy('LONGITUDE', 'LATITUDE', 'ELEVATION', 'MONTH').agg(
    F.avg('WND_3').alias('WND_3_MEAN'))
dew0_means = train_df.groupBy('LONGITUDE', 'LATITUDE', 'ELEVATION', 'MONTH').agg(
    F.avg('DEW_0').alias('DEW_0_MEAN'))

train_df = train_df.join(wnd3_means, on=['LONGITUDE', 'LATITUDE', 'ELEVATION', 'MONTH'], how='left')
train_df = train_df.withColumn('WND_3',
    F.when(F.col('WND_3') == 9999, F.col('WND_3_MEAN')).otherwise(F.col('WND_3'))
).drop('WND_3_MEAN')

train_df = train_df.join(dew0_means, on=['LONGITUDE', 'LATITUDE', 'ELEVATION', 'MONTH'], how='left')
train_df = train_df.withColumn('DEW_0',
    F.when(F.col('DEW_0') == 9999, F.col('DEW_0_MEAN')).otherwise(F.col('DEW_0'))
).drop('DEW_0_MEAN')

# Impute remaining missing values with global training mean
global_wnd3_mean = train_df.agg(F.avg('WND_3')).first()[0]
global_dew0_mean = train_df.agg(F.avg('DEW_0')).first()[0]

train_df = train_df.withColumn('WND_3',
    F.when(F.col('WND_3') == 9999, global_wnd3_mean).otherwise(F.col('WND_3')))
train_df = train_df.withColumn('DEW_0',
    F.when(F.col('DEW_0') == 9999, global_dew0_mean).otherwise(F.col('DEW_0')))

# Apply imputation to test set using training set means
test_df = test_df.join(wnd3_means, on=['LONGITUDE', 'LATITUDE', 'ELEVATION', 'MONTH'], how='left')
test_df = test_df.join(dew0_means, on=['LONGITUDE', 'LATITUDE', 'ELEVATION', 'MONTH'], how='left')

test_df = test_df.withColumn('WND_3',
    F.when(F.col('WND_3') == 9999, F.col('WND_3_MEAN')).otherwise(F.col('WND_3'))
).drop('WND_3_MEAN')
test_df = test_df.withColumn('DEW_0',
    F.when(F.col('DEW_0') == 9999, F.col('DEW_0_MEAN')).otherwise(F.col('DEW_0'))
).drop('DEW_0_MEAN')

test_df = test_df.withColumn('WND_3',
    F.when(F.col('WND_3') == 9999, global_wnd3_mean).otherwise(F.col('WND_3')))
test_df = test_df.withColumn('DEW_0',
    F.when(F.col('DEW_0') == 9999, global_dew0_mean).otherwise(F.col('DEW_0')))

# Standardise numerical columns based on training set 
numerical_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'WND_3', 'DEW_0']

for col in numerical_columns:
    mean = train_df.agg(F.avg(col)).first()[0]
    stddev = train_df.agg(F.stddev(col)).first()[0]
    train_df = train_df.withColumn(col, (F.col(col) - mean) / stddev)
    test_df = test_df.withColumn(col, (F.col(col) - mean) / stddev)

# Write to Parquet files
train_df.write.mode('overwrite').parquet('gs://dsa5208-project-2/parquet/train_data.parquet')
test_df.write.mode('overwrite').parquet('gs://dsa5208-project-2/parquet/test_data.parquet')