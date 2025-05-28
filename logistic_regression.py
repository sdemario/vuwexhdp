from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import numpy as np
import pandas as pd

spark = SparkSession.builder.appName("KDD-LogisticRegression").getOrCreate()

# Adjust the path as needed
df = spark.read.csv("hdfs:///user/walkerthom4/kdd_input/kdd.csv", header=False, inferSchema=True)

# The last column is the target (label), all others are features
feature_cols = df.columns[:-1]
target_col = df.columns[-1]

# Convert the target to binary: 1 if not 'normal', else 0
label_indexer = StringIndexer(inputCol=target_col, outputCol="label")
df = label_indexer.fit(df).transform(df)

# Assemble feature columns into a single vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Scale features (standardization: mean=0, std=1)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scalerModel = scaler.fit(df)
df = scalerModel.transform(df)

results = []

for seed in range(1, 11):
    # Split data
    train, test = df.randomSplit([0.8, 0.2], seed=seed)
    
    # Train Logistic Regression on scaled features
    lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=10)
    start_time = time.time()
    model = lr.fit(train)
    predictions = model.transform(test)
    end_time = time.time()
    processing_time = end_time - start_time

    # Evaluate metrics
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    
    results.append({
        'seed': seed,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'processing_time_sec': processing_time
    })

# Convert to pandas DataFrame for summary (requires toPandas() on driver)
results_df = pd.DataFrame(results)
summary = results_df.describe().loc[['min', 'max', 'mean', 'std']]

# Save results to HDFS as a text file
results_str = "Results across seeds:\n"
results_str += results_df.to_string(index=False)
results_str += "\n\nSummary statistics:\n"
results_str += summary.to_string()

rdd = spark.sparkContext.parallelize([results_str])
rdd.saveAsTextFile("hdfs:///user/walkerthom4/kdd_logistic_regression_results")
