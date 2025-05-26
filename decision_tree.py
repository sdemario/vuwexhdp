#!/usr/pkg/bin/python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd

import sys
import time
import random


# Step 1: Create Spark session
spark = SparkSession.builder.appName("KDDDecisionTree").getOrCreate()

# Step 2: Load data
#data_path = "hdfs:///user/chenxiao17/input/kdd.data"
data_path = "hdfs:///user/almeidshel/group/kdd.data"
raw_data = spark.read.csv(data_path, inferSchema=True)

# Step 3: Rename columns (41 features + 1 label)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]
df = raw_data.toDF(*columns)

# Step 4: Encode string columns
indexers = [ StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in ["protocol_type", "service", "flag"] ]

# Step 5: Feature assembler
feature_cols = [col for col in df.columns if col not in ("label", "protocol_type", "service", "flag")]
feature_cols += ["protocol_type_idx", "service_idx", "flag_idx"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label", handleInvalid="keep")

# Step 6: Classifier
dt = DecisionTreeClassifier(labelCol="indexed_label", featuresCol="features", maxBins=128)

# Step 7: Pipeline
pipeline = Pipeline(stages=indexers + [assembler, label_indexer, dt])

def seed_run(seed=42):
    # Step 8: Train-test split (with seed)
    (train, test) = df.randomSplit([0.7, 0.3], seed=seed)

    # Step 9: Train
    start = time.time()
    model = pipeline.fit(train)
    end = time.time()

    # Step 10: Predict and evaluate
    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexed_label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    runtime = end - start

    print(f"Seed {seed} - Accuracy: {accuracy:.5f}, Runtime: {runtime:.2f} sec")
    return {"seed": seed, "accuracy": accuracy, "runtime": runtime}


results = []
seeds = [random.randint(10, 100) for _ in range(10)]
for seed in seeds:
    results.append(seed_run(seed))

print()
print()
print("Result - Decision Tree")
print("----------------------")
dfRes = pd.DataFrame(results)
print(dfRes)

print()
print()
print("Summary - Decision Tree")
print("----------------------")
stats = dfRes.agg(["mean", "min", "max", "std"])
stats = stats.drop(columns="seed")
print(stats)


spark.stop()
