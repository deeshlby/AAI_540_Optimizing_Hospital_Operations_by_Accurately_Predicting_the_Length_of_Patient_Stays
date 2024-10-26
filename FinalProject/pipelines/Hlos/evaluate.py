"""Evaluation script for measuring accuracy."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import accuracy_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.info("Starting evaluation.")
    
    # Load the model from the tar.gz file
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    logger.info("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    # Read the test data
    logger.info("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    print('eval data', df.head())

    # Extract the labels (y_test) and features (X_test)
    logger.info("Extracting features and labels from test data.")
    y_test = df.iloc[:, 0].to_numpy()  # First column is the label
    df.drop(df.columns[0], axis=1, inplace=True)  # Drop the label column from the features
    X_test = xgboost.DMatrix(df.values)  # Convert features to DMatrix for XGBoost

    # Make predictions
    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    # Convert predictions to integer labels if necessary
    predictions = np.round(predictions).astype(int)  # Ensure predictions are in integer format
    
    # Calculate classification accuracy
    logger.info("Calculating accuracy.")
    accuracy = accuracy_score(y_test, predictions)

    # Create the evaluation report with classification metrics
    report_dict = {
        "classification_metrics": {
            "accuracy": {
                "value": accuracy,
            },
        },
    }

    # Save the evaluation report to the output directory
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with accuracy: %f", accuracy)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    logger.info("Model evaluation completed.")
