"""Feature engineers the Hospital Length of Stay (LOS) dataset."""
import argparse
import logging
import os
import pathlib
import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Define the feature columns based on your `traindata_medium.csv` dataset
feature_columns_names = [
    "case_id",
    "Hospital_code",
    "Hospital_type_code",
    "City_Code_Hospital",
    "Hospital_region_code",
    "AvailableExtraRoomsinHospital",
    "Department",
    "Ward_Type",
    "Ward_Facility_Code",
    "BedGrade",
    "City_Code_Patient",
    "TypeofAdmission",
    "SeverityofIllness",
    "VisitorswithPatient",
    "Age",
    "Admission_Deposit"
]

# The label column is `Stay`
label_column = "Stay"

# Define the data types for the feature columns
feature_columns_dtype = {
    "case_id": np.float64,
    "Hospital_code": np.float64,
    "Hospital_type_code": str,  # Categorical
    "City_Code_Hospital": np.float64,
    "Hospital_region_code": str,  # Categorical
    "AvailableExtraRoomsinHospital": np.float64,
    "Department": str,  # Categorical
    "Ward_Type": str,  # Categorical
    "Ward_Facility_Code": str,  # Categorical
    "BedGrade": np.float64,
    "City_Code_Patient": np.float64,
    "TypeofAdmission": str,  # Categorical
    "SeverityofIllness": str,  # Categorical
    "VisitorswithPatient": np.float64,
    "Age": str,  # Categorical (since it is likely age ranges)
    "Admission_Deposit": np.float64,
}

# The label column `Stay` is a categorical variable
label_column_dtype = {"Stay": str}

def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/hospital_los_data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=0,  # Assuming the CSV file has a header row
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
    )
    # Drop 'patientid' column
    df = df.drop(columns=["patientid"])
    print('df head',df.head())
    print('df dtypes', df.dtypes)
    os.unlink(fn)

    logger.info("Defining transformers.")
    # Numeric features (excluding categorical ones)
    numeric_features = [
        "case_id", "Hospital_code", "City_Code_Hospital", "AvailableExtraRoomsinHospital",
        "BedGrade", "City_Code_Patient", "VisitorswithPatient", "Admission_Deposit"
    ]

    # Categorical features
    categorical_features = [
        "Hospital_type_code", "Hospital_region_code", "Department", "Ward_Type",
        "Ward_Facility_Code", "TypeofAdmission", "SeverityofIllness", "Age"
    ]

    # Pipeline for numerical features (imputation and scaling)
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Pipeline for categorical features (imputation and ordinal encoding)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder())  # Use OrdinalEncoder to convert categories to numerical labels
        ]
    )

    # Combine both numeric and categorical transformations
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    logger.info("Applying transforms.")
    #y = df.pop("Stay")  # Label is `Stay`
    y = df.pop("Stay").astype(str)
    

    
    # Apply preprocessing pipeline
    X_pre = preprocess.fit_transform(df)

    # Encode the labels (if necessary) using LabelEncoder for classification
    label_encoder = LabelEncoder()
    
    y_pre = label_encoder.fit_transform(y).reshape(len(y), 1)
    print("Unique values in y_pre:", np.unique(y_pre))

    # Concatenate features and labels
    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    # Save the datasets into train, validation, and test directories
    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    logger.info("Preprocessing complete.")
