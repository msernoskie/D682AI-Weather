import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

DATA_PATH = "DQN1 Dataset.xlsx"   
TARGET_COLUMN = "healthRiskScore"


def mean_absolute_percentage_error_safe(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


def main():
    # Load dataset
    df = pd.read_excel(DATA_PATH)
    
    # Convert boolean columns to integers for scikit-learn compatibility
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


    # Define model
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=8,
        min_samples_leaf=10
    )

    # Combine preprocessing and model
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("regressor", model)
    ])


    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)


    # Make predictions
    y_pred = pipeline.predict(X_test)


    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error_safe(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

 
    # Display sample results
    results_df = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": y_pred[:10]
    })

    print("\nSample predictions:")
    print(results_df)


if __name__ == "__main__":
    main()