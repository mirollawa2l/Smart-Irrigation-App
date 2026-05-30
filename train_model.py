import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import joblib

print("Loading dataset...")
# Load the dataset
df = pd.read_csv("train.csv")

# Drop the 'id' column as it's not a predictor
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Separate features and target
X = df.drop(columns=['Irrigation_Need'])
y = df['Irrigation_Need']

# XGBoost requires numeric targets. Map text to integers.
target_mapping = {"Low": 0, "Medium": 1, "High": 2}
y_encoded = y.map(target_mapping)

# Identify numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("Building and training pipeline with XGBoost...")
# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Create the full pipeline with XGBoost
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        random_state=42, 
        eval_metric='mlogloss'
    ))
])

# Train the model
model_pipeline.fit(X, y_encoded)

# Save the trained pipeline
joblib.dump(model_pipeline, 'xgboost_irrigation_model.joblib')
print("Model successfully trained and saved as 'xgboost_irrigation_model.joblib'!")