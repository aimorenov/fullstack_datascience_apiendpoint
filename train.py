import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import  Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

DATASET_PATH = "data/get_around_pricing_project_clean.csv"
MODELS_FOLDER = "models"

# Load CSV file
df = pd.read_csv(DATASET_PATH)

# Get X and y
y = df["rental_price_per_day"]
X = df.drop(["rental_price_per_day"], axis=1)

# Split dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Automatically detect positions of numeric/categorical features in explanatory variables dataframe
idx = 0
numeric_features = []
numeric_indices = []
categorical_features = []
categorical_indices = []
for i,t in X.dtypes.iteritems():
    if ('float' in str(t)) or ('int' in str(t)) :
        numeric_features.append(i)
        numeric_indices.append(idx)
    else :
        categorical_features.append(i)
        categorical_indices.append(idx)

    idx = idx + 1

# Pipeline for pre-processing categorical features and standardizing numerical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_indices),    
        ('num', numeric_transformer, numeric_indices)
        ]
    )

# Build a pipeline with featureencoder and linear regressor
regressor_ridge = Pipeline([
        ('preprocessing', featureencoder),
        ('lin_reg', Ridge(alpha=1.5, tol=1e-05))
    ])
# Fit our regressor pipeline
regressor_ridge.fit(X_train, y_train)
# Compute accuracy on test set
accuracy = regressor_ridge.score(X_test,y_test)
print("Accuracy: ", accuracy)
# Save our model with joblib
joblib.dump(regressor_ridge, os.path.join(MODELS_FOLDER, "model.joblib"))