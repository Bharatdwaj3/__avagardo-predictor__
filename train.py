import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import os;

def main():
    print("Loading Avacado Argicultural dataset...")
    df=pd.read_csv('data/data.csv')
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nColumns in the dataset: ")
    print(df.columns.tolist())

    print("\n Basic statistics for numerical columns: ")
    print(df.describe())

    missing_values=df.isnull().sum()
    print("\nMissing Values in each column: ")
    print(missing_values[missing_values>0] if any(missing_values>0) else "No Missing values found")
    print("\n Preparing features for yield production model...")

    x=df[['Country','Region','Cultivated Area (Ha)','Organic Production (%)','Farm Size Category','Production System']]
    y=df['Yield (MT/Ha)']

    categorical_features=['Country','Region','Farm Size Category','Production System']
    categorical_transformer=OneHotEncoder(handle_unknown='ignore')

    numeric_features=['Cultivated Area (Ha)','Organic Production (%)']
    numeric_transformer=StandardScaler()

    preprocessor=ColumnTransformer(
        transformers=[
            ('num',numeric_transformer, numeric_features),
            ('cat',categorical_transformer, categorical_features)
        ]
    )

    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]}samples")

    print("\nTraining Random Forest model for yield prediction...")
    model = Pipeline([
        ('preprocessor',preprocessor),
        ("regressor",RandomForestRegressor(n_estimators=100, random_state=442))
    ])

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation: ")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

    print("\nFeature Importance Analysis: ")
    feature_names=[]
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            for i, features in enumerate(features):
                categories=transformer.categories_[i]
                for cat in categories:
                    feature_names.append(f"{features}_{cat}")
        else:
            feature_names.extend(features)
    
    importances=model.named_steps['regressor'].feature_importances_

    if len(importances) == len(feature_names):
        feature_importance=pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("Top 10 most impoprtant features: ")
        print(feature_importance.head(10))
    else:
        print(f"Feature importance mismatch: {len(importances)} importances vs {len(feature_names)} feature names")

    print("\nSaving model to disk ... ")
    model_dir='model'
     
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, 'avacado_yield_model.pkl'))

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Yield (MT/Ha)')
    plt.ylabel('Predicted Yield (MT/Ha)')
    plt.title('Actual vs Predicted Avacado Yield')
    plt.tight_layout()

    plt.savefig(os.path.join(model_dir, 'yield_predictation_performance.png'))
    print("Model training visualization saved to model/yield_prediction_performance.png")
    print("\nModel training complete and saved to model/avacado_model_yield.pkl")
    print("\nReady for use in the predection application(app.py)")
    
if __name__== "__main__":
    main()