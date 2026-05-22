"""
ML Pipeline 101: Complete Machine Learning Workflow
===================================================

This script contains all the code from the tutorial in a clean, 
production-ready format. Copy and paste to use in your own projects.

Author: Victor Banerjee
Date: May 22' 2025
Dataset: Iris Flower Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Data
# ============================================================================

def load_data(filepath):
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================================
# STEP 2: Exploratory Data Analysis
# ============================================================================

def explore_data(df):
    """Perform initial data exploration"""
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"\nShape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df


def visualize_data(df, target_column):
    """Create visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Distribution of target
    df[target_column].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Target Distribution')
    
    # Histogram of first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols[0]].hist(ax=axes[0, 1])
        axes[0, 1].set_title(f'{numeric_cols[0]} Distribution')
    
    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        sns.heatmap(numeric_df.corr(), ax=axes[1, 0], cmap='coolwarm')
        axes[1, 0].set_title('Feature Correlation')
    
    # Box plot
    if len(numeric_cols) > 1:
        df[numeric_cols[:2]].plot(kind='box', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Box Plots')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# STEP 3: Data Preprocessing
# ============================================================================

def prepare_data(df, target_column):
    """Separate features and target"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"\n✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    """Split into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale/normalize features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"✅ Features scaled")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# STEP 4: Train Models
# ============================================================================

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    print("✅ Logistic Regression trained")
    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    """Train Decision Tree model"""
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    print("✅ Decision Tree trained")
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=5):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest trained")
    return model


# ============================================================================
# STEP 5: Evaluate Models
# ============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    return results


def compare_models(results_list):
    """Compare all models"""
    comparison_df = pd.DataFrame(results_list)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    
    return comparison_df


# ============================================================================
# STEP 6: Feature Importance
# ============================================================================

def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n⭐ FEATURE IMPORTANCE")
        print(f"{'='*50}")
        print(importance.to_string(index=False))
        
        return importance
    else:
        print("Model does not support feature importance")
        return None


# ============================================================================
# STEP 7: Hyperparameter Tuning
# ============================================================================

def tune_hyperparameters(X_train, y_train):
    """Optimize model with GridSearchCV"""
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5]
    }
    
    print("\n🔍 HYPERPARAMETER TUNING")
    print("="*50)
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_*100:.2f}%")
    
    return grid_search.best_estimator_


# ============================================================================
# STEP 8: Save and Load Model
# ============================================================================

def save_model(model, filepath):
    """Save trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {filepath}")


def load_model(filepath):
    """Load trained model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {filepath}")
    return model


def make_predictions(model, X_new):
    """Make predictions on new data"""
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    return predictions, probabilities


# ============================================================================
# STEP 9: Complete Pipeline (All-in-One)
# ============================================================================

def run_complete_pipeline(filepath, target_column):
    """Run complete ML pipeline"""
    
    # 1. Load and explore
    df = load_data(filepath)
    explore_data(df)
    
    # 2. Prepare data
    X, y = prepare_data(df, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 3. Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    model_lr = train_logistic_regression(X_train_scaled, y_train)
    model_dt = train_decision_tree(X_train, y_train)
    model_rf = train_random_forest(X_train, y_train)
    
    # 4. Evaluate models
    print("\n" + "="*50)
    print("EVALUATING MODELS")
    print("="*50)
    
    y_pred_lr = model_lr.predict(X_test_scaled)
    y_pred_dt = model_dt.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)
    
    results_lr = evaluate_model(y_test, y_pred_lr, "Logistic Regression")
    results_dt = evaluate_model(y_test, y_pred_dt, "Decision Tree")
    results_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")
    
    compare_models([results_lr, results_dt, results_rf])
    
    # 5. Tune best model
    print("\n" + "="*50)
    print("TUNING BEST MODEL")
    print("="*50)
    
    best_model = tune_hyperparameters(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    evaluate_model(y_test, y_pred_best, "Tuned Random Forest")
    
    # 6. Feature importance
    get_feature_importance(best_model, X.columns)
    
    # 7. Save model
    save_model(best_model, 'final_model.pkl')
    
    return {
        'model': best_model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_pipeline('iris_dataset.csv', 'Species')
    
    # Make predictions on test data
    model = results['model']
    X_test = results['X_test']
    y_test = results['y_test']
    
    # Predict on first 5 test samples
    predictions, probabilities = make_predictions(model, X_test.head())
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}: Predicted {pred}, Confidence: {max(prob)*100:.1f}%")
