# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os  # For file path management

# Function to perform Species Distribution Modeling (SDM)
def run_sdm():
    global best_rf, best_gb, scaler  # Declare globals for accessibility
    print("\n--- Running Species Distribution Modeling (SDM) ---")
    
    # Load data files
    base_directory = r"C:\Users\ASUS\Downloads\C0S30049 Computing Technology Innovation Project\train_model\data"
    image_data = pd.read_excel(os.path.join(base_directory, "updated_image_data_export_15k.xlsx"))
    predictions = pd.read_csv(os.path.join(base_directory, "predictions.csv"))
    sites = pd.read_excel(os.path.join(base_directory, "sites.xlsx"))
    species = pd.read_csv(os.path.join(base_directory, "species.csv"))

    # Merging data
    merged_df = predictions.merge(species, on="species_id", how="left")
    merged_df = merged_df.merge(sites, left_on="image_id", right_on="site_id", how="left")

    # Selecting relevant columns and renaming for clarity
    modeling_df = merged_df[[
        'species_name', 'confidence_score', 'Category ', 'Latitude', 'Longtitude',
        'elevation_m', 'forestCover_%', 'canopyHeight_m', 'waterOccurrence_%', 
        'distanceToRoad_m', 'humanPopulationCount_n'
    ]].dropna()
    modeling_df = modeling_df.rename(columns={'Category ': 'Category'})

    # Data preparation
    # Encoding categorical 'Category' feature
    modeling_df['Category'] = LabelEncoder().fit_transform(modeling_df['Category'])

    # Convert columns to numeric where necessary and handle non-numeric values in 'waterOccurrence_%'
    modeling_df['waterOccurrence_%'] = pd.to_numeric(modeling_df['waterOccurrence_%'], errors='coerce')
    modeling_df = modeling_df.dropna()  # Drop rows with any NaN values

    # Adding interaction features
    modeling_df['elevation_forest_cover'] = modeling_df['elevation_m'] * modeling_df['forestCover_%']
    modeling_df['canopy_height_elevation'] = modeling_df['canopyHeight_m'] * modeling_df['elevation_m']

    # Standardizing features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(modeling_df[['Latitude', 'Longtitude', 'elevation_m', 'forestCover_%', 
                                                        'canopyHeight_m', 'waterOccurrence_%', 'distanceToRoad_m', 
                                                        'humanPopulationCount_n', 'elevation_forest_cover', 
                                                        'canopy_height_elevation']])
    scaled_df = pd.DataFrame(scaled_features, columns=['Latitude', 'Longtitude', 'elevation_m', 'forestCover_%', 
                                                       'canopyHeight_m', 'waterOccurrence_%', 'distanceToRoad_m', 
                                                       'humanPopulationCount_n', 'elevation_forest_cover', 
                                                       'canopy_height_elevation'])

    # Define features (X) and target (y)
    X = scaled_df
    y = (modeling_df['confidence_score'] > 0.5).astype(int)  # Binary target based on confidence score

    # Balancing classes
    X['target'] = y  # Temporarily add target to features for resampling
    majority = X[X['target'] == 1]
    minority = X[X['target'] == 0]

    # Upsample minority class
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    X_balanced = pd.concat([majority, minority_upsampled]).drop(columns=['target'])
    y_balanced = pd.concat([majority['target'], minority_upsampled['target']])

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Initialize models for ensemble
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    # Hyperparameter tuning using Stratified K-Fold
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # Use GridSearchCV with StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=skf, scoring='roc_auc', n_jobs=-1)

    grid_search_rf.fit(X_train, y_train)
    grid_search_gb.fit(X_train, y_train)

    # Best model selection
    best_rf = grid_search_rf.best_estimator_
    best_gb = grid_search_gb.best_estimator_

    # Predictions and probability estimates for each model
    y_pred_rf = best_rf.predict(X_test)
    y_pred_gb = best_gb.predict(X_test)

    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    y_proba_gb = best_gb.predict_proba(X_test)[:, 1]

    # Ensemble probabilities (average)
    y_proba_ensemble = (y_proba_rf + y_proba_gb) / 2
    y_pred_ensemble = (y_proba_ensemble > 0.5).astype(int)

    # Evaluation metrics for ensemble
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    roc_auc = roc_auc_score(y_test, y_proba_ensemble)
    report = classification_report(y_test, y_pred_ensemble)

    # Output results
    print("Best Parameters RF:", grid_search_rf.best_params_)
    print("Best Parameters GB:", grid_search_gb.best_params_)
    print("Ensemble Accuracy:", accuracy)
    print("Ensemble ROC AUC Score:", roc_auc)
    print("Classification Report:\n", report)

    # Confusion matrix for additional insights
    conf_matrix = confusion_matrix(y_test, y_pred_ensemble)
    print("Confusion Matrix:\n", conf_matrix)

    # Plot and save ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_ensemble)
    plt.plot(fpr, tpr, label=f"Ensemble (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    
    # Save the plot in the same directory as input files
    roc_curve_path = os.path.join(base_directory, "roc_curve.png")
    plt.savefig(roc_curve_path)  # Save the plot as an image
    print(f"\nROC Curve saved at: {roc_curve_path}")
    plt.close()

    print("\n--- SDM Completed ---")


# Function for user input and prediction
def user_input_prediction():
    print("\n--- Enter Environmental Conditions for Prediction ---")

    # Collect user inputs for each feature
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))
    elevation = float(input("Elevation (m): "))
    forest_cover = float(input("Forest Cover (%): "))
    canopy_height = float(input("Canopy Height (m): "))
    water_occurrence = float(input("Water Occurrence (%): "))
    distance_to_road = float(input("Distance to Road (m): "))
    human_population = float(input("Human Population Count: "))

    # Calculate interaction features
    elevation_forest_cover = elevation * forest_cover
    canopy_height_elevation = canopy_height * elevation

    # Combine inputs into a DataFrame
    input_data = pd.DataFrame({
        'Latitude': [latitude],
        'Longtitude': [longitude],
        'elevation_m': [elevation],
        'forestCover_%': [forest_cover],
        'canopyHeight_m': [canopy_height],
        'waterOccurrence_%': [water_occurrence],
        'distanceToRoad_m': [distance_to_road],
        'humanPopulationCount_n': [human_population],
        'elevation_forest_cover': [elevation_forest_cover],
        'canopy_height_elevation': [canopy_height_elevation]
    })

    # Scale the input data using the pre-trained scaler
    scaled_input = scaler.transform(input_data)

    # Convert scaled input back to a DataFrame with the original feature names
    scaled_input_df = pd.DataFrame(scaled_input, columns=input_data.columns)

    # Predict probabilities using the ensemble model
    probability_rf = best_rf.predict_proba(scaled_input_df)[:, 1][0]
    probability_gb = best_gb.predict_proba(scaled_input_df)[:, 1][0]

    # Ensemble probability (average of RF and GB predictions)
    probability_ensemble = (probability_rf + probability_gb) / 2

    # Output results
    print(f"\nPredicted Probability of Species Presence:")
    print(f"Random Forest: {probability_rf:.2f}")
    print(f"Gradient Boosting: {probability_gb:.2f}")
    print(f"Ensemble Model: {probability_ensemble:.2f}")
    print(f"The location is predicted to have species present with a probability of {probability_ensemble:.2%}.")


# Main execution
if __name__ == "__main__":
    run_sdm()  # Runs the SDM modeling
    user_input_prediction()  # Prompt user for inputs
