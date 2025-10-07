import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import io
import warnings
import os # <-- CRITICAL FIX: Import os for file existence check

warnings.filterwarnings("ignore") # Suppress minor future warnings

# --- Configuration and Constants ---
MODEL_PATH = "apc_model.pkl"
# Define column names from the user's data
MOISTURE_COL = "MOISTURE(%)"
APC_COL = "TPC" 
DATE_COL = "DATE OF ISSUE"
# NEW CONSTANT: Use 'ITEM' as the source for your product (e.g., Turmeric Powder)
SOURCE_COL = "ITEM" 
TARGET_COLUMN = "log10_APC"

# --- 1. Utility Functions for Data Preparation ---

def parse_apc(x, lod_default=100.0):
    """Handles '<LOD>' values by returning LOD/2 and converts to float."""
    s = str(x).replace(",", "").strip()
    if s.startswith("<"):
        # Safely parse LOD value, defaulting if necessary
        lod = float(s[1:]) if s[1:].replace(".","",1).isdigit() else lod_default
        return lod / 2
    try: return float(s)
    except: return np.nan

def create_features(df):
    """Engineers features and renames columns for the model."""
    
    # 1. Date and Time Features (FIXED for mixed formats)
    # dayfirst=True handles common European formats like 26/1/2017
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    
    df["Month"] = df[DATE_COL].dt.month
    
    # Mapping Month to Season
    df["Season"] = df["Month"].map({
        12:"DJF", 1:"DJF", 2:"DJF", 
        3:"MAM", 4:"MAM", 5:"MAM", 
        6:"JJA", 7:"JJA", 8:"JJA", 
        9:"SON", 10:"SON", 11:"SON"
    })
    
    # 2. Target Transformation
    if APC_COL in df.columns:
        df["APC_num"] = df[APC_COL].apply(parse_apc)
        df[TARGET_COLUMN] = np.log10(df["APC_num"])
        
    # 3. Rename Moisture column
    if MOISTURE_COL in df.columns:
        df.rename(columns={MOISTURE_COL: "Moisture"}, inplace=True)
    
    # 4. CRITICAL FIX: RENAME ITEM to Source
    if SOURCE_COL in df.columns:
        df.rename(columns={SOURCE_COL: "Source"}, inplace=True)
    
    return df

# --- 2. Model Training Function (Run ONCE to create apc_model.pkl) ---

@st.cache_resource 
def train_and_save_model(data_path):
    df = None

    try:
        st.info(f"Attempting to load data from {data_path}...")
        df = pd.read_excel(data_path) 
        st.success(f"Successfully loaded {len(df)} rows of data.")
    except FileNotFoundError:
        # Crucial to return None on file error to prevent UnboundLocalError
        st.error(f"Error: The training data file '{data_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading Excel file: {e}.")
        return None
    
    if df is None or df.empty:
        st.error("Dataframe could not be loaded or is empty. Cannot proceed with training.")
        return None
        
    try:
        df = create_features(df)
        
        # Check for required columns after feature creation
        required_cols = ["Moisture", "Source", "Month", "Season", TARGET_COLUMN]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Training Failed: Missing required column '{col}' after processing. Check data and column definitions.")
                return None
                
        # Drop rows missing required training data
        df = df.dropna(subset=[TARGET_COLUMN, "Moisture", "Source", DATE_COL]) 
        df = df.sort_values(DATE_COL).reset_index(drop=True) 

        X = df[["Moisture", "Source", "Month", "Season"]]
        y = df[TARGET_COLUMN]
        
        if len(X) < 4: 
             st.error("Too few samples left after cleaning to run TimeSeriesSplit (need at least 4).")
             return None

        # Preprocessing Pipeline Setup
        pre = ColumnTransformer([
            ("num", StandardScaler(), ["Moisture"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Source", "Month", "Season"])
        ])

        # Model Pipeline Setup
        model = Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(random_state=42))])

        # Walk-forward CV (Evaluation)
        tscv = TimeSeriesSplit(n_splits=4)
        maes = []
        
        st.info("Starting Walk-Forward Cross-Validation...")
        for i, (tr, te) in enumerate(tscv.split(X)):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            mae = np.mean(np.abs(y.iloc[te] - pred))
            maes.append(mae)
            
        st.success(f"Model trained and evaluated. Avg MAE (log10): {np.mean(maes):.3f}")

        # Final Fit and Save
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        st.success(f"Final model saved to '{MODEL_PATH}'")
        
        return model
        
    except Exception as e:
        st.error(f"An unexpected error occurred during training: {e}")
        return None

# --- 3. Prediction App (Main Streamlit Logic) ---

def run_prediction_app():
    st.title("APC Prediction AI Service 🧪")
    st.markdown("Upload a file with new batch data to get automated APC predictions.")

    # 3.1 Load the Model
    try:
        apc_model = joblib.load(MODEL_PATH)
        st.sidebar.success(f"Model loaded successfully from '{MODEL_PATH}'")
    except FileNotFoundError:
        st.sidebar.error("**Model not found. Run training (Step 1) or ensure the training file is present.**")
        st.stop() 

    # 3.2 File Uploader Input
    uploaded_file = st.file_uploader(
        "Upload your new data file for prediction (.xls, .xlsx or .csv)", 
        type=['xls', 'xlsx', 'csv']
    )

    if uploaded_file is not None:
        st.subheader("Data Preview")
        
        data_bytes = io.BytesIO(uploaded_file.getvalue())
        # Use pandas to read based on file type
        if uploaded_file.name.endswith(('.csv', '.CSV')):
            df_new = pd.read_csv(data_bytes)
        else:
            # Handles .xls and .xlsx
            df_new = pd.read_excel(data_bytes) 
        
        # FIX: Add a temporary 'index' column immediately to the original dataframe 
        # so it can be used as a stable key for merging later, even after cleaning/dropping rows.
        df_new.reset_index(inplace=True)

        st.dataframe(df_new.head())

        # 3.3 Data Preparation
        df_pred = df_new.copy()
        
        try:
            # Create necessary features (Moisture, Source, Month, Season) on the prediction dataframe
            df_pred = create_features(df_pred)
            
            # Select the features required by the model
            prediction_features = ["Moisture", "Source", "Month", "Season"]
            
            # CRITICAL FIX for prediction NaNs: Create a clean set for prediction
            # df_pred_clean inherits the 'index' column from df_new/df_pred
            df_pred_clean = df_pred.dropna(subset=prediction_features)
            X_pred_clean = df_pred_clean[prediction_features]
            
            if X_pred_clean.empty:
                st.error("All rows in the uploaded file were dropped because they were missing required data (Moisture, Date, or Source).")
                return

            # 3.4 Predict 
            log10_apc_preds = apc_model.predict(X_pred_clean)

            # 3.5 Back-transform to CFU/g space
            df_pred_clean["Predicted_log10_APC"] = log10_apc_preds
            
            # FIX: Use standard float type (np.float64) instead of 'Int64' to avoid compatibility issues.
            # We will handle the integer display in the final formatting step.
            df_pred_clean["Predicted_APC_CFU_g"] = np.power(10, log10_apc_preds).round(0).astype(np.float64) 

            # --- REVISED MERGE LOGIC ---
            # Use pandas merge on the common 'index' column to join the prediction back to the full original dataframe (df_new)
            
            final_output = df_new.merge(
                df_pred_clean[['index', 'Predicted_APC_CFU_g']],
                on='index',
                how='left' # Use left join to keep all original rows
            )
            
            # Clean up: Drop the temporary index column, which is no longer needed
            final_output.drop(columns=['index'], inplace=True)
            
            st.subheader("Prediction Results (All Original Columns + Prediction)")
            
            # FIX: Use lambda function with string formatting to ensure predicted numbers 
            # are displayed as integers (no decimal), while preserving 'N/A (Missing data)'.
            
            # Fill NaN first with a temporary unique marker (e.g., -1) to apply integer format only to valid numbers
            temp_fill_value = -1 
            final_output['Predicted_APC_CFU_g'] = final_output['Predicted_APC_CFU_g'].fillna(temp_fill_value)
            
            # Apply formatting: Convert numbers to int string, convert temporary marker back to N/A string
            final_output['Predicted_APC_CFU_g'] = final_output['Predicted_APC_CFU_g'].apply(
                lambda x: str(int(x)) if x != temp_fill_value else 'N/A (Missing data)'
            )
            
            # The final_output now contains all original columns + the prediction
            st.dataframe(final_output)

            # 3.6 Download Button
            csv_output = final_output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_output,
                file_name='APC_predictions.csv',
                mime='text/csv',
            )
            
        except KeyError as e:
            st.error(f"Prediction Failed: Missing required column {e}. Check that all input columns match the training data format.")
        except Exception as e:
             st.error(f"An unexpected error occurred during prediction: {e}")

# --- Execution Block ---

if __name__ == '__main__':
    st.sidebar.header("Execution Steps")
    st.sidebar.markdown(f"**1. Training:** Create the `{MODEL_PATH}` file using the file defined below.")
    st.sidebar.markdown(f"**2. App:** Run the prediction service.")
    
    # --- IMPORTANT: Training Configuration ---
    # **THIS FILENAME MUST MATCH THE FILE YOU UPLOADED TO THE REPOSITORY!**
    TRAINING_FILE_PATH = "tpc.xls"
    
    # ----------------------------------------
    
    # --- Step 1: Check and Train Model if Not Found (Auto-Fix for Cloud Deployment) ---
    if not os.path.exists(MODEL_PATH):
        st.sidebar.warning(f"Model file '{MODEL_PATH}' not found! Running training automatically...")
        train_and_save_model(TRAINING_FILE_PATH)
    
    # --- Step 2: Run the Prediction App ---
    run_prediction_app()
