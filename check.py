import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
import io
import warnings
warnings.filterwarnings("ignore") # Suppress minor future warnings

# --- Configuration and Constants ---
MODEL_PATH = "apc_model.pkl"
# Define column names based on user's data
MOISTURE_COL = "MOISTURE(%)"
APC_COL = "TPC" 
DATE_COL = "DATE OF ISSUE"
SOURCE_COL = "ITEM" 
TARGET_COLUMN = "log10_APC"

# --- 1. Utility Functions for Data Preparation ---

def parse_apc(x, lod_default=100.0):
    """Handles '<LOD>' values by returning LOD/2 and converts to float."""
    s = str(x).replace(",", "").strip()
    if s.startswith("<"):
        lod = float(s[1:]) if s[1:].replace(".","",1).isdigit() else lod_default
        return lod / 2
    try: return float(s)
    except: return np.nan

def create_features(df):
    """Engineers features and renames columns for the model."""
    
    # 1. Date and Time Features (Robustly handles mixed string/serial date formats)
    # dayfirst=True ensures formats like '26/1/2017' are correctly parsed as day/month/year
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    
    # Use .dt.is_month_start or similar to ensure Month is always present after date conversion
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
    
    # 4. RENAME ITEM to Source (for the model's categorical feature)
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
        st.sidebar.error("**Model not found. Please complete Step 1 (Training) first.**")
        st.stop() 

    # 3.2 File Uploader Input (Now includes .xls)
    uploaded_file = st.file_uploader(
        "Upload your new data file for prediction (.xlsx, .xls, or .csv)", 
        type=['xlsx', 'xls', 'csv']
    )

    if uploaded_file is not None:
        st.subheader("Data Preview")
        
        data_bytes = io.BytesIO(uploaded_file.getvalue())
        
        # Check file extension to choose the correct reader
        if uploaded_file.name.endswith(('.csv', '.CSV')):
            df_new = pd.read_csv(data_bytes)
        # Use read_excel for both .xlsx and .xls
        else:
            df_new = pd.read_excel(data_bytes)

        st.dataframe(df_new.head())

        # 3.3 Data Preparation
        df_pred = df_new.copy()
        
        try:
            df_pred = create_features(df_pred)
            
            # --- FIX: Drop rows with missing feature values before prediction ---
            prediction_features = ["Moisture", "Source", "Month", "Season"]
            
            # Record the number of rows before dropping NaNs
            original_count = len(df_pred)
            
            # Drop rows with any NaN in the required features
            df_pred_clean = df_pred.dropna(subset=prediction_features)
            
            # Check if any rows were dropped due to missing data
            dropped_count = original_count - len(df_pred_clean)
            if dropped_count > 0:
                st.warning(f"⚠️ {dropped_count} rows were skipped for prediction due to missing data in 'Moisture' or required Date columns.")
                
            if df_pred_clean.empty:
                st.error("Prediction Failed: No valid rows remaining after cleaning missing features.")
                return

            # Select the features required by the model
            X_pred = df_pred_clean[prediction_features]
            
            # 3.4 Predict 
            log10_apc_preds = apc_model.predict(X_pred)

            # 3.5 Back-transform to CFU/g space
            df_pred_clean["Predicted_log10_APC"] = log10_apc_preds
            # Rounding to 0 decimal places for easier reading (CFU/g are counts)
            df_pred_clean["Predicted_APC_CFU_g"] = np.power(10, log10_apc_preds).round(0)
            
            st.subheader("Prediction Results")
            # Show original data + the new prediction columns
            # Ensure we only use columns present in the cleaned prediction dataframe
            output_cols = [DATE_COL, "Moisture", "Source", APC_COL, "Predicted_APC_CFU_g"]
            final_output = df_pred_clean[[c for c in output_cols if c in df_pred_clean.columns]]

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
            st.error(f"Prediction Failed: Missing required column {e}. Check that all input columns match the training data format (DATE OF ISSUE, MOISTURE(%), TPC, ITEM).")
        except Exception as e:
             st.error(f"An unexpected error occurred during prediction: {e}")

# --- Execution Block ---

if __name__ == '__main__':
    st.sidebar.header("Execution Steps")
    st.sidebar.markdown(f"**1. Training:** Create the `{MODEL_PATH}` file.")
    st.sidebar.markdown(f"**2. App:** Run the prediction service.")
    
    # --- IMPORTANT: Training Configuration ---
    # **REPLACE THE PATH BELOW WITH YOUR EXACT FILE LOCATION AND NAME**
    TRAINING_FILE_PATH = "C://Users//KARTHIKA//Downloads//New Microsoft Office Excel 97-2003 Worksheet.xls"
    
    # ----------------------------------------
    
    # --- STEP 1: RUN THIS BLOCK ONCE TO TRAIN THE MODEL ---
    # **UNCOMMENT** the line below and run the script once to create the model.
    # train_and_save_model(TRAINING_FILE_PATH)
    
    # --- STEP 2: **COMMENT OUT** THE LINE ABOVE AND RUN THIS BLOCK FOR THE WEB APP ---
    run_prediction_app()
