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
import os

warnings.filterwarnings("ignore") 

MODEL_PATH = "apc_model.pkl"
MOISTURE_COL = "MOISTURE(%)"
APC_COL = "TPC" 
DATE_COL = "DATE OF ISSUE"
SOURCE_COL = "ITEM" 
TARGET_COLUMN = "log10_APC"


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
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    
    df["Month"] = df[DATE_COL].dt.month

    df["Season"] = df["Month"].map({
        12:"DJF", 1:"DJF", 2:"DJF", 
        3:"MAM", 4:"MAM", 5:"MAM", 
        6:"JJA", 7:"JJA", 8:"JJA", 
        9:"SON", 10:"SON", 11:"SON"
    })
    
    if APC_COL in df.columns:
        df["APC_num"] = df[APC_COL].apply(parse_apc)
        df[TARGET_COLUMN] = np.log10(df["APC_num"])
        

    if MOISTURE_COL in df.columns:
        df.rename(columns={MOISTURE_COL: "Moisture"}, inplace=True)
 
    if SOURCE_COL in df.columns:
        df.rename(columns={SOURCE_COL: "Source"}, inplace=True)
    
    return df


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

        required_cols = ["Moisture", "Source", "Month", "Season", TARGET_COLUMN]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Training Failed: Missing required column '{col}' after processing. Check data and column definitions.")
                return None
   
        df = df.dropna(subset=[TARGET_COLUMN, "Moisture", "Source", DATE_COL]) 
        df = df.sort_values(DATE_COL).reset_index(drop=True) 

        X = df[["Moisture", "Source", "Month", "Season"]]
        y = df[TARGET_COLUMN]
        
        if len(X) < 4: 
             st.error("Too few samples left after cleaning to run TimeSeriesSplit (need at least 4).")
             return None

        pre = ColumnTransformer([
            ("num", StandardScaler(), ["Moisture"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Source", "Month", "Season"])
        ])

        model = Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(random_state=42))])

        tscv = TimeSeriesSplit(n_splits=4)
        maes = []
        
        st.info("Starting Walk-Forward Cross-Validation...")
        for i, (tr, te) in enumerate(tscv.split(X)):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            mae = np.mean(np.abs(y.iloc[te] - pred))
            maes.append(mae)
            
        st.success(f"Model trained and evaluated. Avg MAE (log10): {np.mean(maes):.3f}")

        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        st.success(f"Final model saved to '{MODEL_PATH}'")
        
        return model
        
    except Exception as e:
        st.error(f"An unexpected error occurred during training: {e}")
        return None

def run_prediction_app():
    st.title("APC Prediction AI Service 🧪")
    st.markdown("Upload a file with new batch data to get automated APC predictions.")

    try:
        apc_model = joblib.load(MODEL_PATH)
        st.sidebar.success(f"Model loaded successfully from '{MODEL_PATH}'")
    except FileNotFoundError:
        st.sidebar.error("**Model not found. Run training (Step 1) or ensure the training file is present.**")
        st.stop()
    uploaded_file = st.file_uploader(
        "Upload your new data file for prediction (.xls, .xlsx or .csv)", 
        type=['xls', 'xlsx', 'csv']
    )

    if uploaded_file is not None:
        st.subheader("Data Preview")
        
        data_bytes = io.BytesIO(uploaded_file.getvalue())
        if uploaded_file.name.endswith(('.csv', '.CSV')):
            df_new = pd.read_csv(data_bytes)
        else:
            df_new = pd.read_excel(data_bytes) 

        df_new.reset_index(inplace=True)

        st.dataframe(df_new.head())

        df_pred = df_new.copy()
        
        try:
            df_pred = create_features(df_pred)

            prediction_features = ["Moisture", "Source", "Month", "Season"]

            df_pred_clean = df_pred.dropna(subset=prediction_features)
            X_pred_clean = df_pred_clean[prediction_features]
            
            if X_pred_clean.empty:
                st.error("All rows in the uploaded file were dropped because they were missing required data (Moisture, Date, or Source).")
                return

            log10_apc_preds = apc_model.predict(X_pred_clean)

            df_pred_clean["Predicted_log10_APC"] = log10_apc_preds
            
            df_pred_clean["Predicted_APC_CFU_g"] = np.power(10, log10_apc_preds).round(0).astype(np.float64) 

   
            
            final_output = df_new.merge(
                df_pred_clean[['index', 'Predicted_APC_CFU_g']],
                on='index',
                how='left'
            )
            
   
            final_output.drop(columns=['index'], inplace=True)
            
            st.subheader("Prediction Results (All Original Columns + Prediction)")
     
            temp_fill_value = -1 
            final_output['Predicted_APC_CFU_g'] = final_output['Predicted_APC_CFU_g'].fillna(temp_fill_value)

            final_output['Predicted_APC_CFU_g'] = final_output['Predicted_APC_CFU_g'].apply(
                lambda x: str(int(x)) if x != temp_fill_value else 'N/A (Missing data)'
            )
  
            st.dataframe(final_output)

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


if __name__ == '__main__':
    st.sidebar.header("Execution Steps")
    st.sidebar.markdown(f"**1. Training:** Create the `{MODEL_PATH}` file using the file defined below.")
    st.sidebar.markdown(f"**2. App:** Run the prediction service.")
    
    TRAINING_FILE_PATH = "tpc.xls"
    
    if not os.path.exists(MODEL_PATH):
        st.sidebar.warning(f"Model file '{MODEL_PATH}' not found! Running training automatically...")
        train_and_save_model(TRAINING_FILE_PATH)

    run_prediction_app()


