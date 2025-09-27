import streamlit as st
import pandas as pd
from src.clean import clean_data
from sklearn.model_selection import train_test_split
from src.preprocessing import process
from src.train import train_model, evaluate_model, save_model, predict_exoplanet
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# Assume you have functions like:
# load_data, preprocess_data, train_model, predict_exoplanet, get_model_stats

# --- Sidebar Content ---
st.sidebar.title("🚀 System Controls")

# Data Ingestion (New Data Input)
st.sidebar.header("📥 Data Management")
uploaded_file = st.sidebar.file_uploader(
    "Upload New Exoplanet Data (CSV/FITS)", type=["csv", "fits"]
)
if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully! Ready for processing.")
    # Logic to load_data and preprocess_data would go here
    data = pd.read_csv(uploaded_file)  # Example for CSV

    if "df" not in st.session_state:
        st.session_state.df = data

    X, y = clean_data(st.session_state.df, target_col="koi_pdisposition")
    ##st.session_state.df = pd.concat([X, y], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = process(X_train, X_test)
    st.sidebar.info("Data cleaned and preprocessed. Ready for training.")


# Model Training/Retraining
st.sidebar.header("⚙️ Model Training")
train_btn = st.sidebar.button("Train/Retrain Model")
if train_btn:
    st.sidebar.info("Training initiated... Please wait.")
    my_svm = SVC()
    my_rf = RandomForestClassifier()
    my_lr = LogisticRegression()
    train_model(my_rf, X_train, y_train)
    train_model(my_svm, X_train, y_train)
    train_model(my_lr, X_train, y_train)

    y_pred_rf = predict_exoplanet(my_rf, X_test)
    y_pred_lr = predict_exoplanet(my_lr, X_test)
    y_pred_svm = predict_exoplanet(my_svm, X_test)
    st.session_state.df["rf_prediction"] = y_pred_rf
    st.session_state.df["lr_prediction"] = y_pred_lr
    st.session_state.df["svm_prediction"] = y_pred_svm

    save_model(my_svm, "models/svm_model.joblib")
    save_model(my_rf, "models/rf_model.joblib")
    save_model(my_lr, "models/lr_model.joblib")

    st.sidebar.success("Models trained and saved!")

st.sidebar.markdown("---")
st.sidebar.caption("Current Model Status: **Ready**")
st.sidebar.caption(f"Last Training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Main Content ---
st.title("🌌 Exoplanet Deduction System")
st.subheader("An interactive tool for classifying exoplanetary data.")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔭 Predict & Explore",
        "📊 Model Performance",
        "🔧 Hyperparameter Tuning",
        "View Table",
    ]
)

# -----------------
# TAB 1: Predict & Explore (For Novices and Quick Detections)
# -----------------
with tab1:
    st.header("Predict Exoplanet Status")
    st.markdown(
        "Enter or select parameters for a single candidate to get an instant deduction."
    )
    # feature inputs
    koi_score = st.number_input("Koi Score", min_value=0.0, max_value=1.0, value=0.5)
    koi_fpflag_nt = st.selectbox("Koi FPFlag NT", [0, 1])
    koi_fpflag_ss = st.selectbox("Koi FPFlag SS", [0, 1])
    koi_fpflag_co = st.selectbox("Koi FPFlag CO", [0, 1])
    koi_fpflag_ec = st.selectbox("Koi FPFlag EC", [0, 1])
    koi_period = st.number_input("Koi Period", min_value=0.0, value=10.0)
    koi_time0bk = st.number_input("Koi Time0bk", min_value=0.0, value=100.0)
    koi_impact = st.number_input("Koi Impact", min_value=0.0, value=0.5)
    koi_impact_err1 = st.number_input("Koi Impact Err1", min_value=0.0, value=0.1)
    koi_impact_err2 = st.number_input("Koi Impact Err2", min_value=0.0, value=0.1)
    koi_duration = st.number_input("Koi Duration", min_value=0.0, value=5.0)
    koi_duration_err1 = st.number_input("Koi Duration Err1", min_value=0.0, value=0.1)
    koi_duration_err2 = st.number_input("Koi Duration Err2", min_value=0.0, value=0.1)
    koi_depth = st.number_input("Koi Depth", min_value=0.0, value=500.0)
    koi_depth_err1 = st.number_input("Koi Depth Err1", min_value=0.0, value=10.0)
    koi_depth_err2 = st.number_input("Koi Depth Err2", min_value=0.0, value=10.0)
    koi_prad = st.number_input("Koi Prad", min_value=0.0, value=1.0)
    koi_prad_err1 = st.number_input("Koi Prad Err1", min_value=0.0, value=0.1)
    koi_prad_err2 = st.number_input("Koi Prad Err2", min_value=0.0, value=0.1)
    koi_teq = st.number_input("Koi Teq", min_value=0.0, value=300.0)
    koi_insol = st.number_input("Koi Insol", min_value=0.0, value=100.0)
    koi_insol_err1 = st.number_input("Koi Insol Err1", min_value=0.0, value=10.0)
    koi_insol_err2 = st.number_input("Koi Insol Err2", min_value=0.0, value=10.0)
    koi_model_snr = st.number_input("Koi Model SNR", min_value=0.0, value=10.0)
    koi_tce_plnt_num = st.number_input("Koi TCE Plnt Num", min_value=1, value=1)
    koi_steff = st.number_input("Koi Steff", min_value=2000, value=5500)
    koi_steff_err1 = st.number_input("Koi Steff Err1", min_value=0, value=100)
    koi_steff_err2 = st.number_input("Koi Steff Err2", min_value=0, value=100)
    koi_slogg = st.number_input("Koi Slogg", min_value=0.0, value=4.5)
    koi_srad = st.number_input("Koi Srad", min_value=0.0, value=1.0)
    koi_srad_err1 = st.number_input("Koi Srad Err1", min_value=0.0, value=0.1)
    koi_srad_err2 = st.number_input("Koi Srad Err2", min_value=0.0, value=0.1)
    ra = st.number_input("RA", min_value=0.0, value=290.0)
    dec = st.number_input("DEC", min_value=-90.0, max_value=90.0, value=45.0)
    koi_kepmag = st.number_input("Koi Kepmag", min_value=0.0, value=12.0)
    input_data = pd.DataFrame(
        {
            "koi_score": [koi_score],
            "koi_fpflag_nt": [koi_fpflag_nt],
            "koi_fpflag_ss": [koi_fpflag_ss],
            "koi_fpflag_co": [koi_fpflag_co],
            "koi_fpflag_ec": [koi_fpflag_ec],
            "koi_period": [koi_period],
            "koi_time0bk": [koi_time0bk],
            "koi_impact": [koi_impact],
            "koi_impact_err1": [koi_impact_err1],
            "koi_impact_err2": [koi_impact_err2],
            "koi_duration": [koi_duration],
            "koi_duration_err1": [koi_duration_err1],
            "koi_duration_err2": [koi_duration_err2],
            "koi_depth": [koi_depth],
            "koi_depth_err1": [koi_depth_err1],
            "koi_depth_err2": [koi_depth_err2],
            "koi_prad": [koi_prad],
            "koi_prad_err1": [koi_prad_err1],
            "koi_prad_err2": [koi_prad_err2],
            "koi_teq": [koi_teq],
            "koi_insol": [koi_insol],
            "koi_insol_err1": [koi_insol_err1],
            "koi_insol_err2": [koi_insol_err2],
            "koi_model_snr": [koi_model_snr],
            "koi_tce_plnt_num": [koi_tce_plnt_num],
            "koi_steff": [koi_steff],
            "koi_steff_err1": [koi_steff_err1],
            "koi_steff_err2": [koi_steff_err2],
            "koi_slogg": [koi_slogg],
            "koi_srad": [koi_srad],
            "koi_srad_err1": [koi_srad_err1],
            "koi_srad_err2": [koi_srad_err2],
            "ra": [ra],
            "dec": [dec],
            "koi_kepmag": [koi_kepmag],
        }
    )
    # load models
    my_svm = joblib.load("models/svm_model.joblib")
    my_rf = joblib.load("models/rf_model.joblib")
    my_lr = joblib.load("models/lr_model.joblib")
    btr = st.button("Submit for Prediction")
    if btr:
        # scaling
        scaler = StandardScaler()
        input_data = StandardScaler.fit_transform(input_data)
        st.write(f"svm {my_svm.predict(input_data)}")
        st.write(f"rf {my_rf.predict(input_data)}")
        st.write(f"lr {my_lr.predict(input_data)}")
# -----------------
# TAB 2: Model Performance (Statistics about Accuracy)
# -----------------
with tab2:
    model_name = st.selectbox(
        "Select Model to View Performance:",
        ["Support Vector Machine (SVM)", "Random Forest", "Logistic Regression"],
    )
    st.header("Current Model Statistics")
    st.markdown(
        "Review the accuracy and performance metrics of the currently loaded model."
    )

    if model_name == "Support Vector Machine (SVM)":
        my_svm = joblib.load("models/svm_model.joblib")
        my_svm_report_train = evaluate_model(my_svm, X_train, y_train)
        my_svm_report_test = evaluate_model(my_svm, X_test, y_test)
        report_train = my_svm_report_train
        report_test = my_svm_report_test
        st.dataframe(report_train)
        st.dataframe(report_test)
    elif model_name == "Random Forest":
        my_rf = joblib.load("models/rf_model.joblib")
        my_rf_report_train = evaluate_model(my_rf, X_train, y_train)
        my_rf_report_test = evaluate_model(my_rf, X_test, y_test)
        report_train = my_rf_report_train
        report_test = my_rf_report_test
        st.dataframe(report_train)
        st.dataframe(report_test)
    elif model_name == "Logistic Regression":
        my_lr = joblib.load("models/lr_model.joblib")
        my_lr_report_train = evaluate_model(my_lr, X_train, y_train)
        my_lr_report_test = evaluate_model(my_lr, X_test, y_test)
        report_train = my_lr_report_train
        report_test = my_lr_report_test
        st.dataframe(report_train)
        st.dataframe(report_test)
    st.markdown("**Note:** These metrics are based on the last training session.")


# -----------------
# TAB 3: Hyperparameter Tuning (For Researchers/Advanced Users)
# -----------------
with tab3:
    st.header("Model Hyperparameter Adjustment")
    st.markdown(
        "Modify the parameters for the selected model and **re-train** the system."
    )

    # Model Selection (if you have multiple models)
    model_choice = st.selectbox(
        "Select Model Type for Tuning:",
        ["Random Forest", "Support Vector Machine (SVM)", "logistic Regression"],
    )

    st.markdown("---")

    if model_choice == "Random Forest":
        st.subheader("Random Forest Parameters")
        n_estimators = st.number_input(
            "Number of Trees", min_value=10, value=100, step=10
        )
        max_depth = st.number_input("Max Depth", min_value=1, value=10)
        min_samples_split = st.number_input(
            "Min Samples Split", min_value=2, value=2, step=1
        )
        if st.button("Apply Random Forest Parameters and Retrain"):
            my_rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )
            train_model(my_rf, X_train, y_train)
            save_model(my_rf, "models/rf_model.joblib")
            st.success("Random Forest model retrained and saved with new parameters!")

    elif model_choice == "Support Vector Machine (SVM)":
        st.subheader("SVM Parameters")
        C = st.number_input("C (Regularization)", min_value=0.01, value=1.0)
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        gamma = st.selectbox("Gamma", ["scale", "auto"])
        if st.button("Apply SVM Parameters and Retrain"):
            my_svm = SVC(C=C, kernel=kernel, gamma=gamma)
            train_model(my_svm, X_train, y_train)
            save_model(my_svm, "models/svm_model.joblib")
            st.success("SVM model retrained and saved with new parameters!")
    elif model_choice == "logistic Regression":
        st.subheader("Logistic Regression Parameters")
        C = st.number_input("C (Regularization)", min_value=0.01, value=1.0)
        max_iter = st.number_input("Max Iterations", min_value=50, value=100)
        if st.button("Apply Logistic Regression Parameters and Retrain"):
            my_lr = LogisticRegression(C=C, max_iter=max_iter)
            train_model(my_lr, X_train, y_train)
            save_model(my_lr, "models/lr_model.joblib")
            st.success(
                "Logistic Regression model retrained and saved with new parameters!"
            )
    st.markdown(
        "**Note:** Retraining may take several minutes depending on the dataset size."
    )

with tab4:
    st.header("Exoplanet Data Table")
    st.markdown("View the raw exoplanet data used for training and predictions.")
    st.dataframe(st.session_state.df)
