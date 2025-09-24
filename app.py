import streamlit as st 
import pandas as pd
# change background color
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #faad;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("Nasa Exoplanet Toolkit")


Analyse , machine_tab, inference =  st.tabs(["📈 Analyse", "🤖 ML","🔮 inference"])



# this in tab 2
with machine_tab:
    data_set = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if data_set is not None:
        if data_set.name.endswith("csv"):
            df = pd.read_csv(data_set)
        else:
            df = pd.read_excel(data_set)
        st.dataframe(df.head())
        st.write(df.describe().T)
        st.write(df.info())
        st.write(df.isna().sum())
        
    model = st.selectbox(
        "choice your model"
        ,("Logistic Regression", "SVC", "LGBMClassifier")
    )

    st.write("You selected:", model)

### inference tab
with inference:
    st.write("This is the inference tab.")
    uploaded_file = st.file_uploader("Choose a file for inference", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith("csv"):
            df_infer = pd.read_csv(uploaded_file)
        else:
            df_infer = pd.read_excel(uploaded_file)
        st.dataframe(df_infer.head())
        st.write("Inference will be done here.")
        

### analyse tab
with Analyse:
    st.write("This is the analysis tab.")
    uploaded_file = st.file_uploader("Choose a file for analysis", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith("csv"):
            df_analyse = pd.read_csv(uploaded_file)
        else:
            df_analyse = pd.read_excel(uploaded_file)
        st.dataframe(df_analyse.head())
        st.write("Data analysis will be done here.")