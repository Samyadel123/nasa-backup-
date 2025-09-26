# 🌌 Exoplanet Deduction System

An interactive Streamlit application for classifying exoplanetary data, designed for both educational exploration and research-grade analysis.

## Table of Contents
1.  [🚀 Overview](#-overview)
2.  [✨ Key Features](#-key-features)
3.  [💻 Installation & Setup](#-installation--setup)
4.  [🛠️ Usage Guide](#-usage-guide)
5.  [📂 Project Structure](#-project-structure)
6.  [🤝 Contributing](#-contributing)
7.  [© License](#-license)

---

## 🚀 Overview

The **Exoplanet Deduction System** is a dynamic, web-based tool built with **Streamlit** and machine learning models (e.g., Random Forest, SVM, XGboos,lightgbm) to streamline the process of classifying exoplanet candidates. It provides a simple UI for novices to explore data and powerful controls for researchers to manage data, tweak models, and analyze performance.

---

## ✨ Key Features

| Feature | Target Audience | Location | Description |
| :--- | :--- | :--- | :--- |
| **🔍 Interactive Prediction** | Novices & Researchers | Tab 1: Predict & Explore | Input candidate parameters and receive an **instant classification** (Planet / Not a Planet) with a confidence score. |
| **📥 Dynamic Data Ingestion** | Researchers | Sidebar | Use the file uploader to **ingest new CSV or FITS data** for model retraining. |
| **📈 Model Performance Metrics** | All Users | Tab 2: Model Performance | View essential metrics like **Accuracy, Precision, Recall,** and a detailed statistical breakdown. |
| **🔧 Hyperparameter Tuning** | Researchers | Tab 3: Hyperparameter Tuning | **Tweak core model settings** (e.g., $n\_estimators$, $max\_depth$) and initiate a re-training cycle directly from the interface. |
| **⚙️ On-Demand Training** | Researchers | Sidebar | Manually trigger the model's training pipeline using any newly uploaded data. |

---

## 💻 Installation & Setup

### Prerequisites

you should have uv 

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone git clone https://github.com/Samyadel123/nasa-backup-
    

2.  **Install dependencies:**
    *(Ensure your `requirements.txt` includes `streamlit`, `pandas`, `scikit-learn`, etc.)*
    ```bash
    uv sync
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The app will open automatically in your browser at `http://localhost:8501`.

---

## 🛠️ Usage Guide

The application is split into the **Sidebar** (for system control) and the **Main Window** (organized by tabs).

### 1. The Sidebar: Data & Training Controls

* **Upload Data:** Use the **"Upload New Exoplanet Data"** file uploader to feed new datasets into the system for processing.
* **Train/Retrain Model:** Click this button to execute the training pipeline, updating the system's model with either the newly uploaded data or an updated set of hyperparameters.

### 2. Main Window Tabs

#### 🔭 Predict & Explore
* Input the numerical parameters for a new candidate object.
* Click **"Deduce Candidate Status"** to see the classification result and confidence level.

#### 📊 Model Performance
* Review the high-level **Accuracy, Precision, and Recall** metrics.
* Examine visualizations (like a Confusion Matrix) and detailed **Model Configuration** metadata to understand the current model's state.

#### 🔧 Hyperparameter Tuning
* Select the model type (if multiple are available).
* Adjust the model's parameters using the provided sliders and selectors.
* Click **"Apply Changes and Retrain Model (Tuning)"** to train a new model instance with your specified configuration.

---

## 📂 Project Structure
