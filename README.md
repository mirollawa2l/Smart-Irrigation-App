# 🌱 Smart Irrigation App

An interactive machine learning application designed to optimize agricultural water usage. The system analyzes environmental and soil conditions to predict irrigation needs and classifies them as **Low**, **Medium**, or **High**.

The project explores multiple machine learning models including **Logistic Regression**, **Random Forest**, and **XGBoost**, with the best-performing model saved and deployed through a Streamlit web application.

---

## ✨ Features

- 🤖 Predict irrigation requirements using trained ML models
- 🔁 Compare multiple algorithms (Logistic Regression, Random Forest, XGBoost)
- 🌡️ Input environmental and soil parameters through an interactive UI
- ⚡ Real-time prediction results
- 📊 Simple visual feedback for predictions
- 🌐 Web-based interface using Streamlit

---

## 🛠️ Tech Stack

| Category             | Technologies                                               |
| -------------------- | ---------------------------------------------------------- |
| Programming Language | Python                                                     |
| Data Processing      | Pandas, NumPy                                              |
| Machine Learning     | Logistic Regression, Random Forest, XGBoost (Scikit-Learn) |
| Web Framework        | Streamlit                                                  |
| Model Serialization  | Joblib                                                     |
| Visualization        | Matplotlib, Plotly                                         |

---

## 📁 Repository Structure

```plaintext
Smart-Irrigation-App/
│
├── app.py                           # Streamlit web application
├── train_model.py                   # Model training & comparison script
├── code.ipynb                       # EDA and experiments notebook
├── xgboost_irrigation_model.joblib  # Trained (best) model
├── playground-series-s6e4.zip       # Dataset
└── README.md
```

---

## 📊 Dataset

This project uses the **Kaggle Playground Series Season 6 Episode 4 (S6E4)** dataset.

It contains environmental and soil-related features used to predict irrigation requirements.

---

## 🧠 Machine Learning Models

The following models were trained and compared:

- 📈 Logistic Regression
- 🌲 Random Forest
- ⚡ XGBoost

The best-performing model (based on evaluation during training) was saved as:

```text
xgboost_irrigation_model.joblib
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/mirollawa2l/Smart-Irrigation-App.git
cd Smart-Irrigation-App
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🧪 Model Training

To retrain and compare models:

```bash
python train_model.py
```

This script:

- Trains Logistic Regression
- Trains Random Forest
- Trains XGBoost
- Compares performance
- Saves the best model as `xgboost_irrigation_model.joblib`

---

## 💻 Usage

1. Open the web app
2. Enter environmental and soil values:
   - Soil moisture
   - Temperature
   - Humidity
   - Other features

3. Click **Predict**
4. Get irrigation level:
   - 🟢 Low
   - 🟡 Medium
   - 🔴 High

---

## 📓 Notebook

`code.ipynb` includes:

- Data exploration (EDA)
- Feature engineering
- Model training experiments
- Comparison between ML models

---

## 👨‍💻 Author

**Mirolla Wael**

Machine Learning Engineer & AI Developer

GitHub: https://github.com/mirollawa2l

---

## 🙏 Acknowledgments

- Kaggle Playground Series S6E4
- Scikit-Learn
- XGBoost
- Streamlit
- Pandas & NumPy
- Open-source ML community
