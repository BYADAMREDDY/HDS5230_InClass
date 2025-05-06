## Data Science - Group Project (Heart Stroke Prediction)
-----------
### Group Members names
- Bala Sai Phani Krishna Yadamreddy
- Dhamodhar Reddy Atla
- Mrunal Reddy Ragi
------------

[Link to the Video Demo of Project (Meeting passcode: JX=7i37s)](https://slu.zoom.us/rec/share/yDQFcJxymNr7upxJqI4Hg0uK-wLcLF_FpyBEt4roN-10xO7jP3u7bEMdYWarS_st.KQp8wnY-UjOHMBVG)


This project is a machine learning-powered application designed to predict whether a patient is at risk of having a stroke, based on various health parameters. It supports multiple classification algorithms and offers both model evaluation and a user-friendly web interface for prediction.

## 📁 Project Structure

```
├── heart_stroke_model_export.py   # Trains multiple models and saves them
├── heart_stroke_gradio.py     # Gradio-based UI for stroke prediction
├── models.pickle              # Serialized trained models
├── scaler.pickle              # Serialized MinMaxScaler
├── healthcare-dataset-stroke-data.csv  # Input dataset
└── README.md                  # Project documentation
```

---

## 🚀 Features

* Preprocessing and feature encoding
* Model training and hyperparameter tuning using `GridSearchCV`
* Multiple classifiers: Random Forest, Logistic Regression, SVM, Decision Tree, KNN
* Evaluation via accuracy, confusion matrix, and classification report
* Saves best models and scaler using `pickle`
* User interface built using **Gradio** for live predictions

---

## 🧪 Model Training and Evaluation

Run the following script to:

* Load and clean the dataset
* Encode categorical features
* Scale features
* Train and evaluate models
* Save the best-trained models and scaler to `.pickle` files

```bash
python heart_stroke_model_export.py
```

### Output

* `models.pickle`: Contains all trained models
* `scaler.pickle`: Contains the fitted MinMaxScaler
* Visual comparison of model scores and training durations

---

## 🖥️ Live Prediction Interface

To launch the Gradio-based UI for live predictions:

```bash
python heart_stroke_gradio.py
```

### UI Features:

* Select the machine learning model to use
* Input patient data like age, gender, glucose level, BMI, etc.
* Get instant prediction result: **"Stroke"** or **"No Stroke"**

---

## 📊 Dataset Info

* **Source**: `healthcare-dataset-stroke-data.csv`
* **Target Variable**: `stroke`
* **Features**:

  * `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`
  * `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

---

## 📦 Dependencies

Install required libraries using:

```bash
pip install -r requirements.txt
```

### Requirements (main)

* `pandas`
* `numpy`
* `scikit-learn`
* `seaborn`
* `matplotlib`
* `gradio`

---

## 🔒 Notes

* Unknown or missing values are dropped during data loading.
* Categorical values are encoded manually using dictionary mapping.
* `MinMaxScaler` is used to normalize features before model training and prediction.
* Models are trained with hyperparameter tuning using 10-fold cross-validation.

---

## 🧠 Example Use Case

1. A healthcare provider inputs patient data into the Gradio app.
2. The app scales and encodes the data based on training logic.
3. The chosen ML model predicts stroke risk and displays the result instantly.

