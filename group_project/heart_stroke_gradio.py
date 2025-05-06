import pandas as pd
import seaborn as sns
import warnings
import gradio as gr
import pickle

with open("models.pickle", mode = 'rb') as buffer_read:
    trained_models = pickle.load(buffer_read)

with open("scaler.pickle", mode = 'rb') as buffer_read:
    scaler = pickle.load(buffer_read)


def predict_stroke(
    model_name, gender, age, hypertension, heart_disease, ever_married,
    work_type, Residence_type, avg_glucose_level, bmi, smoking_status
):
    input_data = pd.DataFrame([{
        'gender': {'Male': 0, 'Female': 1, 'Other': 2}[gender],
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': {'Yes': 0, 'No': 1}[ever_married],
        'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}[work_type],
        'Residence_type': {'Urban': 0, 'Rural': 1}[Residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}[smoking_status]
    }])
    
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
    model = trained_models[model_name]
    prediction = model.predict(input_scaled)[0]
    return "Stroke" if prediction == 1 else "No Stroke"

def main():
    warnings.filterwarnings("ignore")
    sns.set_style("darkgrid")
    pd.set_option('display.max_columns', None)

    model_dropdown = gr.Dropdown(
        list(trained_models.keys()), label="Select Model", value="RandomForestClassifier"
    )

    gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
    age = gr.Slider(0, 100, value=50, label="Age")
    hypertension = gr.Radio([0, 1], label="Hypertension (0: No, 1: Yes)")
    heart_disease = gr.Radio([0, 1], label="Heart Disease (0: No, 1: Yes)")
    ever_married = gr.Radio(["Yes", "No"], label="Ever Married")
    work_type = gr.Dropdown(["Private", "Self-employed", "Govt_job", "children", "Never_worked"], label="Work Type")
    residence_type = gr.Radio(["Urban", "Rural"], label="Residence Type")
    avg_glucose_level = gr.Slider(50, 300, value=100, label="Average Glucose Level")
    bmi = gr.Slider(10, 60, value=25, label="BMI")
    smoking_status = gr.Dropdown(["formerly smoked", "never smoked", "smokes", "Unknown"], label="Smoking Status")

    inputs = [
        model_dropdown, gender, age, hypertension, heart_disease, ever_married,
        work_type, residence_type, avg_glucose_level, bmi, smoking_status
    ]

    gr.Interface(
        fn=predict_stroke,
        inputs=inputs,
        outputs="text",
        title="Stroke Prediction UI",
        description="Select model and input patient data to predict stroke."
    ).launch()

    
if __name__ == "__main__":
    main()
