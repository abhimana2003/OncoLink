import pandas as pd
import subprocess
import numpy as np
import joblib

RESULTS_PATH = "outputs_metabric/model_results/model_comparison.csv"
DATA_PATH = "outputs_metabric/X_all_genes.csv"
LABEL_PATH = "outputs_metabric/y_labels.csv"


def load_results():
    return pd.read_csv(RESULTS_PATH)


def load_data():
    X = pd.read_csv(DATA_PATH)
    y = pd.read_csv(LABEL_PATH).iloc[:, 0]
    return X, y



def query_ollama(prompt: str, model: str = "llama3"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()


def explain_model_results(question: str):
    results_df = load_results()
    table_str = results_df.to_string(index=False)

    prompt = f"""
        You are an AI clinical data scientist working on predicting breast cancer treatment response.

        The goal of the project is:
        Predict whether a patient will respond to treatment using gene expression and clinical features.

        Here are model evaluation results:

        {table_str}

        Answer the following question:
        {question}

        Focus on:
        - Which model is best for predicting treatment response
        - Whether gene features or clinical features matter more
        - Practical implications for patient prediction
        """

    response = query_ollama(prompt)

    print("\n=== MODEL INSIGHT ===\n")
    print(response)


def explain_patient(index: int):
    import os

    # Load data
    X = pd.read_csv("outputs_metabric/X_all_genes.csv")
    y = pd.read_csv("outputs_metabric/y_labels.csv").iloc[:, 0]

    # Load trained model
    model_path = "outputs_metabric/model_results/best_model.pkl"

    if not os.path.exists(model_path):
        print("ERROR: Run model.py first to generate trained model.")
        return

    model = joblib.load(model_path)

    # Select patient
    patient_features = X.iloc[[index]]  # keep as DataFrame
    true_label = y.iloc[index]

    # Predict
    pred = model.predict(patient_features)[0]
    prob = model.predict_proba(patient_features)[0][1]

    # Prepare small summary for LLM
    feature_preview = patient_features.iloc[0].values[:20]

    prompt = f"""
        You are an AI clinical assistant helping interpret a machine learning prediction.

        Goal:
        Predict treatment response (1 = responder, 0 = non-responder).

        Prediction for this patient:
        - Predicted label: {pred}
        - Probability of response: {prob:.3f}
        - True label: {true_label}

        Sample of patient features:
        {feature_preview}

        Explain:
        1. What the prediction means
        2. Why the model might have made this prediction
        3. Role of gene vs clinical features
        4. Confidence and uncertainty
        5. Whether this prediction should influence treatment decisions
        """

    response = query_ollama(prompt)

    print(f"\n PATIENT {index} (REAL MODEL PREDICTION) \n")
    print(response)


def treatment_decision_support():
    results_df = load_results()

    best_model = results_df.iloc[0]

    prompt = f"""
        You are assisting in a clinical decision-support system.

        Best model result:
        {best_model.to_dict()}

        Answer:
        - Should this model be trusted for predicting treatment response?
        - What are its strengths?
        - What are its risks in a real clinical setting?
        - What additional data would improve predictions?

        Keep it realistic and grounded.
        """

    response = query_ollama(prompt)

    print("\n DECISION SUPPORT \n")
    print(response)


if __name__ == "__main__":
    explain_model_results("Which feature type is most useful for predicting treatment response?")
    explain_patient(index=5)
    treatment_decision_support()