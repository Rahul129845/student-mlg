from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder

model_names = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "SVC"
]


loaded_models = {}
for model_name in model_names:
    model_path = f"{model_name}.pkl"
    with open(model_path, 'rb') as file:
        loaded_models[model_name] = pickle.load(file)
       

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')  # The HTML file above


@app.route('/predict_<model_name>', methods=['POST'])
def predict_model(model_name):
    # Step 1: Collect input data
    input_data = request.form.to_dict()  # Collect form data as a dictionary
    df_input = pd.DataFrame([input_data])  # Convert input data into a DataFrame

    # Step 2: Encode categorical features
    encode = LabelEncoder()
    df_categorical = []
    for column in df_input.columns:
        if df_input[column].dtypes == "object":
            df_categorical.append(column)
    for column in df_categorical:
        df_input[column] = encode.fit_transform(df_input[column])

    # Step 3: Convert data types to numeric
    df_input = df_input.apply(pd.to_numeric)

    # Step 4: Scale numeric features (if required)
    scaler = StandardScaler()
    df_input_scaled = scaler.fit_transform(df_input)

    # Step 5: Prepare final array for model input
    final_features = np.array(df_input_scaled)

    # Step 6: Model prediction
    prediction = None
    if model_name == "logistic":
        prediction = loaded_models["LogisticRegression"].predict(final_features)
    elif model_name == "decision_tree":
        prediction = loaded_models["DecisionTreeClassifier"].predict(final_features)
    elif model_name == "random_forest":
        prediction = loaded_models["RandomForestClassifier"].predict(final_features)
    elif model_name == "gradient_boosting":
        prediction = loaded_models["GradientBoostingClassifier"].predict(final_features)
    elif model_name == "adaboost":
        prediction = loaded_models["AdaBoostClassifier"].predict(final_features)
    elif model_name == "svc":
        prediction = loaded_models["SVC"].predict(final_features)
    elif model_name == "ensemble":
        with open("Ensemble.pkl","rb") as file:
            ensemble_model=pickle.load(file)
        prediction=ensemble_model.predict(final_features)
    output = 'Depressed' if prediction[0] == 1 else 'Not Depressed'
    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)







