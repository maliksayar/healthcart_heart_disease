from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open("heart_disease_predictor.pkl", "rb") as f:
    bagging_model = pickle.load(f)

# Load OneHotEncoder
with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def heart():

    if request.method == "POST":

        # ==========================
        # Numerical Features
        # ==========================

        num_features = pd.DataFrame({

            "Height_(cm)": [float(request.form["Height_(cm)"])],
            "Weight_(kg)": [float(request.form["Weight_(kg)"])],
            "BMI": [float(request.form["BMI"])],
            "Alcohol_Consumption": [float(request.form["Alcohol_Consumption"])],
            "Fruit_Consumption": [float(request.form["Fruit_Consumption"])],
            "Green_Vegetables_Consumption": [float(request.form["Green_Vegetables_Consumption"])],
            "FriedPotato_Consumption": [float(request.form["FriedPotato_Consumption"])]

        })


        # ==========================
        # Categorical Features
        # ==========================

        cat_features = pd.DataFrame({

            "General_Health":[request.form["General_Health"]],
            "Checkup":[request.form["Checkup"]],
            "Exercise":[request.form["Exercise"]],
            "Skin_Cancer":[request.form["Skin_Cancer"]],
            "Other_Cancer":[request.form["Other_Cancer"]],
            "Depression":[request.form["Depression"]],
            "Diabetes":[request.form["Diabetes"]],
            "Arthritis":[request.form["Arthritis"]],
            "Sex":[request.form["Sex"]],
            "Age_Category":[request.form["Age_Category"]],
            "Smoking_History":[request.form["Smoking_History"]]

        })


        # ==========================
        # One Hot Encode
        # ==========================

        encoded = ohe.transform(cat_features)

        # convert sparse -> dense if needed
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()


        # ==========================
        # Merge Numeric + Encoded
        # Numeric FIRST
        # ==========================

        final_features = np.hstack((num_features.values, encoded))


        # ==========================
        # Prediction
        # ==========================

        prediction = bagging_model.predict(final_features)[0]

        if prediction == 1:
            result = "High Risk of Heart Disease"
        else:
            result = "Low Risk of Heart Disease"

        return render_template(
            "index.html",
            prediction_text=result
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)