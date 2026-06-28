from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "https://jarvis0852-diabetes_prediction_ai_ml.hf.space/predict"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html', prediction=None)

    try:
        data = {}

        # Gender
        data["gender"] = request.form["gender"]

        # Age
        data["age"] = float(request.form["age"])

        # Binary values
        data["hypertension"] = int(request.form["hypertension"])
        data["heart_disease"] = int(request.form["heart_disease"])

        # Smoking
        data["smoking_history"] = request.form["smoking_history"]

        # BMI
        data["bmi"] = float(request.form["bmi"])

        # HbA1c
        data["HbA1c_level"] = float(request.form["HbA1c_level"])

        # Blood Glucose
        data["blood_glucose_level"] = float(request.form["blood_glucose_level"])

        print("\n========================")
        print("Sending JSON to API:")
        print(data)
        print("========================")

        response = requests.post(
            API_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print("\n========== API RESPONSE ==========")
        print("Status Code:", response.status_code)
        print("Headers:", response.headers)
        print("Body:", response.text)
        print("==================================\n")

        if response.status_code == 200:
            result = response.json()

            prediction = result.get("prediction", None)

            return render_template(
                "index.html",
                prediction=[prediction],
                error=None
            )

        return render_template(
            "index.html",
            prediction=None,
            error=f"API Error {response.status_code}: {response.text}"
        )

    except Exception as e:
        print("Exception:", e)
        return render_template(
            "index.html",
            prediction=None,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
