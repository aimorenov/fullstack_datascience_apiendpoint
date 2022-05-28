from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        print(req)
        # Check mandatory key (simple input key, expect user to load data in correct order)
        if "input" in req.keys():
            # Convert to dataframe
            df = pd.DataFrame(req["input"]).T
            # Load model
            regressor = joblib.load("models/reg_model.joblib")
            # Predict
            prediction = regressor.predict(df)
            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            prediction = str(round(prediction[0]))
            #response = {"prediction": prediction.tolist()[0]}
            #return response
            return jsonify({"The predicted rental price per day in euros is": prediction}), 200     
    return jsonify({"msg": "Error: not a JSON or no input key in your request"})



if __name__ == "__main__":
    app.run(debug=True)
