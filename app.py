from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    weather = None
    image = None

    if request.method == "POST":

        # Get input values
        humidity = float(request.form["humidity"])
        pressure = float(request.form["pressure"])
        windspeed = float(request.form["windspeed"])
        cloudcover = float(request.form["cloudcover"])

        # Create DataFrame with feature names
        features = pd.DataFrame(
            [[humidity, pressure, windspeed, cloudcover]],
            columns=[
                "Humidity",
                "Pressure",
                "WindSpeed",
                "CloudCover"
            ]
        )

        # Predict temperature
        prediction = model.predict(features)[0]

        # Select weather condition and image

        if cloudcover <= 20 and humidity <= 60:
            weather = "Sunny"
            image = "https://images.unsplash.com/photo-1601297183305-6df142704ea2?w=800"

        elif cloudcover <= 50 and humidity <= 75:
            weather = "Partly Cloudy"
            image = "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=800"

        elif humidity >= 80 and cloudcover >= 70:
            weather = "Rainy"
            image = "https://images.unsplash.com/photo-1519692933481-e162a57d6721?w=800"

        else:
            weather = "Cloudy"
            image = "https://images.unsplash.com/photo-1534274988757-a28bf1a57c17?w=800"

    return render_template(
        "index.html",
        prediction=prediction,
        weather=weather,
        image=image
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )