from flask import Flask, render_template, flash, request
from get_model import *
import numpy as np

MODEL_PATH = "models/logistic_red.sav"
model = LoadModel(MODEL_PATH)

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST','GET'])
def predict():
    name = request.form['name']
    pregnant = request.form['pregnant']
    glucose = request.form['glucose']
    bp = request.form['bp']
    skin = request.form['skin']
    insulin = request.form['insulin']
    bmi = request.form['bmi']
    dp = request.form['dp']
    age = request.form['age']
    # arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
    prediction = model.predict_class(pregnant,glucose,bp,skin,insulin,bmi,dp,age)
    print(prediction)
    # return render_template('one.html')
    # int_features = [int(x) for x in request.form.values()]
    # final = [np.array(int_features)]
    # prediction = model.predict_class(final)

    if prediction == 0:
        return render_template("one.html", result="true")
    elif prediction == 1:
        return render_template("two.html", result="true")
    


if __name__ == "__main__":
        app.run()
