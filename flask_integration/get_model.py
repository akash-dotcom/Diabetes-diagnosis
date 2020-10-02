import pickle
import pandas as pd


class LoadModel:
    def __init__(self, MODEL_PATH):
        self.loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
        

    def predict_class(self, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age):
        data = [[Pregnancies, Glucose, BloodPressure,
                 SkinThickness, Insulin, BMI, DiabetesPedigree, Age]]

        df = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                         'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age'])
        new_pred = self.loaded_model.predict(df)
        return new_pred


if __name__ == "__main__":

    MODEL_PATH = "models/logistic_red.sav"
    model = LoadModel(MODEL_PATH)
    predicted_class = model.predict_class(1, 85, 66, 29, 0, 26.6, 0.351, 31)
    print(predicted_class)
    