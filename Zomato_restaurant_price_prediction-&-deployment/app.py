#flask using pickle

import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__) 
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your rating is : {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)


