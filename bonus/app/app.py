from flask import Flask, request
from flask import render_template, make_response
import os 

import numpy as np
import pickle 
import os


APP = Flask(__name__)

@APP.route('/')
def index():
    return render_template('indexx.html')
    
@APP.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('indexx.html', prediction_text='The price for this house is :{} $'.format(output))

if __name__ == '__main__':

    APP.run(host='0.0.0.0',port=5000,debug=True)
