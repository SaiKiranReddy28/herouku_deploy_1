import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import taxi_price_model
app=Flask(__name__)
model=pickle.load(open("model.sav","rb"))

@app.route('/')
def home():
    return render_template("index.html")
    
@app.route('/predict', methods=["POST"])
def predict():

    int_features=[float(x) for x in request.form.values()]
    val=taxi_price_model.haversine(int_features[0],int_features[1],int_features[2],int_features[3])
    
    int_features.append(val)
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)

    return render_template("index.html",prediction_text="The price for the trip is ${}".format(output),distance="The approx distance of travel is {}".format(val))


if __name__=="__main__":
    app.run(debug=True)
    
    

