from flask import Flask,render_template, request
import numpy as np
import pickle

model = pickle.load(open('insurence.pkl',"rb"))


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    final_feat = list(request.form.values())
    #final_Feat = [i for i in final_feat]
    final_feat = np.array(final_feat).reshape(1,6)

    output = model.predict(final_feat)[0]

    return render_template("predict.html",output=output)

if __name__ =="__main__":
    app.run(port=8000,host='0.0.0.0',debug=True)
