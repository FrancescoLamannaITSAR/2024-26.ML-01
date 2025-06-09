from flask import Flask, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)

with open("lamanna/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()

    try:
        features = [
            data['pos_perc'],
            data['release_date'],
            data['median_playtime'],
            data['price'],
            data['Genre']['Action'],
            data['Genre']['Adventure'],
            data['Genre']['Casual'],
            data['Genre']['Early Access'],
            data['Genre']['Free to Play'],
            data['Genre']['Indie'],
            data['Genre']['Massively Multiplayer'],
            data['Genre']['RPG'],
            data['Genre']['Racing'],
            data['Genre']['Simulation'],
            data['Genre']['Sports'],
            data['Genre']['Strategy'],
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400

    X = np.array([features])
    prediction = model.predict(X)[0]
    # Costruisci risposta
    response_data = {
        'result': {
            'value': prediction
        }
    }
    return jsonify(response_data)


'''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    name = data.get('name', 'Stranger')

    # mymodel = joblib.load("..../model.joblib")
    # inter_result = mymodel.predict(param1)
    # response_data = {
    #     "result" : {
    #         "value" : infer_result
    #     }
    # }
    # return jsonify(response_data)
    
    return jsonify({"message": "Hello {}!".format(name)})

@app.route('/infer', methods=['GET'])
def hi():
    # data = request.get_json()
    # name = data.get('name', 'Stranger')
    return "<h1> Hello Stranger! </h1>"

if __name__ == '__main__':
    app.run(debug=True)
'''