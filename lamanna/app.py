from flask import flask, request, jsonify

app = flask(__name__)

df_prediction = pd.read_csv('predict.csv')

@app.route('/infer', methods=('POST'))
def hello():
    data = request.het_json()
    param1 = data.get('param1')

    response_data = {
        'result': {
            'value': infer_result
        }
    }


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