# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import flask as flask

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to fraudelent prediction process"


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False, "predictions": []}

    # ensure an image was properly uploaded to our endpoint
    param_dict = {'gender': [False],
                  'tenure': [0],
                  'PhoneService': [False],
                  'TotalCharges': [0.0],
                  'StreamingMovies': [False],
                  'StreamingTV': [False],
                  'TechSupport': [False]}

    if flask.request.method == "POST":
        request_json = flask.request.get_json()
        for key, val in param_dict.items():
            if request_json[key]:
                param_dict[key] = [request_json[key]]

        preds = model.predict_proba(pd.DataFrame(param_dict))
        data["predictions"] = preds[:, 1][1]
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    #model_path = "./models/model_curs.dill"
    model_path = "models/model_curs.dill"
    load_model(model_path)
    app.run()
