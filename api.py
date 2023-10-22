#API made with Flask

#Importing Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import TextVectorization
import numpy as np

MODEL_PATH = f'./model/nlp_toxicity_model.h5'
VECTORIZER_PATH = f'./model/vectorizer_layer.pkl'

toxicity = ["Toxic: ", "Severe Toxic: ", "Obscene: ", "Threat: ", "Insult: ", "Identity Hate: "]

model=''

#Loading the Model
if model=='':
    model = tf.keras.models.load_model(
        (MODEL_PATH),
        custom_objects={'KerasLayer':hub.KerasLayer}
    )

#Vectorizer Text Function
def load_vectorizer(text) :
    from_disk = pickle.load(open(VECTORIZER_PATH, "rb"))
    vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=from_disk['config']['output_sequence_length'])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(from_disk['weights'])

    text_cv = vectorizer(text)
    return text_cv

#Prediction Function
def predict(text):
    pred = load_vectorizer(text)
    if (pred.shape == 1800) :
        res = model.predict(np.expand_dims(pred,0))
        print(res)

        prob_toxic = res[0][0] * 100
        prob_toxic = round(prob_toxic, 2)

        prob_severe_toxic = res[0][1] * 100
        prob_severe_toxic = round(prob_severe_toxic, 2)

        prob_obscene = res[0][2] * 100
        prob_obscene = round(prob_obscene, 2)

        prob_threat = res[0][3] * 100
        prob_threat = round(prob_threat, 2)

        prob_insult = res[0][4] * 100
        prob_insult = round(prob_insult, 2)

        prob_identity_hate = res[0][5] * 100
        prob_identity_hate = round(prob_identity_hate, 2)
        
        results = {
            "prob_toxic" : float(prob_toxic),
            "prob_severe_toxic": float(prob_severe_toxic),
            "prob_obscene": float(prob_obscene),
            "prob_threat" : float(prob_threat),
            "prob_insult": float(prob_insult),
            "prob_identity_hate": float(prob_identity_hate),
        }

        return results
    else :
        results = {
            "result": "Try Again"
        }

        return results

app = Flask(__name__)

#Flask App
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        np_array = np.array(data['array'])
        try:
            results = predict(np_array)
            print('Success!', 200)
            return jsonify({"Results": results})

    
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
