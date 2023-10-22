import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from PIL import Image
import requests

#URL of the API made with Flask
API = "http://127.0.0.1:5000"

IMG_SIDEBAR_PATH = "./assets/img.jpg"

#Loading the NLP Model
model=''

if model=='':
    model = tf.keras.models.load_model(
        ("./model/nlp_toxicity_model.h5"),
        custom_objects={'KerasLayer':hub.KerasLayer}
    )

toxicity = ["Toxic: ", "Severe Toxic: ", "Obscene: ", "Threat: ", "Insult: ", "Identity Hate: "]

#Vectorizer Function
def load_vectorizer(text) :
    from_disk = pickle.load(open("./model/vectorizer_layer.pkl", "rb"))
    vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=from_disk['config']['output_sequence_length'])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(from_disk['weights'])

    text_cv = vectorizer(text)
    return text_cv

#Prediction Function
def prediction(model, text) :
    data = {'array': text}

    resp = requests.post(API, json=data)

    pred = load_vectorizer(text)

    st.header("Results of the Analysis ✅:")
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.4,1, 1],  gap="medium")

    if (pred.shape == 1800) :
        #res = model.predict(np.expand_dims(pred,0))
        #print(res)

        col1.subheader("Insult 💬:")
        col2.subheader("Percentage 📊:")

        col3.subheader("Sentiment Analysis Completed ✔️")
        col3.image("./assets/bad_words.jpg", width=430)
        col3.write("`Be aware that this NLP App Should Not be used as a Substitute for a Final Diagnosis and Prediction.`")

        res = [resp.json()["Results"]["prob_toxic"], resp.json()["Results"]["prob_severe_toxic"], resp.json()["Results"]["prob_obscene"], resp.json()["Results"]["prob_threat"], resp.json()["Results"]["prob_insult"], resp.json()["Results"]["prob_identity_hate"]]

        for i in range(0, len(toxicity)) :
            p = res[i] 
            answer = "%.2f" % p + "%"

            col1.error(toxicity[i])
            col2.success(answer)
        
        st.markdown("<hr/>", unsafe_allow_html=True)

    else :
        st.error("Try Again")

#Streamlit Sidebar
def add_sidebar():
  st.sidebar.header("Analysis of Toxicity `App 🗨️`")
  image = np.array(Image.open(IMG_SIDEBAR_PATH))
  st.sidebar.image(image)

  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.write("This Artificial Intelligence App can Analysis if a given Phrase is considered Toxic, Severe Toxic, Obscene, Threat, Insult or Identity Hate.")

  text = st.sidebar.text_input("Write Something:")
  st.sidebar.button("Analyze Phrase ✅", on_click=prediction, args=(model, text))
  st.sidebar.write("`This Artificial Intelligence can Assist for any Analysis about the Toxicy of a Text, but Should Not be used as a Substitute for a Final Diagnosis and Prediction.`")

  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.markdown('''
  🧑🏻‍💻 Created by [Luis Jose Mendez](https://github.com/mendez-luisjose).
  ''')

def main() :  
    st.set_page_config(
        page_title="Analysis of Toxicity",
        layout="wide",
        initial_sidebar_state="expanded"
    )
  
    add_sidebar()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )    

    with st.container() :
        st.title("NLP Toxicity Analyzer 🗣️")
        st.write("This Natural Language Processing Machine Learning Model can understand and analyze if a text is considered Toxic, Severe Toxic, Obscene, Threat, Insult or Identity Hate. The NLP Model show the porcentage of the analysis.")
        _, col, _ = st.columns([0.4, 1, 0.2])
        col.image("./assets/nlp.png", width=550)
        st.markdown("<hr/>", unsafe_allow_html=True)


if __name__ == "__main__" :
    main()

    print("App Running!")


