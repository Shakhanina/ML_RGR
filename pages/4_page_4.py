import streamlit as st 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, rand_score
import pickle
import tensorflow as tf
import numpy as np

def load_models():
    gini_clf = pickle.load(open("models/gini_clf.pick", 'rb'))
    kmeans = pickle.load(open("models/kmeans.pick", 'rb'))
    grad_clf = pickle.load(open("models/grad_clf.pick", 'rb'))
    stack_clf = pickle.load(open("models/stack_clf.pick", 'rb'))
    bag_clf = pickle.load(open("models/bag_clf.pick", 'rb'))
    dnn_clf = tf.keras.models.load_model("models/dnn_clf.h5")
    return gini_clf, kmeans, grad_clf,bag_clf, stack_clf, dnn_clf

st.title("Модели")

uploaded_file = st.file_uploader("Choose a file .csv", type='csv')
df = pd.read_csv(uploaded_file)
st.write(df.head())

scaler = StandardScaler().fit(df.drop(['bomb_planted'], axis=1))
X = scaler.transform(df.drop(['bomb_planted'], axis=1))
X = pd.DataFrame(X)
Y = df['bomb_planted']
rus = RandomUnderSampler()

X_resampled, y_resampled = rus.fit_resample(X, Y)

st.write(y_resampled.shape)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.3, random_state=42, stratify=y_resampled)

input_data = {}
feature_names = df.drop("bomb_planted", axis=1).columns
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=None, placeholder="Type a number...")

button1 = st.button('Сделать предсказание')
button2 = st.button('Проверить на тестах')

gini_clf, k_means, grad_clf,bag_clf, stack_clf, dnn = load_models()

if button1:
    result = []
    input_data = pd.DataFrame([input_data])
    input_data = scaler.transform(input_data)
    
    predict_gini_clf = gini_clf.predict(input_data)[0]
    predict_k_means = k_means.predict(input_data)[0]
    predict_grad = grad_clf.predict(input_data)[0]
    predict_bag = bag_clf.predict(input_data)[0]
    predict_stack = stack_clf.predict(input_data)[0]
    predict_dnn = dnn.predict(input_data)[0]
    
    st.write(f'Результат предсказания DesicionTreeClasiffier - {predict_gini_clf}')
    st.write(f'Результат предсказания K-Means - {predict_k_means}')
    st.write(f'Результат предсказания GradientBoostingClassifier - {predict_grad}')
    st.write(f'Результат предсказания BaggingClassifier - {predict_bag}')
    st.write(f'Результат предсказания StackingClassifier - {predict_stack}')
    st.write(f'Результат предсказания DNN - {predict_dnn}')
     
elif button2:
    predict_knn = gini_clf.predict(X_test)
    predict_grad = grad_clf.predict(X_test)
    predict_bag = bag_clf.predict(X_test)
    predict_stack = stack_clf.predict(X_test)
    predict_dnn = dnn.predict(X_test)
    
    st.write(f'Результат метрики f1 модели DesicionTreeClasiffier- {f1_score(y_test, predict_knn)}')
    st.write(f'Результат метрики rand модели KMeans - {rand_score(k_means.labels_, Y)}')
    st.write(f'Результат метрики f1 модели GradientBoostingClassifier - {f1_score(y_test, predict_grad)}')
    st.write(f'Результат метрики f1 модели BaggingClassifier -{f1_score(y_test, predict_bag)}')
    st.write(f'Результат метрики f1 модели StackingClassifier - {f1_score(y_test, predict_stack)}')
    st.write(f'Результат метрики f1 модели DNN - {f1_score(y_test, np.around(dnn.predict(X_test, verbose=None)))}')
