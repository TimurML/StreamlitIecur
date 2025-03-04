import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

model = joblib.load('svm_model.pkl')
scaler = joblib.load('robust_scaler.pkl')

@st.dialog("Результат прогнозирования:")
def but_func(input_data):
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    match prediction[0]:
        case 1:
            st.markdown("# Вы здоровы!")
        case 2:
            st.markdown("# У вас Гепатит С")
        case 3:
            st.markdown("# У вас фиброз")
        case 4:
            st.markdown("# У вас Цирроз")

    

st.title("Диагностика заболеваний печени")
st.subheader("Как с этим работать?")
st.write("Для начала вам необходимо пройти лабораторное обследование.\nАнализы необходимые для прогнозирования имеются ниже.\nВам необходимо ввести данные анализов в каждом пункте, затем нажать на кнопку ПРОГНОЗИРОВАНИЕ.")

st.title("Признаки для прогнозирования")

col1, col2= st.columns([1, 2])
with col1:
    sex = st.selectbox("Пол", ['Мужчина', 'Женщина'])
with col2:
    age = st.slider("Возраст", min_value=15.0, max_value=100.0,
                            step=1.0)
    
col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1])   
with col3:
    alb = st.number_input("ALB", min_value=1.0, max_value=100.0,
                         step=0.1, help="Уровень альбумина в крови (г/л)")
    che	= st.number_input("CHE", min_value=1.000, max_value=20.000,
                         step=0.001, help="Уровень холинэстеразы в крови (Ед/л)")
with col4:
    alp	= st.number_input("ALP", min_value=10.0, max_value=500.0,
                         step=0.1, help="Уровень щелочной фосфатазы в крови (Ед/л)")
    chol = st.number_input("CHOL", min_value=2.6, max_value=10.3,
                          step=0.1, help="Уровень общего холестерина в крови (ммоль/л)")
with col5:
    alt = st.number_input("ALT", min_value=1.0, max_value=300.0,
                         step=0.1, help="Уровень аланина аминотрансферазы в крови (Ед/л)")
    crea = st.number_input("CREA", min_value=9.0, max_value=884.0,
                          step=1.0, help="Уровень гамма-глутамилтрансферазы в крови (мкмоль/л)")
with col6:
    ast = st.number_input("AST", min_value=1.0, max_value=200.0,
                         step=0.1, help="Уровень аспартата аминотрансферазы в крови (Ед/л)")
    ggt	= st.number_input("GGT", min_value=1.0, max_value=500.0,
                          step=0.1, help="Уровень креатинина в крови (Ед/л)")
with col7:
    bil	= st.number_input("BIL", min_value=2.0, max_value=342.0,
                         step=0.1, help="Уровень билирубина в крови (мкмоль/л)")
    prot = st.number_input("PROT", min_value=40.0, max_value=120.0,
                         step=1.0, help="Уровень общего белка в крови (г/л)")

sex_mapping = {'Мужчина': 1, 'Женщина': 0}
sex_numeric = sex_mapping[sex]
input_data = np.array([[age,sex_numeric,alb,alp,alt,ast,bil,che,chol,crea,ggt,prot]])

cont = st.container()
cont.markdown("""
<style>
.custom-text {
    font-size: 20px;
    color: red;
}
</style>
<p class="custom-text">Перед нажатием на кнопку, убедитесь в том, что вы заполнили все поля!</p>
""", unsafe_allow_html=True)
cont.button("ПРОГНОЗИРОВАНИЕ", on_click=but_func, args=(input_data, ))