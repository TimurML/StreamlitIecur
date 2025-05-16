import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

model_svm = joblib.load('svm_model.pkl')
scaler_hep = joblib.load('robust_scaler.pkl')
model_mlp = joblib.load('mlp_model_cirr.pkl')
scaler_cirr = joblib.load('robust_scaler_cirr.pkl')

if 'state' not in st.session_state:
    st.session_state.state = 0

#--------------------------------FUNCTIONS----------------------------------------
def prediction_hep(input_data):
    input_data = scaler_hep.transform(input_data)
    prediction = model_svm.predict(input_data)
    st.session_state.state = prediction[0]

def prediction_cirr(input_data):
    input_data = scaler_cirr.transform(input_data)
    prediction = model_mlp.predict(input_data)
    st.session_state.state = prediction[0] + 6

def back():
    st.session_state.state = 0

def let_cirr():
    st.session_state.state = 5
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
if st.session_state.state == 0:
    st.title("Диагностика заболеваний печени")
    st.markdown("##### Прогнозируется наличие Гепатита С, а так же стадий поражений печени: Фиброза и Цирроза")

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
        bil	= st.number_input("BIL", min_value=0.0, max_value=342.0,
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
    cont.button("ПРОГНОЗИРОВАНИЕ", on_click=prediction_hep, args=(input_data,))

#---------------------------------------------------------------------------------
elif st.session_state.state > 0 and st.session_state.state <= 4:
    match st.session_state.state:
        case 1:
            st.markdown("## Результат прогнозирования:")
            st.markdown("""<p style="font-size: 80px; color: red;">ЗДОРОВ</p>""", unsafe_allow_html=True)
            st.button("Вернуться", on_click=back, type="primary")
        case 2:
            st.markdown("## Результат прогнозирования:")
            st.markdown("""<p style="font-size: 80px; color: red;">ГЕПАТИТ С</p>""", unsafe_allow_html=True)
            st.button("Вернуться", on_click=back, type="primary")
        case 3:
            st.markdown("## Результат прогнозирования:")
            st.markdown("""<p style="font-size: 80px; color: red;">ФИБРОЗ</p>""", unsafe_allow_html=True)
            cirrcol1, cirrcol2, cirrcol3 = st.columns([2, 1, 6])
            with cirrcol1:
                st.button("Вернуться", on_click=back, type="primary")
            with cirrcol3:
                st.button("Перейти к прогнозированию Цирроза", on_click=let_cirr)
        case 4:
            st.markdown("## Результат прогнозирования:")
            st.markdown("""<p style="font-size: 80px; color: red;">ЦИРРОЗ</p>""", unsafe_allow_html=True)
            cirrcol4, cirrcol5, cirrcol6 = st.columns([2, 1, 6])
            with cirrcol4:
                st.button("Вернуться", on_click=back, type="primary")
            with cirrcol6:
                st.button("Перейти к прогнозированию Цирроза", on_click=let_cirr)

#---------------------------------------------------------------------------------
elif st.session_state.state == 5:
    st.markdown("# Прогнозирование стадии поражения печени")
    st.markdown("## Признаки для прогнозирования:")

    age = st.slider("Возраст", min_value=15.0, max_value=100.0,
                                    step=1.0)
    
    col8, col9, col10 = st.columns([1, 1, 1])
    with col8:
        asc = st.selectbox("Наличие асцита", ['Есть', 'Нет'])
        hepat = st.selectbox("Увеличение печени", ['Есть', 'Нет'])
        spi = st.selectbox("Наличие паутинных сосудов", ['Есть', 'Нет'])
    with col9:
        edema = st.selectbox("Наличие отёков", ['Отёки присутствуют, несмотря на терапию диуретиками', 'Отёки присутствуют без диуретиков или отеки разрешились с помощью диуретиков', 'Отсутствуют и нет терапии диуретиками'])
        bil	= st.number_input("BIL", min_value=0.0, max_value=342.0,
                            step=0.1, help="Уровень билирубина в крови (мкмоль/л)")
        alb = st.number_input("ALB", min_value=1.0, max_value=100.0,
                            step=0.1, help="Уровень альбумина в крови (г/л)")
    with col10:
        cop	= st.number_input("COPPER", min_value=1.0, max_value=500.0,
                            step=1.0, help="Уровень меди в крови (мкг/дл)")
        plat	= st.number_input("PLATELETS", min_value=1.0, max_value=600.0,
                            step=1.0, help="Уровень тромбоцитов в крови (тыс./мкл)")
        prothr	= st.number_input("PROTHROMBIN", min_value=1.0, max_value=30.0,
                            step=0.1, help="Протромбиновое время (сек)")
    
    st.markdown("""
    <style>
    .custom-text {
        font-size: 20px;
        color: red;
    }
    </style>
    <p class="custom-text">Перед нажатием на кнопку, убедитесь в том, что вы заполнили все поля!</p>
    """, unsafe_allow_html=True)

    col11, col12, col13 = st.columns([2, 4, 2])
    with col11:
        st.button("Вернуться", on_click=back, type="primary")

    mapping = {'Есть': 1, 'Нет': 0}
    edema_mapping = {'Отёки присутствуют, несмотря на терапию диуретиками': 2, 'Отёки присутствуют без диуретиков или отеки разрешились с помощью диуретиков': 1, 'Отсутствуют и нет терапии диуретиками': 0}
    asc_numeric = mapping[asc]
    hepat_numeric = mapping[hepat]
    spi_numeric = mapping[spi]
    edema_numeric = edema_mapping[edema]
    input_data = np.array([[age, asc_numeric, hepat_numeric, spi_numeric, edema_numeric, bil, alb, cop, plat, prothr]])
    with col13:
        st.button("Прогнозирование", on_click=prediction_cirr, args=(input_data,))
    
else:
    match st.session_state.state:
        case 6:
            st.markdown("## Результат прогнозирования (METAVIR):")
            st.markdown("# F1 - F2")
            st.markdown("""<p style="font-size: 40px; color: red;">НАЧАЛЬНЫЙ ИЛИ УМЕРЕННЫЙ ФИБРОЗ</p>""", unsafe_allow_html=True)
            st.button("Вернуться", on_click=let_cirr, type="primary")
        case 7:
            st.markdown("## Результат прогнозирования (METAVIR):")
            st.markdown("# F3")
            st.markdown("""<p style="font-size: 40px; color: red;">ВЫРАЖЕННЫЙ ФИБРОЗ</p>""", unsafe_allow_html=True)
            st.button("Вернуться", on_click=let_cirr, type="primary")
        case 8:
            st.markdown("## Результат прогнозирования (METAVIR):")
            st.markdown("# F4")
            st.markdown("""<p style="font-size: 40px; color: red;">ЦИРРОЗ</p>""", unsafe_allow_html=True)
            st.button("Вернуться", on_click=let_cirr, type="primary")