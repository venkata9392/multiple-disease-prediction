import streamlit as st
import pickle
import numpy as np

# -------------------- Load Models --------------------

diabetes_model = pickle.load(open('D:\\multiple\\diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('D:\\multiple\\heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('D:\\multiple\\parkinsons_model.sav', 'rb'))

# -------------------- Sidebar --------------------

st.sidebar.title("Multiple Disease Prediction System")

option = st.sidebar.selectbox(
    "Select Disease",
    ("Diabetes Prediction", 
     "Heart Disease Prediction", 
     "Parkinson's Prediction")
)

# =====================================================
# ================== DIABETES =========================
# =====================================================

if option == "Diabetes Prediction":

    st.title("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.text_input("Pregnancies")
        glucose = st.text_input("Glucose Level")
        blood_pressure = st.text_input("Blood Pressure")
        skin_thickness = st.text_input("Skin Thickness")

    with col2:
        insulin = st.text_input("Insulin Level")
        bmi = st.text_input("BMI")
        dpf = st.text_input("Diabetes Pedigree Function")
        age = st.text_input("Age")

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]], dtype=float)

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Person has Diabetes")
        else:
            st.success("Person does not have Diabetes")


# =====================================================
# ================== HEART DISEASE ====================
# =====================================================

elif option == "Heart Disease Prediction":

    st.title("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age")
        sex = st.text_input("Sex (1=Male, 0=Female)")
        cp = st.text_input("Chest Pain Type")
        trestbps = st.text_input("Resting Blood Pressure")

    with col2:
        chol = st.text_input("Cholesterol")
        fbs = st.text_input("Fasting Blood Sugar")
        restecg = st.text_input("Rest ECG")
        thalach = st.text_input("Max Heart Rate")

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps,
                                chol, fbs, restecg, thalach]], dtype=float)

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Person has Heart Disease")
        else:
            st.success("Person does not have Heart Disease")


# =====================================================
# ================== PARKINSONS =======================
# =====================================================

elif option == "Parkinson's Prediction":

    st.title("Parkinson's Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fo = st.text_input("MDVP:Fo(Hz)")
        fhi = st.text_input("MDVP:Fhi(Hz)")
        flo = st.text_input("MDVP:Flo(Hz)")
        jitter = st.text_input("Jitter(%)")

    with col2:
        shimmer = st.text_input("Shimmer")
        nhr = st.text_input("NHR")
        hnr = st.text_input("HNR")
        rpde = st.text_input("RPDE")

    if st.button("Predict Parkinson's"):
        input_data = np.array([[fo, fhi, flo, jitter,
                                shimmer, nhr, hnr, rpde]], dtype=float)

        prediction = parkinsons_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Person has Parkinson's Disease")
        else:
            st.success("Person does not have Parkinson's Disease")
