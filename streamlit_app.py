import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#building the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Navigation Menu",["Home", "Symptom-Based Disease Guide", "Symptom-Based Medicine Guide"])

#Home Page

#displays all the available disease prediction options in the web app
if rad=="Home":
    st.title("SymptomSnap")
    st.image("Medical Prediction Home Page.jpg")
    st.header("Find Answer to Your Symptoms")
    st.text("Input your symptoms and discover possible conditions and treatments.")

df = pd.read_csv('Disease_Symptom.csv')    
# 특성과 타겟 데이터 준비
X = df.drop('Disease', axis=1)
X = np.array(X)
y = pd.DataFrame(df['Disease'])
    
# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# RandomForestClassifier 모델 생성
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
# 모델 훈련
rf_classifier.fit(X_train, y_train)
    

if rad == "Symptom-Based Disease Guide":
    # 위의 코드와 같은 내용
    
    #### 병 예측
    # CSV 파일 읽기
    df = pd.read_csv('Disease_Symptom.csv')
    
    # 사용자가 선택한 증상들 입력 받기
    options = st.multiselect(
        "Choose Your Symptoms",
        ['itching', ' skin_rash', ' nodal_skin_eruptions', ' dischromic _patches', ' continuous_sneezing', ' shivering', ' chills', ' watering_from_eyes', ' stomach_pain', ' acidity', ' ulcers_on_tongue', ' vomiting', ' cough', ' yellowish_skin', ' nausea', ' loss_of_appetite', ' burning_micturition', ' spotting_ urination', ' abdominal_pain', ' passage_of_gases', ' indigestion', ' muscle_wasting', ' patches_in_throat', ' high_fever', ' extra_marital_contacts', ' fatigue', ' weight_loss', ' restlessness', ' lethargy', ' irregular_sugar_level', ' sunken_eyes', ' dehydration', ' diarrhoea', ' breathlessness', ' family_history', ' headache', ' chest_pain', ' dizziness', ' loss_of_balance', ' lack_of_concentration', ' blurred_and_distorted_vision', ' excessive_hunger', ' back_pain', ' weakness_in_limbs', ' neck_pain', ' weakness_of_one_body_side', ' altered_sensorium', ' sweating', ' joint_pain', ' dark_urine', ' yellowing_of_eyes', ' swelling_of_stomach', ' distention_of_abdomen', ' constipation', ' pain_during_bowel_movements', ' pain_in_anal_region', ' bloody_stool', ' irritation_in_anus', ' cramps', ' bruising', ' obesity', ' swollen_legs', ' weight_gain', ' cold_hands_and_feets', ' mood_swings', ' anxiety', ' knee_pain', ' hip_joint_pain', ' swelling_joints', ' muscle_weakness', ' stiff_neck', ' movement_stiffness', ' painful_walking', ' spinning_movements', ' pus_filled_pimples', ' blackheads', ' scurring', ' bladder_discomfort', ' foul_smell_of urine', ' continuous_feel_of_urine', ' skin_peeling', ' silver_like_dusting', ' small_dents_in_nails', ' blister', ' red_sore_around_nose', ' yellow_crust_ooze']
        , [], key="symptom_multiselect")
    
    st.write("You selected:", options)
    
    # 사용자가 선택한 증상에 대한 예측 확률
    selected_symptoms = pd.DataFrame(columns=X.columns)  # 빈 데이터프레임 생성
    selected_symptoms.loc[0] = 0  # 첫 번째 행에 0으로 초기화
    for symptom in options:
        selected_symptoms[symptom] = 1  # 사용자가 선택한 증상들에 해당하는 열을 1로 설정


    
    predicted_probabilities = rf_classifier.predict_proba(selected_symptoms)
    
    # 상위 5개 질병 예측
    top_5_diseases = predicted_probabilities.argsort()[0][-5:][::-1]  # 상위 5개 질병의 인덱스
    top_5_probabilities = predicted_probabilities[0][top_5_diseases]  # 상위 5개 질병의 예측 확률
    
    # 예측된 상위 5개 질병과 확률 출력
    st.subheader("Top 5 Predicted Diseases and Probabilities")
    for disease, probability in zip(top_5_diseases, top_5_probabilities):
        st.write(f"Disease: {rf_classifier.classes_[disease]}, Probability: {probability:.4f}")


    
    #### 병 설명
    st.subheader("Explaining Your Diagnosis")
    df = pd.read_csv('Disease_Description.csv')
    disease_description = dict(zip(df['Disease'], df['Description']))

    # Diagnosis Details
    selected_diseases = st.multiselect("Select Diseases", df['Disease'].tolist(), key="diagnosis_multiselect")
    for disease in selected_diseases:
        if disease in disease_description:
            st.write(f"Diagnosis Details ({disease}): {disease_description[disease]}")
        else:
            st.write(f"No diagnosis details available for {disease}.")

    #### 예방 조치
    # 선택된 질병에 대한 예방 조치 출력
    st.subheader("Precautions for Your Condition")
    # CSV 파일 읽기
    df_precautions = pd.read_csv('Disease_Precautions.csv')

    # 줄 바꿈 문자 제거
    df_precautions.columns = df_precautions.columns.str.replace('\n', '')

    # 질병에 따른 예방 조치 데이터
    disease_precautions = {}
    for index, row in df_precautions.iterrows():
        disease = row['Disease']
        precautions = [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]
        disease_precautions[disease] = precautions

    # 선택된 질병에 대해 예방 조치 출력
    selected_diseases = st.multiselect("Select Diseases", df_precautions['Disease'].tolist(), key="precaution_multiselect")
    for disease in selected_diseases:
        if disease in disease_precautions:
            st.write(f"Precautions for {disease}:")
            for precaution in disease_precautions[disease]:
                st.write(f"- {precaution}")
        else:
            st.write(f"No precautions available for {disease}.")

    #### 식단
    st.subheader("Recommendation of Foods for Your Condition")
    # CSV 파일 읽기
    df = pd.read_csv('Disease_Diet.csv')

    # 질병에 따른 식단 데이터
    disease_diets = dict(zip(df['Disease'], df['Diet']))

    # 선택된 질병에 따라 식단 표시
    selected_diseases = st.multiselect("Select Diseases", df['Disease'].tolist(),key="diet_multiselect")
    for disease in selected_diseases:
        if disease in disease_diets:
            st.write(f"Dietary Recommendations ({disease}): {disease_diets[disease]}")
        else:
            st.write(f"No dietary recommendations available for {disease}.")


if rad=="Symptom-Based Medicine Guide":
    st.title('SymptomSnap')
    st.subheader("Predicting Medicines from Symptoms")
