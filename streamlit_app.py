import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#building the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Navigation Menu",["Home", "Symptom Checker", "Symptom-Based Disease Guide", "Condition-Based Medicine Guide"])

#Home Page

#displays all the available disease prediction options in the web app
if rad == "Home":
    # 컬럼 레이아웃 생성
    col1, col2 = st.columns([2, 8])

    # 첫 번째 컬럼에는 이미지를 표시
    with col1:
        st.image("SymptomSnap.png", width=100)

    # 두 번째 컬럼에는 타이틀 및 텍스트를 표시
    with col2:
        st.title("SymptomSnap")
        st.image("Medical Prediction Home Page.jpg")
        st.header("Find Answer to Your Symptoms")
        st.text("Input your symptoms and discover possible conditions and treatments.")
        st.text("The Following Guides Are Available ->")
        st.text("1. Symptom Checker")
        st.text("2. Symptom-Based Disease Guide")
        st.text("3. Condition-Based Medicine Guide")

if rad=="Symptom Checker":
    st.title('SymptomSnap')
    st.subheader("Predicting Diseases from Symptoms")
    df = pd.read_csv('Symptom_Checker.csv')
    # region 선택
    selected_region = st.selectbox('Select a region', df['region'].unique())
    
    # 선택된 region에 해당하는 sub-region 필터링
    sub_regions = df[df['region'] == selected_region]['sub-region'].unique()
    
    # sub-region 선택
    selected_sub_region = st.selectbox('Select a sub-region', sub_regions)
    
    # 선택된 sub-region에 해당하는 conditions 필터링
    conditions = df[(df['region'] == selected_region) & (df['sub-region'] == selected_sub_region)]['conditions']
    
    # condition 선택
    selected_condition = st.selectbox('Select a condition', conditions)
    
    # 선택된 condition에 해당하는 symptoms 표시
    description = df[(df['region'] == selected_region) & (df['sub-region'] == selected_sub_region) & (df['conditions'] == selected_condition)]['symptoms'].iloc[0]
    st.write('Description:', description)


# CSV 파일 로드
data = pd.read_csv("Disease_Symptom.csv")

# 데이터 전처리: 각 증상을 이진 특성으로 인코딩
symptoms = data.drop("Disease", axis=1).stack().str.get_dummies().groupby(level=0).max()

X = symptoms  # 이진 특성을 사용
y = data["Disease"]

# 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 분류기 모델 생성 및 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


if rad=="Symptom-Based Disease Guide":
    st.title('SymptomSnap')
    st.subheader("Predicting Diseases from Symptoms")
    options = st.multiselect(
        "Choose Your Symptoms",
        ['itching', ' skin_rash', ' nodal_skin_eruptions', ' dischromic _patches', ' continuous_sneezing', ' shivering', ' chills', ' watering_from_eyes', ' stomach_pain', ' acidity', ' ulcers_on_tongue', ' vomiting', ' cough', ' yellowish_skin', ' nausea', ' loss_of_appetite', ' burning_micturition', ' spotting_ urination', ' abdominal_pain', ' passage_of_gases', ' indigestion', ' muscle_wasting', ' patches_in_throat', ' high_fever', ' extra_marital_contacts', ' fatigue', ' weight_loss', ' restlessness', ' lethargy', ' irregular_sugar_level', ' sunken_eyes', ' dehydration', ' diarrhoea', ' breathlessness', ' family_history', ' headache', ' chest_pain', ' dizziness', ' loss_of_balance', ' lack_of_concentration', ' blurred_and_distorted_vision', ' excessive_hunger', ' back_pain', ' weakness_in_limbs', ' neck_pain', ' weakness_of_one_body_side', ' altered_sensorium', ' sweating', ' joint_pain', ' dark_urine', ' yellowing_of_eyes', ' swelling_of_stomach', ' distention_of_abdomen', ' constipation', ' pain_during_bowel_movements', ' pain_in_anal_region', ' bloody_stool', ' irritation_in_anus', ' cramps', ' bruising', ' obesity', ' swollen_legs', ' weight_gain', ' cold_hands_and_feets', ' mood_swings', ' anxiety', ' knee_pain', ' hip_joint_pain', ' swelling_joints', ' muscle_weakness', ' stiff_neck', ' movement_stiffness', ' painful_walking', ' spinning_movements', ' pus_filled_pimples', ' blackheads', ' scurring', ' bladder_discomfort', ' foul_smell_of urine', ' continuous_feel_of_urine', ' skin_peeling', ' silver_like_dusting', ' small_dents_in_nails', ' blister', ' red_sore_around_nose', ' yellow_crust_ooze']
        , [], key="symptom_multiselect")

    st.write("You selected:", options)

    #### 병
    # CSV 파일 읽기
    df = pd.read_csv('Disease_Symptom.csv')

    # options에 선택된 증상들을 new_symptoms에 할당
    new_symptoms = options
    
    # new_symptoms를 이진 특성으로 인코딩
    new_symptoms_encoded = pd.DataFrame(0, index=[0], columns=X.columns)
    for symptom in new_symptoms:
        if symptom.strip() in new_symptoms_encoded.columns:
            new_symptoms_encoded[symptom.strip()] = 1
    
    # 모델을 사용하여 새로운 증상에 대한 예측 확률 계산
    prediction_proba = model.predict_proba(new_symptoms_encoded)[0]
    predictions_df = pd.DataFrame({"Disease": model.classes_, "Probability": prediction_proba})

    # 가장 가능성이 높은 병 선택
    most_likely_disease = predictions_df.loc[predictions_df['Probability'].idxmax()]

    # 예측 확률을 백분율로 변환
    most_likely_disease["Probability"] *= 100

    # 가장 가능성이 높은 병을 따로 출력
    st.write("Most Likely Disease:")
    st.write(most_likely_disease)
    
    # 예측 확률을 기준으로 내림차순 정렬하여 상위 5개 병 선택
    top_5_diseases = predictions_df.sort_values(by="Probability", ascending=False).head(5)
    
    # 예측 확률을 백분율로 변환
    top_5_diseases["Probability"] = top_5_diseases["Probability"] * 100
    
    # 표로 나타내기
    st.write("Top 5 Most Likely Diseases:")
    st.write(top_5_diseases)

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_diseases["Disease"], top_5_diseases["Probability"], color='skyblue')
    
    # 그래프 제목과 축 라벨 설정
    plt.title('Top 5 Most Likely Diseases')
    plt.xlabel('Disease')
    plt.ylabel('Probability (%)')
    
    # 그래프를 Streamlit에 표시
    st.pyplot(plt)




    
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

# Train model for predicting medicine
# CSV 파일 로드
data = pd.read_csv("Drug_Condition.csv")

# condition이 공백인 행 삭제
data.dropna(subset=['condition'], inplace=True)

# 데이터 전처리: 각 컨디션을 이진 특성으로 인코딩
condition = data.drop(["drugName", "uniqueID"], axis=1).stack().str.get_dummies().groupby(level=0).max()

X2 = condition  # 이진 특성을 사용
y2 = data["drugName"]

# 훈련 세트와 테스트 세트로 분할
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# 랜덤 포레스트 분류기 모델 생성 및 훈련
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train2, y_train2)


if rad=="Condition-Based Medicine Guide":
    st.title('SymptomSnap')
    st.subheader("Predicting Medicines from Condition")

    # CSV 파일 로드
    df = pd.read_csv("Drug_Condition.csv")
    
    # condition 열의 값들을 리스트로 추출
    conditions = df['condition'].tolist()
    
    # 중복 제거
    conditions = list(set(conditions))
    
    # 빈 값을 제거
    conditions = [condition for condition in conditions if pd.notna(condition)]

    
    options = st.multiselect(
        "Choose Your Conditions",
        conditions
        , [], key="condition_multiselect")

    st.write("You selected:", options)

    # options에 선택된 증상들을 new_symptoms에 할당
    new_conditions = options
    
    # new_conditions를 이진 특성으로 인코딩
    new_conditions_encoded = pd.DataFrame(0, index=[0], columns=X2.columns)
    for condition in new_conditions:
        if condition.strip() in new_conditions_encoded.columns:
            new_conditions_encoded[condition.strip()] = 1
    
    # 모델을 사용하여 새로운 증상에 대한 예측 확률 계산
    prediction_proba = model2.predict_proba(new_conditions_encoded)[0]
    predictions_df = pd.DataFrame({"Drug": model2.classes_, "Probability": prediction_proba})

    # 가장 가능성이 높은 약 선택
    most_likely_drug = predictions_df.loc[predictions_df['Probability'].idxmax()]

    # 예측 확률을 백분율로 변환
    most_likely_drug["Probability"] *= 100

    # 가장 가능성이 높은 병을 따로 출력
    st.write("Most Suitable medicine:")
    st.write(most_likely_drug)
    
    # 예측 확률을 기준으로 내림차순 정렬하여 상위 5개 병 선택
    top_5_drug = predictions_df.sort_values(by="Probability", ascending=False).head(5)
    
    # 예측 확률을 백분율로 변환
    top_5_drug["Probability"] = top_5_drug["Probability"] * 100
    
    # 표로 나타내기
    st.write("Top 5 Most Suitable Medicines:")
    st.write(top_5_drug)

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_drug["Drug"], top_5_drug["Probability"], color='skyblue')
    
    # 그래프 제목과 축 라벨 설정
    plt.title('Top 5 Most Suitable Medicines')
    plt.xlabel('Medicine')
    plt.ylabel('Probability (%)')
    
    # 그래프를 Streamlit에 표시
    st.pyplot(plt)


    
    #### Side Effect of Medicine
    st.subheader("Side Effect of Medicine")
    df = pd.read_csv("Drug_Sideeffect.csv")
    sideeffect_description = dict(zip(df['drug_name'], df['side_effects']))

    # Side Effects
    selected_medicines = st.multiselect("Select Medicine", top_5_drug, key="sideeffects_multiselect")
    for medicine in selected_medicines:
        if medicine in sideeffect_description:
            st.write(f"Side Effects of Medicine ({medicine}): {sideeffect_description[medicine]}")
        else:
            st.write(f"No side effects of medicine available for {medicine}.")
    
    # URL 링크 생성
    url = 'https://www.drugs.com'
    link_text = 'Get detailed information about the drug'
    link = f'<a href="{url}" target="_blank">{link_text}</a>'
    
    # 생성된 링크를 출력
    st.markdown(link, unsafe_allow_html=True)
