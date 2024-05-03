import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.title('SymptomSnap')

import streamlit as st

options = st.multiselect(
    "Choose Your Symptoms",
    ['itching', ' skin_rash', ' nodal_skin_eruptions', ' dischromic _patches', ' continuous_sneezing', ' shivering', ' chills', ' watering_from_eyes', ' stomach_pain', ' acidity', ' ulcers_on_tongue', ' vomiting', ' cough', ' yellowish_skin', ' nausea', ' loss_of_appetite', ' burning_micturition', ' spotting_ urination', ' abdominal_pain', ' passage_of_gases', ' indigestion', ' muscle_wasting', ' patches_in_throat', ' high_fever', ' extra_marital_contacts', ' fatigue', ' weight_loss', ' restlessness', ' lethargy', ' irregular_sugar_level', ' sunken_eyes', ' dehydration', ' diarrhoea', ' breathlessness', ' family_history', ' headache', ' chest_pain', ' dizziness', ' loss_of_balance', ' lack_of_concentration', ' blurred_and_distorted_vision', ' excessive_hunger', ' back_pain', ' weakness_in_limbs', ' neck_pain', ' weakness_of_one_body_side', ' altered_sensorium', ' sweating', ' joint_pain', ' dark_urine', ' yellowing_of_eyes', ' swelling_of_stomach', ' distention_of_abdomen', ' constipation', ' pain_during_bowel_movements', ' pain_in_anal_region', ' bloody_stool', ' irritation_in_anus', ' cramps', ' bruising', ' obesity', ' swollen_legs', ' weight_gain', ' cold_hands_and_feets', ' mood_swings', ' anxiety', ' knee_pain', ' hip_joint_pain', ' swelling_joints', ' muscle_weakness', ' stiff_neck', ' movement_stiffness', ' painful_walking', ' spinning_movements', ' pus_filled_pimples', ' blackheads', ' scurring', ' bladder_discomfort', ' foul_smell_of urine', ' continuous_feel_of_urine', ' skin_peeling', ' silver_like_dusting', ' small_dents_in_nails', ' blister', ' red_sore_around_nose', ' yellow_crust_ooze']
    , [])

st.write("You selected:", options)

# CSV 파일 읽기
df = pd.read_csv('Disease_Symptom.csv')

# 사용자가 선택한 증상 필터링
filtered_df = df[df['Symptom_1'].isin(options) |
                 df['Symptom_2'].isin(options) |
                 df['Symptom_3'].isin(options) |
                 df['Symptom_4'].isin(options)]

# 병별로 증상의 유사도 계산
disease_counts = filtered_df['Disease'].value_counts().reset_index()
disease_counts.columns = ['Disease', 'Count']

# 막대 그래프 생성
bar_chart = alt.Chart(disease_counts).mark_bar().encode(
    x='Count',
    y=alt.Y('Disease', sort='-x'),
    color='Disease'
).properties(
    title='가능성이 높은 병 TOP 5'
)

# 시각화 결과 출력
st.write(bar_chart)
