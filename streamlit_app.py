import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.title('SymptomSnap')

import streamlit as st

options = st.multiselect(
    "What are your favorite colors",
    ["Green", "Yellow", "Red", "Blue"],
    ["Yellow", "Red"])

st.write("You selected:", options)
