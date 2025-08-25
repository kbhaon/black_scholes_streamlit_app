import streamlit as st

tab1,tab2 = "Custom Option Input", "Real Option Picker"

with tab1:
    st.header("Custom Option Input")



    s = st.number_input("Stock Price (S)", value=100.0)
    k = st.number_input("Strike Price (K)", value=100.0)
    t = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    option_type = st.radio("Option Type", ["call", "put"])
    key = st.number_input("Key", value=100)
    n_simulations = st.number_input("Number of Simulations", value=1000)
    n_steps = st.number_input("Number of Steps", value=1)