from jax import jit
import jax.numpy as jnp
from jax.scipy.stats import norm
import streamlit as st



@jit
def black_scholes(s, k, t, r, sigma, option_type="call"):
    
    # S = Stock Price
    # K = Strike Price
    # T = time til maturity (in years)
    # r = risk free interest rate
    # sigma = volatility
    # option_type = call or put

    d1 = (jnp.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)


    if option_type == "call":
        price = s * norm.cdf(d1) - k * jnp.exp(-r * t) * norm.cdf(d2)

    elif option_type == "put":
        price = k * jnp.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    else:
        raise ValueError("Option type must be Call or Put")
    
    return price



st.title("Black Scholes Option Pricing")

st.markdown("Enter the option parameters below:")

s = st.number_input("Stock Price (S)", value=100.0)
k = st.number_input("Strike Price (K)", value=100.0)
t = st.number_input("Time to Maturity (T in years)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
option_type = st.radio("Option Type", ["call", "put"])

if st.button("Calculate Price"):
    price = black_scholes(s, k, t, r, sigma, option_type)
    st.success(f"{option_type.capitalize()} Option Price: {price:.4f}")
