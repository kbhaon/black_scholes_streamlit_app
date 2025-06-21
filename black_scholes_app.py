from jax import jit
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import grad
from jax import vmap
import streamlit as st
import pandas as pd
import json


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

black_scholes = jit(black_scholes, static_argnames=["option_type"])

def black_scholes_autodiff(s, k, t, r, sigma, option_type):

    #make function differientiable
    bsm_pricer = lambda s, k, t, r, sigma: black_scholes(s, k, t, r, sigma, option_type)

    #first and second order derivatives
    delta = grad(bsm_pricer, argnums=0)
    gamma = grad(delta, argnums=0)
    vega = grad(bsm_pricer, argnums=4)
    rho = grad(bsm_pricer, argnums=3)
    theta = grad(bsm_pricer, argnums=2)

    #Evaluate Greeks
    return {
    "delta": delta(s, k, t, r, sigma),
    "gamma": gamma(s, k, t, r, sigma),
    "vega": vega(s, k, t, r, sigma),
    "rho": rho(s, k, t, r, sigma),
    "theta": theta(s, k, t, r, sigma)
}


def bs_price_and_greeks(s, k, t, r, sigma, option_type):
    price = black_scholes(s, k, t, r, sigma, option_type)
    greeks = black_scholes_autodiff(s, k, t, r, sigma, option_type)
    return {
        "s": s,
        "k": k,
        "t": t,
        "r": r,
        "sigma": sigma,
        "price": price,
        **greeks
    }



st.title("Black-Scholes Options Pricing Tool")

tab1, tab2 = st.tabs(["Single Option Pricing", "Batch Pricing via JSON"])

with tab1:
    st.header("Single Option Pricing + Greeks")

    s = st.number_input("Stock Price (S)", value=100.0)
    k = st.number_input("Strike Price (K)", value=100.0)
    t = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    option_type = st.radio("Option Type", ["call", "put"])

    if st.button("Calculate Price"):
        price = black_scholes(s, k, t, r, sigma, option_type)
        greeks = black_scholes_autodiff(s, k, t, r, sigma, option_type)
        st.success(f"{option_type.capitalize()} Option Price: {price:.4f}")

with tab2:
    st.header("Upload JSON for Batch Pricing + Greeks")

    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

    if uploaded_file is not None:
        try:
            options = json.load(uploaded_file)

            if not isinstance(options, list):
                st.error("JSON must be a list of option objects.")
            else: 
                calls = [o for o in options if o["option_type"] == "call"]
                puts = [o for o in options if o["option_type"] == "put"]
                
                def run_batch(options_list, type_label):
                    s = jnp.array([float(o["s"]) for o in options_list], dtype=jnp.float32)
                    k = jnp.array([float(o["k"]) for o in options_list], dtype=jnp.float32)
                    t = jnp.array([float(o["t"]) for o in options_list], dtype=jnp.float32)
                    r = jnp.array([float(o["r"]) for o in options_list], dtype=jnp.float32)
                    sigma = jnp.array([float(o["sigma"]) for o in options_list], dtype=jnp.float32)

                    batched_func = vmap(bs_price_and_greeks, in_axes=(0, 0, 0, 0, 0, None))
                    results = batched_func(s, k, t, r, sigma, type_label)
                    return results
                
                results = []

                if calls:
                    results.extend(run_batch(calls, "call"))
                if puts:
                    results.extend(run_batch(puts, "put"))

                df = pd.DataFrame(results)
                st.success(f"Processed {len(df)} options.")
                st.dataframe(df)


                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv, "batch_options_output.csv", "text/csv")

        except Exception as e:
            st.error(f"Failed to process file: {e}")

