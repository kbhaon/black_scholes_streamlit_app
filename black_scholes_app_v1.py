from jax import jit
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import grad
from jax import vmap
import streamlit as st
import pandas as pd
import json
import yfinance as yf
import datetime

# will use to directly import yfinance options data into pricer
def get_options(ticker, r=0.05):
    dat = yf.Ticker(ticker)
    S = dat.history(period="1d")["Close"][-1]

    options_data = []
    for exp in dat.options:
        expiration = datetime.strptime(exp, "%Y-%m-%d")
        T = (expiration - datetime.now()).days / 365.0

        #Calls
        calls = dat.option_chain(exp).calls.copy()
        calls["type"] = "call"
        calls["expiration"] = exp
        calls["T"] = T
        calls["S"] = S
        
        #Puts
        puts = dat.option_chain(exp).puts.copy()
        puts["type"] = "put"
        puts["expiration"] = exp
        puts["T"] = T
        puts["S"] = S

    return pd.concat(options_data, ignore_index=True)


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
        "delta (Δ)": float(delta(s, k, t, r, sigma)),
        "gamma (Γ)": float(gamma(s, k, t, r, sigma)),
        "vega (𝜈)": float(vega(s, k, t, r, sigma)),
        "rho (ρ)": float(rho(s, k, t, r, sigma)),
        "theta (Θ)": float(theta(s, k, t, r, sigma))
}


def bs_price_and_greeks(s, k, t, r, sigma, option_type):
    price = float(black_scholes(s, k, t, r, sigma, option_type))
    greeks = black_scholes_autodiff(s, k, t, r, sigma, option_type)

    greeks = {key: float(val) for key, val in greeks.items()}

    return {
        "option_type": option_type,
        "s": float(s),
        "k": float(k),
        "t": float(t),
        "r": float(r),
        "sigma": float(sigma),
        "price": price,
        **greeks
    }



st.title("Black-Scholes Options Pricing Tool")

tab1, tab2, tab3, tab4 = st.tabs(["Single Option Pricing", "Batch Pricing via JSON", "Real-Time Pricing via ticker", "Why make this?"])

with tab1:
    st.header("Single Option Pricing + Greeks")

    s = st.number_input("Stock Price (S)", value=100.0)
    k = st.number_input("Strike Price (K)", value=100.0)
    t = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    option_type = st.radio("Option Type", ["call", "put"])

    if st.button("Calculate Price"):
        result = bs_price_and_greeks(
            float(s), float(k), float(t), float(r), float(sigma), option_type
        )
        price = result["price"]
        greeks = {k: v for k, v in result.items() if k not in ["price", "option_type", "s", "k", "t", "r", "sigma"]}

        st.success(f"{option_type.capitalize()} Option Price: {price:.4f}")
        st.subheader("Greeks")
        for greek, value in greeks.items():
            st.write(f"**{greek}**: {value:.4f}")

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
                    results = []
                    for o in options_list:
                        try:
                            res = bs_price_and_greeks(
                                float(o["s"]),
                                float(o["k"]),
                                float(o["t"]),
                                float(o["r"]),
                                float(o["sigma"]),
                                type_label
                            )
                            results.append(res)
                        except Exception as e:
                            results.append({"error": str(e), **o})
                    return results
                
                results = []

                if calls:
                    results.extend(run_batch(calls, "call"))
                if puts:
                    results.extend(run_batch(puts, "put"))

                df = pd.DataFrame(results)
                result_cols = [
                    "option_type", "price",
                    "delta (Δ)", "gamma (Γ)", "vega (𝜈)", "rho (ρ)", "theta (Θ)"
                ]
                df = df[[col for col in result_cols if col in df.columns]]
                st.success(f"Processed {len(df)} options.")
                st.dataframe(df, use_container_width=True)



                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "batch_options_output.csv", "text/csv")

        except Exception as e:
            st.error(f"Failed to process file: {e}")
    

with tab3:
    st.header("Real-Time pricing via ticker")
    st.subheader("""*Data is sourced from yfinance*""")

    st.text_input("")

with tab4:

    st.header("Background: Why I am making this?")
    st.markdown("""
    This past year, I've developed a strong interest in derivative markets and quantitative finance.  
    I wanted to build a project using some kind of model that would improve both my coding skills and knowledge of the math/statistics behind pricing models at the same time.

    This led me to try and build a simple Black-Scholes options pricing calculator. The Black-Scholes model seems to be the most common and foundational models to learn about.  
    This project is not meant to be a reliable tool for trading. I am aware of all the limitaions of the Black-Scholes model.
    """)

    st.header("Libraries Used")
    st.markdown("""
    To Build this python app, I used:

    - **Streamlit** – For building the web interface 
    - **Pandas** – For handling and manipulating batch json file uploads  
    - **JAX / NumPy** – For fast numerical computations 

    I don't believe JAX is being fully utilized in this app with its little calulation needs. Regular numpy should've been fine. 
    However, I still chose to use it as I feel getting familiar with the syntax will be useful in the future. I have read that JAX is useful for quantitative models using python.
    """)

    st.markdown("---")
    st.header("📎 Connect With Me")

    st.markdown("""
    - [![GitHub](https://img.shields.io/badge/GitHub-kbhaon-black?logo=github)](https://github.com/kbhaon)
    - [![LinkedIn](https://img.shields.io/badge/LinkedIn-Noah%20Kauss-blue?logo=linkedin)](https://www.linkedin.com/in/noah-kauss-33719a333/)
    """)

