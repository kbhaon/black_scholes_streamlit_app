import streamlit as st
from black_scholes import black_scholes_formulas as bs
import json
import pandas as pd

tab1, tab2 = st.tabs["Single", "Bulk Import/Export"]

with tab1:
    st.header("Single Option Pricing + Greeks")

    s = st.number_input("Stock Price (S)", value=100.0)
    k = st.number_input("Strike Price (K)", value=100.0)
    t = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    option_type = st.radio("Option Type", ["call", "put"])

    if st.button("Calculate Price"):
        result = bs.bs_price_and_greeks(
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
                            res = bs.bs_price_and_greeks(
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
                st.download_button("📥 Download CSV", csv, "batch_options_output.csv", "text/csv")

        except Exception as e:
            st.error(f"Failed to process file: {e}")
    