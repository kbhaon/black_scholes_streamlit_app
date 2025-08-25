import jax.numpy as jnp
from scipy.stats import norm
from jax import jit, grad


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

