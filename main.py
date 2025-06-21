from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jax import jit, grad
import jax.numpy as jnp
from jax.scipy.stats import norm
import json
from typing import List

app = FastAPI()

# Allow frontend to talk to backend (localhost CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def black_scholes(s, k, t, r, sigma, option_type="call"):
    d1 = (jnp.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)

    if option_type == "call":
        return s * norm.cdf(d1) - k * jnp.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        return k * jnp.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type")

black_scholes = jit(black_scholes, static_argnames=["option_type"])


def greeks(s, k, t, r, sigma, option_type):
    bsm = lambda s, k, t, r, sigma: black_scholes(s, k, t, r, sigma, option_type)
    delta = grad(bsm, 0)(s, k, t, r, sigma)
    gamma = grad(grad(bsm, 0), 0)(s, k, t, r, sigma)
    vega = grad(bsm, 4)(s, k, t, r, sigma)
    rho = grad(bsm, 3)(s, k, t, r, sigma)
    theta = grad(bsm, 2)(s, k, t, r, sigma)
    return dict(delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)

class OptionInput(BaseModel):
    s: float
    k: float
    t: float
    r: float
    sigma: float
    option_type: str

@app.post("/calculate")
def calculate(option: OptionInput):
    price = float(black_scholes(option.s, option.k, option.t, option.r, option.sigma))
    g = greeks(option.s, option.k, option.t, option.r, option.sigma, option.option_type)
    return {"price": price, **{k: float(v) for k, v in g.items()}}


@app.post("/batch")
async def batch(request: Request):
    try:
        data = await request.json()
        results = []

        for option in data:
            s = float(option["s"])
            k = float(option["k"])
            t = float(option["t"])
            r = float(option["r"])
            sigma = float(option["sigma"])
            option_type = option["option_type"]

            price = float(black_scholes(s, k, t, r, sigma, option_type))
            g = greeks(s, k, t, r, sigma, option_type)

            results.append({
                "s": s,
                "k": k,
                "t": t,
                "r": r,
                "sigma": sigma,
                "option_type": option_type,
                "price": price,
                **g
            })

        return results
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)