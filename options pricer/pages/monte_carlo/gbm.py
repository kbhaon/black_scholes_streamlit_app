from jax import jit, random
import jax.numpy as jnp

key = random.PRNGKey(99)

@jit
def generate_paths_asia(key, n_simulations, n_steps, S, K, T, r, sigma):
    dt = T / n_steps
    Z = random.normal(key, (n_simulations, n_steps))
    
    log_returns = (r - 0.5 * sigma**2) * dt + sigma *jnp.sqrt(dt) * Z
    cumulative_log_returns = jnp.cumsum(log_returns, axis=1)
    log_prices = jnp.log(S) + cumulative_log_returns
    
    S_Paths = jnp.exp(log_prices)
    S_column = jnp.full((n_simulations, 1), S)
    S_Full = jnp.concatenate([S_column, S_Paths], axis=1)


    return S_Full

@jit
def option_price_put_asia(key, n_simulations, n_steps, S, K, T, r, sigma):

    S_Full = generate_paths_asia()
    
    avg_prices = jnp.mean(S_Full, axis=1)

    payoffs = jnp.maximum(K - avg_prices, 0)

    expected_payoffs = jnp.mean(payoffs)

    option_price_put = jnp.exp(-r * T) * expected_payoffs

    return option_price_put

@jit
def option_price_call_asia(key, n_simulations, n_steps, S, K, T, r, sigma):

    S_Full = generate_paths_asia()
    
    avg_prices = jnp.mean(S_Full, axis=1)

    payoffs = jnp.maximum(avg_prices - K, 0)

    expected_payoffs = jnp.mean(payoffs)

    option_price_call = jnp.exp(-r * T) * expected_payoffs


    return option_price_call

"""@jit
def option_price_call_asia():

    return


@jit
def geometric_brownian_motion_europe():
    return """
