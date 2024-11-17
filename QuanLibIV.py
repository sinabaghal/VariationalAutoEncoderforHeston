import torch
import numpy as np 
import QuantLib as ql


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
day_count,calendar = ql.Actual365Fixed(),ql.TARGET()
def calculate_sigma(tau, spot, strike, option_price):

    term1 = np.sqrt(2 * np.pi) / (spot + strike)
    term2_part1 = option_price - (spot - strike) / 2
    term2_part2 = np.sqrt(term2_part1**2 - (spot - strike)**2 / np.pi)
    term2 = term2_part1 + term2_part2
    sigma = (1 / np.sqrt(tau)) * (term1 + term2)
    
    return sigma

def handle_error_with_default(default_value):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default_value
        return wrapper
    return decorator


@handle_error_with_default(np.nan)
def quantlib_iv(tau, strike, spot,r0,q0,calculation_date,expiry_date,option_price):

    
    exercise = ql.EuropeanExercise(expiry_date)
    ql.Settings.instance().evaluationDate = calculation_date

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    option = ql.VanillaOption(payoff,exercise)
    S = ql.QuoteHandle(ql.SimpleQuote(spot))
    r = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r0, day_count))
    q = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, q0, day_count))
    sigma = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, calendar, 1, day_count))
    process = ql.BlackScholesMertonProcess(S,q,r,sigma)
    iv = option.impliedVolatility(option_price,  process, 1e-10, 10000, 0.05, 4)


    return iv 