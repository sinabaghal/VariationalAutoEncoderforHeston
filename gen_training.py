
import numpy as np
import torch
import os 
import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from QuanLibIV import quantlib_iv

# v0 = 0.05, kappa = 0.5, eta = 0.25**2, rho = -0.75, sigma = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### Variables: Time
T = 1
timesteps = 365
today = ql.Date(1, 7, 2020)
Times = ql.TimeGrid(T, timesteps)
dt = torch.tensor(T / (timesteps), dtype=torch.float64,device=device)
taus_days = [0,6,6*2,6*3]+[26*x for x in range(1,15)]
Times_taus_days = [Times[x] for x in taus_days][1:]

#### Variable: Heston
 

spot = 1.0 # Initial stock   price
r = torch.tensor(0.00)    # Risk-free interest rate
q = 0.0 
v0    = 0.05

#### Variables: MC

num_path_log = 23
round_log = 0
num_paths = 2**num_path_log


kmin = 0.4 
kmax = 0.6
x_min = -( kmin)**(1/3)
x_max = ( kmax)**(1/3)
k_aux = [x**3 for x in np.linspace(x_min, x_max, 100)]
if 0 not in k_aux:
    k_aux.append(0); k_aux.sort()

k_aux_min = [k for k in k_aux if np.abs(k)<0.1]
# k_aux_min = k_aux
K_aux_min = np.exp(k_aux_min)
K_aux = np.exp(k_aux)
# import pdb; pdb.set_trace()

def pricing(kappa,eta,rho,sigma):


    kappa_dt = torch.tensor(kappa * dt)
    sdt = torch.sqrt(torch.tensor(dt, dtype=torch.float64,device=device))

    df_gpus = []

    for _ in range(2**round_log):

        df_gpu = pd.DataFrame(index=taus_days[1:], columns=K_aux)
        stocks = {t: [] for t in taus_days[1:]}
        S = torch.full((num_paths,), spot, dtype=torch.float64, device=device)  # Initial asset price
        v = torch.full((num_paths,), v0, dtype=torch.float64, device=device)  # Initial variance 
        for t in range(1, timesteps + 1):

            
            Zs = torch.randn(num_paths,  dtype=torch.float64, device=device)
            Z1 = torch.randn(num_paths,  dtype=torch.float64, device=device)

            vol = torch.sqrt(torch.relu(v))
            volsigma = vol*sigma

            v = v + kappa_dt * (eta - vol) + volsigma*sdt * (rho * Zs - torch.sqrt(1 - rho**2) * Z1)
            S = S * torch.exp((- 0.5 * vol**2) * dt + vol*sdt * Zs)
            
            if t in taus_days[1:]:
                stocks[t].append(S)  
            
        
        for t in taus_days[1:]: 

            stocks[t] = torch.cat(stocks[t],dim=0)

        stocks_tensor = torch.stack([stocks[t] for t in taus_days[1:]], dim=0)
        del stocks 
        for strike in K_aux: df_gpu.loc[:, strike] = torch.relu(stocks_tensor-strike).mean(axis=1).cpu()

        df_gpus.append(df_gpu)



    df_av = sum(df_gpus)/len(df_gpus)


    ### Computing the implied volatility 

    ivs = pd.DataFrame(index=taus_days[1:], columns=K_aux)
    for tau_idx, tau_day in enumerate(taus_days[1:]):

        tau = Times[tau_day]
        expiry_date = today+ql.Period(tau_day,ql.Days)
        
        for strike_idx, strike in enumerate(K_aux):
        
            option_price = df_av.iloc[tau_idx, strike_idx]
            iv = quantlib_iv(tau,strike, spot, float(r),float(q),today,expiry_date,option_price)        
            ivs.loc[tau_day, strike] = iv

    ivs[ivs==0.05] = np.nan

    return ivs 

if __name__ == "__main__":

    num_steps = 3
    kappa_range = torch.linspace(0.1, 0.9, steps=num_steps)   # e.g., 5 steps from 0.1 to 0.5
    eta_range = torch.linspace(0.05**2, 0.25**2, steps=num_steps)
    rho_range = torch.linspace(-0.9, -0.1, steps=num_steps)
    sigma_range = torch.linspace(0.1, 0.5, steps=num_steps)

    for i_kappa, kappa in enumerate(kappa_range):
        for i_eta, eta in enumerate(eta_range):
            for i_rho, rho in enumerate(rho_range):
                for i_sigma, sigma in enumerate(sigma_range):
                    filename = f'trainingdata\\ivs_{i_kappa}_{i_eta}_{i_rho}_{i_sigma}.npy'
                    if os.path.exists(filename): continue 
                    result = pricing(kappa,eta,rho,sigma)
                    
                    np.save(filename, result)


    data = np.load('trainingdata\\ivs_0_0_0_0.npy',allow_pickle=True).astype(np.float64)
    df_ivs = pd.DataFrame(data, index=taus_days[1:], columns=K_aux)
    no_nan_mask = np.ones_like(data, dtype=bool)

    df_vars =[]

    for filename in os.listdir('trainingdata'):
        if filename.endswith('.npy'):

            filepath = os.path.join('trainingdata', filename)
            data = np.load(filepath,allow_pickle=True).astype(np.float64)  # Load the .npy file
            df_ivs = pd.DataFrame(data, index=taus_days[1:], columns=K_aux)
            df_var = (df_ivs*df_ivs).mul(df_ivs.index, axis=0)
            no_nan_mask &= ~np.isnan(data)
            df_vars.append(df_var)

    no_nan_mask_df = pd.DataFrame(no_nan_mask, index=taus_days[1:], columns=K_aux).astype(int)
    df_vars = [((df_var*no_nan_mask_df).replace(0, np.nan)).values for df_var in df_vars]
    df_vars = [array[~np.isnan(array)] for array in df_vars]
    training_data = np.stack(df_vars)

    def ivcurve_plot(dfs, tau_list):

        plt.figure()
        for i, df in enumerate(dfs):

            for index, row in df.iterrows():
                if tau_list is None:
                    pass

                else:
                    
                    if index not in tau_list: 
                        continue 
                
                these_K = K_aux[~row.isna()]
                these_iv = row[~row.isna()]
                plt.plot(these_K, these_iv, marker='o', markersize=2, label=f'tau = {index} - df {i}')

        
        plt.title(f'Sliced Vol Surface')
        plt.xlabel('K_aux')
        plt.ylabel('iv')
        
        plt.legend()
        plt.grid(True)
        plt.show()
