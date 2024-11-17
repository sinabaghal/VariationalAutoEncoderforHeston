import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import os 
import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import imageio
from torch.nn import functional as F
from gen_training import pricing


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
 
spot = 1.0 
r = torch.tensor(0.00)   
q = 0.0 
v0    = 0.05

#### Variables: Setting

kmin = 0.4 
kmax = 0.6
x_min = -( kmin)**(1/3)
x_max = ( kmax)**(1/3)
k_aux = [x**3 for x in np.linspace(x_min, x_max, 100)]
if 0 not in k_aux:
    k_aux.append(0); k_aux.sort()

k_aux_min = [k for k in k_aux if np.abs(k)<0.1]
K_aux_min = np.exp(k_aux_min)
K_aux = np.exp(k_aux)

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
training_data = torch.tensor(np.stack(df_vars)).to(device=device)
# df_training_data = pd.DataFrame(array_training_data).to_csv('training_data.csv')

class Encoder(nn.Module):

    def __init__(self, hidden_sizes_encoder, latent_dims):

        super(Encoder, self).__init__()
        layers = []
        in_size = training_data.shape[1]
        

        for hidden_size in hidden_sizes_encoder:
            
            layers.append(nn.Linear(in_size, hidden_size).to(torch.float64))
            layers.append(nn.LeakyReLU())
            in_size = hidden_size

        self.layer_mu = nn.Linear(in_size, latent_dims).to(torch.float64)
        self.layer_var = nn.Linear(in_size, latent_dims).to(torch.float64)
        self.network = nn.Sequential(*layers)

    def forward(self, x):

        x = self.network(x)
        mu =  self.layer_mu(x)
        log_var = self.layer_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps * std + mu #torch.Size([625, 4])
        self.kl = 0.5*(mu ** 2 + log_var.exp()-1 - log_var).mean()

        return z

class Decoder(nn.Module):

    def __init__(self, hidden_sizes_decoder, latent_dims):
        super(Decoder, self).__init__()

        layers = []
        in_size = latent_dims
        layers.append(nn.Linear(in_size, hidden_sizes_decoder[0]).to(torch.float64))

        for i in range(len(hidden_sizes_decoder)-1):
            
            dim_1, dim_2 = hidden_sizes_decoder[i],hidden_sizes_decoder[i+1]
            layers.append(nn.Linear(dim_1, dim_2).to(torch.float64))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_sizes_decoder[-1], training_data.shape[1]).to(torch.float64))
        layers.append(nn.LeakyReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)    
    
    
class VariationalAutoencoder(nn.Module):

    def __init__(self, hidden_sizes_encoder, hidden_sizes_decoder, latent_dims):

        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(hidden_sizes_encoder, latent_dims)
        self.decoder = Decoder(hidden_sizes_decoder, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z) 
    
print_epoch = 1000
print(f'Print every {print_epoch} epochs!')
def train(autoencoder, epochs=print_epoch*10000):

    best_loss = float('inf')
    kl_saved = float('inf')
    mse_saved = float('inf')
    header_message = f"{'Epoch':>5} | {'Cur LR':>20} | {'Cur MSE':>20} | {'Cur KL':>20} | {'Saved KL':>20} | {'Saved MSE':>20} | {'KL Reg':>20}"
        
    print(header_message)    
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.01,weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=3*print_epoch, min_lr=1e-6)

    epoch = 0
    beta = 1

    while best_loss > 1e-5 and epoch <epochs:

        opt.zero_grad() 
        iv_stacked_hat = autoencoder(training_data)
        loss_mse_ = F.mse_loss(iv_stacked_hat, training_data)
        loss_mse =  loss_mse_*training_data.shape[1]
        kl = autoencoder.encoder.kl
        loss = loss_mse+beta*kl
        loss.backward()
        opt.step()
        cur_lr = opt.state_dict()["param_groups"][0]["lr"]
        total_loss =  loss_mse + beta*kl
        
        scheduler.step(total_loss)
        if best_loss > loss.item():

            best_loss = loss.item()
            mse_saved = loss_mse_.item()
            kl_saved = kl.item()
            torch.save(autoencoder.state_dict(), 'autoencoder.pth')

        if epoch % print_epoch == 0: 
            print(f"{int(epoch/print_epoch):>5} | {cur_lr:>20.10f} | {loss_mse_.item():>20.10f} | {kl.item():>20.10f} | {kl_saved:>20.10f} | {mse_saved:>20.10f} | {beta:>20.10f}")
        
        epoch = epoch + 1

    return autoencoder

if __name__ == "__main__":

    hidden_sizes_encoder = [32, 64,128]
    hidden_sizes_decoder = [dim for dim in reversed(hidden_sizes_encoder)]
    latent_dims = 4
    vae = VariationalAutoencoder(hidden_sizes_encoder, hidden_sizes_decoder, latent_dims).to(device)
    # train(vae)
    
    
    vae.load_state_dict(torch.load('autoencoder.pth')) 

    def plotrand():


        n_steps = 400
        start = np.zeros(latent_dims)  

        walk = [start]
        dt = 0.2
        for _ in range(n_steps):
            direction = np.random.randn(4)
            walk.append(walk[-1] + dt*direction/np.linalg.norm(direction))

        walk = torch.tensor(walk,dtype=torch.float64).to(next(vae.parameters()).device)
        frames = []
        with torch.no_grad():
            for i in range(walk.shape[0]):

                z = walk[i]
                decoded_output = vae.decoder(z)
                flat_mask = no_nan_mask_df.values.flatten()
                one_positions = np.where(flat_mask == 1)[0]
                flat_result = np.zeros_like(flat_mask, dtype=float)
                flat_result[one_positions] = decoded_output.cpu().numpy()
                result_df = pd.DataFrame(flat_result.reshape(no_nan_mask_df.shape), 
                                    index=no_nan_mask_df.index, columns=no_nan_mask_df.columns)
                result_df = result_df.replace(0, np.nan)

                
                file_path = os.path.join('frames',f"frame_{i}.png")
                
                fig = plt.figure(figsize=(15, 10))
                for idx, row in result_df.iterrows():
                    plt.plot(row, marker = 'o', label=f"{idx} D")
                plt.xlabel("Strike = K/Spot")
                plt.ylabel("Total Variance")
                plt.title("VAE-generated Total Variance")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.savefig(file_path)
                frames.append(file_path)
                plt.close(fig)

        output_gif = "random_walk_4D.gif"
        with imageio.get_writer(output_gif, mode='I', duration=0.05) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)

        # Cleanup temporary files
        for frame in frames:
            os.remove(frame)
        
        return result_df

# plotrand()

kappa_min, kappa_max = 0.1, 0.9
eta_min, eta_max = 0.05**2, 0.25**2
rho_min, rho_max = -0.9, -0.1
sigma_min, sigma_max = 0.1, 0.5
while True:

    kappa_random = torch.empty(1).uniform_(kappa_min, kappa_max).to(device)
    eta_random = torch.empty(1).uniform_(eta_min, eta_max).to(device)
    rho_random = torch.empty(1).uniform_(rho_min, rho_max).to(device)
    sigma_random = torch.empty(1).uniform_(sigma_min, sigma_max).to(device)

    random_prices = pricing(kappa_random,eta_random,rho_random,sigma_random)
    random_prices = (random_prices*random_prices).mul(random_prices.index, axis=0)
    random_array = (random_prices*no_nan_mask_df).replace(0, np.nan).values
    random_tensor = torch.tensor(random_array[~np.isnan(random_array)]).to(device)
    # import pdb; pdb.set_trace()
    if random_tensor.shape[0] == training_data.shape[1]:
        break 
class GDModel(nn.Module):
    
    def __init__(self):
    
        super(GDModel, self).__init__()
        self.linear = nn.Linear(1,latent_dims).to(torch.float64)
        
    def forward(self):
        # import pdb; pdb.set_trace()
        out = self.linear(torch.tensor([1.0]).to(torch.float64).to(device))
        return out

vae.eval()

best_loss = float('inf')
epoch = 0
print('NEXT')

gdmodel = GDModel().to(device)
optimizer = optim.Adam(gdmodel.parameters(), lr=0.1)
scheduler_args = {'mode':'min', 'factor':0.9, 'patience':100, 'threshold':0.05}
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

losses = []
num_epochs  = 50000
with torch.no_grad():
    fit_vae = vae.decoder(gdmodel())
while epoch < num_epochs:

    optimizer.zero_grad()
    loss = F.mse_loss(vae.decoder(gdmodel()),random_tensor)
    loss.backward()
    optimizer.step()
    # scheduler.step(loss)
    if loss < best_loss: 
        best_loss = loss
        print(best_loss.item())
        with torch.no_grad():
            fit_vae = vae.decoder(gdmodel())
    cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
    losses.append(loss.item())
    epoch += 1

def plotz(zs):

    markers = ['o', '+']
    plt.figure(figsize=(15, 10))
    for id, z in enumerate(zs):

        flat_mask = no_nan_mask_df.values.flatten()
        one_positions = np.where(flat_mask == 1)[0]
        flat_result = np.zeros_like(flat_mask, dtype=float)
        flat_result[one_positions] = z.cpu().numpy()
        result_df = pd.DataFrame(flat_result.reshape(no_nan_mask_df.shape), 
                            index=no_nan_mask_df.index, columns=no_nan_mask_df.columns)
        result_df = result_df.replace(0, np.nan)
        


        for idx, row in result_df.iterrows():
            plt.plot(row, marker = markers[id], label=f"D {idx}")

        
        
    plt.xlabel("Total Variance")
    plt.ylabel("Strike = K/Spot")
    plt.title("VAE fits random Heston Total Variance Surace")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('fig.png')

    return result_df

with torch.no_grad():
    plotz([fit_vae,random_tensor])
    import pdb; pdb.set_trace()

