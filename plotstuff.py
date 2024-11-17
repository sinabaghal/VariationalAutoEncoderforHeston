import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('trainingplot.csv')
df.columns = ['Epoch', 'Cur LR', 'Cur MSE', 'Cur KL', 'Saved KL', 'Saved MSE', 'KL Reg']

# Plot side-by-side graphs for MSE and KL
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot for MSE
axes[0].plot(df.index, np.log(df['Saved MSE']), label='Log MSE', linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Log(MSE)')
axes[0].set_title('VAE Loss Plot - Log(MSE)')
axes[0].legend()
axes[0].grid()

# Plot for KL
axes[1].plot(df.index, np.log(df['Saved KL']), label='VAE.KL', linestyle='-', color='orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Log(KL)')
axes[1].set_title('VAE Loss Plot - VAE.KL')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
