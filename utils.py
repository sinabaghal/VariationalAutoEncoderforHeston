import matplotlib.pyplot as plt

def heatmap(non_nan_mask):
    plt.figure(figsize=(10, 30))
    plt.imshow(non_nan_mask, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Non-NaN Mask (True=Non-NaN, False=NaN)')
    plt.title('Heatmap of Non-NaN Mask')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()