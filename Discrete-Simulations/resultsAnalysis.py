import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Read dataframe from csv and trim the data
results = pd.read_csv("results7V.csv")
results["product"] = results["efficiency"]/10000 * results["cleaned"]* results["cleaned"]
results["product2"] = results["efficiency"]/100 * results["cleaned"]
mask = (results['theta'] == 0.001) & (results['certainty'] < 1.0) & (results['certainty'] > 0.2)
results2= results[mask]

# Plot a heatmap on the intended parameters
gropued = results2.groupby(['certainty', 'gamma'])
avg_eff = gropued['efficiency'].mean().unstack()
avg_cln = gropued['cleaned'].mean().unstack()
avg_prod = gropued['product'].mean().unstack()
avg_prod2 = gropued['product2'].mean().unstack()
sns.heatmap(avg_eff)
plt.show()