import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# data = np.random.rand(4, 6)
# print(data)
# heat_map = sb.heatmap(data)
# plt.show()
sns.set_style(style="darkgrid")

exp = 1


ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/sac_a1_r1.csv")
