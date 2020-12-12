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

if exp == 1:
    ddpg_df = pd.read_csv("./csv/ddpg_bin_shoot_opp.csv")

    # penguins = sns.load_dataset("penguins")
    print(ddpg_df)
    g = sns.displot(ddpg_df, x="x", y="y", binwidth=(.5, .5), cbar=True)
    # g = sns.displot(ddpg_df, x="bx", y="by", binwidth=(.5, .5), cbar=True, color="r")
    g.set(xlabel='Ball X Position', ylabel='Ball Y Position')

    plt.ylim(-30, 30)
    plt.xlim(5, 55)
    plt.xlabel('Agent X Position',fontsize=20)
    plt.ylabel('Agent Y Position',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.show()

if exp == 2:
    ddpg_df = pd.read_csv("./csv/sac_a1_r1.csv")

    g = sns.displot(ddpg_df, x="x", y="y", binwidth=(.5, .5), cbar=True)
    g.set(xlabel='Agent X Position', ylabel='Agent Y Position')

    plt.ylim(-10, 10)
    plt.xlim(-15, 15)
    plt.xlabel('Agent X Position',fontsize=20)
    plt.ylabel('Agent Y Position',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.show()