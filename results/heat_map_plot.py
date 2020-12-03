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
    ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/ddpg_bin_shoot_opp.csv")

    # penguins = sns.load_dataset("penguins")
    print(ddpg_df)
    g = sns.displot(ddpg_df, x="x", y="y", binwidth=(.5, .5), cbar=True)
    # g = sns.displot(ddpg_df, x="bx", y="by", binwidth=(.5, .5), cbar=True, color="r")
    g.set(xlabel='Ball X Position', ylabel='Ball Y Position')
    # sns.lmplot(x="x", y="y",data=ddpg_df, fit_reg=False)
    # sns.displot(ddpg_df, x="o1", y="o2", binwidth=(2, .5), cbar=True)
    # fig, ax = plt.subplots()
    # ax.plot(x, y, marker='s', linestyle='none', label='small')
    # ax.legend(loc='upper left', fontsize=20,bbox_to_anchor=(0, 1.1))
    # ax.set_xlabel('X_axi',fontsize=20)
    # ax.set_ylabel('Y_axis',fontsize=20)

    plt.ylim(-30, 30)
    plt.xlim(5, 55)
    plt.xlabel('Agent X Position',fontsize=20)
    plt.ylabel('Agent Y Position',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.show()

if exp == 2:
    ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/sac_a1_r1.csv")

    # penguins = sns.load_dataset("penguins")
    g = sns.displot(ddpg_df, x="x", y="y", binwidth=(.5, .5), cbar=True)
    g.set(xlabel='Agent X Position', ylabel='Agent Y Position')
    # sns.lmplot(x="x", y="y",data=ddpg_df, fit_reg=False)
    # sns.displot(ddpg_df, x="o1", y="o2", binwidth=(2, .5), cbar=True)

    plt.ylim(-10, 10)
    plt.xlim(-15, 15)
    plt.xlabel('Agent X Position',fontsize=20)
    plt.ylabel('Agent Y Position',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    plt.show()

# ./shell/run_agent_play.sh go_to_ball/go_to_ball_ddpg 1000 200 0
# ./shell/run_agent_play.sh ball_to_goal/ball_to_goal_sac 1000 200 11