import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

EXPERIMENT = 4
SMOOTHING_FACTOR = 5

if EXPERIMENT == 1:
    sac_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-23_21-19-16_SAC_GO_TO_BALL_A1_R1-tag-Rewards_epi_reward.csv")
    td3_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-23_23-02-55_TD3_GO_TO_BALL_A1_R1-tag-Rewards_epi_reward.csv")
    ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-23_22-19-41_DDPG_GO_TO_BALL_A1_R1-tag-Rewards_epi_reward.csv")
    # ddpg_lr3_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-24_00-12-30_DDPG_GO_TO_BALL_A1_R1_LR3-tag-Rewards_epi_reward.csv")
    del sac_df["Wall time"]
    del td3_df["Wall time"]
    del ddpg_df["Wall time"]
    # del ddpg_lr3_df["Wall time"]

    smoothing_count = 0
    while smoothing_count < SMOOTHING_FACTOR:
        sac_df["Value"] = sac_df["Value"].ewm(com=0.99).mean()
        td3_df["Value"] = td3_df["Value"].ewm(com=0.99).mean()
        ddpg_df["Value"] = ddpg_df["Value"].ewm(com=0.99).mean()
        # ddpg_lr3_df["Value"] = ddpg_lr3_df["Value"].ewm(com=0.99).mean()
        smoothing_count+=1

    sac_df["Model"] = "SAC"
    td3_df["Model"] = "TD3"
    ddpg_df["Model"] = "DDPG"
    # ddpg_lr3_df["Model"] = "DDPG LR MODIFIED"

    frames = [sac_df, td3_df, ddpg_df]
    result = pd.concat(frames)

    g = sns.relplot(x="Step", y="Value", hue="Model", kind="line", data=result)
    g.set(xlabel='Episode', ylabel='Reward Per Episode')
    g.fig.autofmt_xdate()

    plt.show()

elif EXPERIMENT == 2:
    sac_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-23_21-46-23_SAC_GO_TO_BALL_WITH_POWER_A2_R1-tag-Rewards_epi_reward.csv")
    td3_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-24_22-01-29_TD3_GO_TO_BALL_WITH_POWER_A2_R1-tag-Rewards_epi_reward.csv")
    ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-24_18-56-04_DDPG_GO_TO_BALL_WITH_POWER_A2_R1-tag-Rewards_epi_reward.csv")
    del sac_df["Wall time"]
    del td3_df["Wall time"]
    del ddpg_df["Wall time"]

    sac_df = sac_df.drop(sac_df[sac_df["Step"] > 5000].index)
    ddpg_df = ddpg_df.drop(ddpg_df[ddpg_df["Step"] > 5000].index)

    smoothing_count = 0
    while smoothing_count < SMOOTHING_FACTOR:
        sac_df["Value"] = sac_df["Value"].ewm(com=0.99).mean()
        td3_df["Value"] = td3_df["Value"].ewm(com=0.99).mean()
        ddpg_df["Value"] = ddpg_df["Value"].ewm(com=0.99).mean()
        smoothing_count+=1

    sac_df["Model"] = "SAC"
    td3_df["Model"] = "TD3"
    ddpg_df["Model"] = "DDPG"

    frames = [sac_df, td3_df, ddpg_df]
    result = pd.concat(frames)

    g = sns.relplot(x="Step", y="Value", hue="Model", kind="line", data=result)
    g.set(xlabel='Episode', ylabel='Reward Per Episode')
    g.fig.autofmt_xdate()

    plt.show()

elif EXPERIMENT == 3:
    sac_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-25_21-01-22_SAC_BALL_TO_GOAL_A7_R5_S3-tag-Rewards_epi_reward.csv")
    td3_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-25_20-34-06_TD3_BALL_TO_GOAL_A7_R5_S3-tag-Rewards_epi_reward.csv")
    ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-26_21-27-17_DDPG_BALL_TO_GOAL_A7_R5_S3-tag-Rewards_epi_reward.csv")
    del sac_df["Wall time"]
    del td3_df["Wall time"]
    del ddpg_df["Wall time"]

    sac_df = sac_df.drop(sac_df[sac_df["Step"] > 15000].index)
    td3_df = td3_df.drop(td3_df[td3_df["Step"] > 15000].index)
    ddpg_df = ddpg_df.drop(ddpg_df[ddpg_df["Step"] > 15000].index)

    smoothing_count = 0
    while smoothing_count < SMOOTHING_FACTOR:
        sac_df["Value"] = sac_df["Value"].ewm(com=0.99).mean()
        td3_df["Value"] = td3_df["Value"].ewm(com=0.99).mean()
        ddpg_df["Value"] = ddpg_df["Value"].ewm(com=0.99).mean()
        smoothing_count+=1

    sac_df["Model"] = "SAC"
    td3_df["Model"] = "TD3"
    ddpg_df["Model"] = "DDPG"

    frames = [sac_df, td3_df, ddpg_df]
    result = pd.concat(frames)

    g = sns.relplot(x="Step", y="Value", hue="Model", kind="line", data=result)
    g.set(xlabel='Episode', ylabel='Reward Per Episode')
    g.fig.autofmt_xdate()

    plt.show()

elif EXPERIMENT == 4:
    sac_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-27_09-40-34_SAC_BALL_TO_GOAL_A7_R3_S3-tag-Rewards_epi_reward.csv")
    td3_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-27_20-07-41_TD3_BALL_TO_GOAL_A7_R3_S3-tag-Rewards_epi_reward.csv")
    # ddpg_df = pd.read_csv("/Users/cristianooliveira/personal-development/tcc/tg-repo/results/csv/run_2020-09-26_21-27-17_DDPG_BALL_TO_GOAL_A7_R5_S3-tag-Rewards_epi_reward.csv")
    del sac_df["Wall time"]
    del td3_df["Wall time"]
    # del ddpg_df["Wall time"]

    sac_df = sac_df.drop(sac_df[sac_df["Step"] > 15000].index)
    td3_df = td3_df.drop(td3_df[td3_df["Step"] > 15000].index)
    # ddpg_df = ddpg_df.drop(ddpg_df[ddpg_df["Step"] > 15000].index)

    smoothing_count = 0
    while smoothing_count < SMOOTHING_FACTOR:
        sac_df["Value"] = sac_df["Value"].ewm(com=0.99).mean()
        td3_df["Value"] = td3_df["Value"].ewm(com=0.99).mean()
        # ddpg_df["Value"] = ddpg_df["Value"].ewm(com=0.99).mean()
        smoothing_count+=1

    sac_df["Model"] = "SAC"
    td3_df["Model"] = "TD3"
    # ddpg_df["Model"] = "DDPG"

    # frames = [sac_df, td3_df, ddpg_df]
    frames = [sac_df, td3_df]
    result = pd.concat(frames)

    g = sns.relplot(x="Step", y="Value", hue="Model", kind="line", data=result)
    g.set(xlabel='Episode', ylabel='Reward Per Episode')
    g.fig.autofmt_xdate()

    plt.show()