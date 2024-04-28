# %% 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property

import pandas as pd


from src.utils.config import Config
from src.plots.dtu_colors import DTUColors

import pdb

def update_elo(winner_elo, loser_elo, k=32):
    """ Update ELO ratings for the winner and loser of a match """
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_winner_elo = winner_elo + k * (1 - expected_win)
    new_loser_elo = loser_elo + k * (0 - (1 - expected_win))
    return new_winner_elo, new_loser_elo

def plot_elo(elo_df: pd.DataFrame, dtu_colors: DTUColors, save_path: str = None):

    color = dtu_colors.get_secondary_color("dtulightgreen")

    # make histogram of elo scores
    # side by side with ECDF plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    ax = axs[0]
    ax.hist(elo_df["elo"], bins=30, color=color, edgecolor='white', linewidth=1.2)
    ax.set_xlabel("Elo Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Elo Scores")

    # make ECDF plot
    ax = axs[1]
    elo_sorted = elo_df["elo"].sort_values()
    n = elo_sorted.size
    y = np.arange(1, n+1) / n
    ax.plot(elo_sorted, y, color=color)
    ax.set_xlabel("Elo Score")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of Elo Scores")

    # remove white background from plot 
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor('none')
    fig.patch.set_facecolor('none')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig("figures/elo_score_hist_ecdf.pdf")

def plot_elo_history(elo_df: pd.DataFrame, matches_file_path: str, top_n: int, dtu_colors: DTUColors, save_path: str = None):
    top_games = elo_df.sort_values("games", ascending=False).head(top_n)

    elo_history = {name: [] for name in elo_df.index}

    # reset elo in elo_df to baseline
    elo_df["elo"] = 1000

    # Ensure the DataFrame is indexed by the 'name' for easy look-up
    elo_df.set_index('name', inplace=True)

    # Initialize a dictionary to store ELO history for plotting
    elo_history = {name: [] for name in elo_df.index}

    with open(matches_file_path, "r") as f:
        for line in f:
            img1, img2, winner = line.strip().split(',')
            winner = int(winner)

            # Get the current ELO scores
            elo1 = elo_df.at[img1, 'elo']
            elo2 = elo_df.at[img2, 'elo']

            # Update based on who won
            if winner == 0:
                new_elo1, new_elo2 = update_elo(elo1, elo2)
            else:
                new_elo2, new_elo1 = update_elo(elo2, elo1)

            # Store the updated ELO scores back into the DataFrame
            elo_df.at[img1, 'elo'] = new_elo1
            elo_df.at[img2, 'elo'] = new_elo2

            # Record ELO history for each game
            elo_history[img1].append(new_elo1)
            elo_history[img2].append(new_elo2)


    # get name column from top_five_games
    top_five_names = top_games["name"]

    # get elo history for top five games
    top_five_elo_history = {name: elo_history[name] for name in top_five_names}

    colors_list = list(dtu_colors.get_secondary_color_dict().values())
    len_colors = len(colors_list)

    # plot top five elo history
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, history) in enumerate(top_five_elo_history.items()):
        x = np.arange(1,len(history)+1)
        # color, rotate through colors
        color = colors_list[i % len_colors]
        ax.plot(x, history, label=name, color=color)

    ax.set_xlabel("Game Number")
    ax.set_ylabel("ELO Score")
    ax.set_title(f"ELO History of Top {top_n} Games")
    # no background
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor('none')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig("figures/top_elo_history.pdf")

if __name__ == "__main__":

    cfg = Config("/Users/alf/Repos/adv_dl_in_cv_exam/configs/config.yaml")

    dtu_colors = DTUColors()


    elo_file_name = "calle2.csv"
    matches_file = "calle2_matches.txt"

    elo_annotations_path = cfg.get("data", "elo_annotations_path")

    elo_file_path = os.path.join(elo_annotations_path, elo_file_name)
    matches_file_path = os.path.join(elo_annotations_path, matches_file)

    # columns: name, path, elo, games, discard
    elo_df = pd.read_csv(elo_file_path)

    plot_elo(elo_df, dtu_colors)

    # top games from elo_df
    top_n = 10
    plot_elo_history(elo_df, matches_file_path, top_n, dtu_colors)
    

    


