import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import os
import random
from lib.classes import Bin, Item
import numpy as np

def plot_heatmap(x, y, z, x_label, y_label, z_label, graph_name):
    # Vérifier que les dimensions sont correctes
    assert len(x) * len(y) == len(z), "Les dimensions des listes ne correspondent pas."

    # Convertir les listes en tableaux 2D avec meshgrid
    X, Y = np.meshgrid(x, y)
    Z = np.reshape(z, (len(y), len(x)))

    # Créer la figure et l'axe
    fig, ax = plt.subplots()

    # Créer le heatmap
    heatmap = ax.pcolor(X, Y, Z, cmap='hot')
    # Ajouter une légende
    cbar = plt.colorbar(heatmap)

    # Paramètres des axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(graph_name)
    # Afficher les valeurs dans les cellules
    # for i in range(len(y)):
    #     for j in range(len(x)):
    #         ax.text(j+0.5, i+0.5, f'{Z[i, j]:.2f}', ha='center', va='center', color='black')
    
    # fig.tight_layout(pad=1.0)
    # Afficher le heatmap
    # plt.show()
    plt.savefig(os.path.join("plots", f"{graph_name}.png"))


def plot_bar_chart(x, y, x_label, y_label, graph_name):
    # Créer la figure et l'axe
    fig, ax = plt.subplots()
    # increase the space between bars
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    # Créer le bar chart
    ax.bar(x, y, width=0.5, color='blue', edgecolor='white')
    # Paramètres des axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set y axe only between min(y) and max(y)
    # ax.set_ylim(min(y) - 1, max(y) + 1)
    # add space between bars
    ax.set_xticks(x)

    ax.set_title(graph_name)
    # Afficher le bar chart
    # plt.show()
    plt.savefig(os.path.join("plots", f"{graph_name}.png"))

def plot_5_bins(list_bins,t,titles):
    
    gs = gridspec.GridSpec(3, 5)

    fig = pl.figure(figsize=(14,14))
    ax1 = pl.subplot(gs[0, 0:2]) # row 0, col 0

    ax2 = pl.subplot(gs[0, 2:4]) # row 0, col 1

    ax3 = pl.subplot(gs[1, 0:2]) 

    ax4 = pl.subplot(gs[1, 2:4])

    ax5 = pl.subplot(gs[2, 1:3])



    initial_capacity=list_bins[0][0].initial_capacity
    
    ind=0
    axes = [ax1,ax2,ax3,ax4,ax5]

    for bins in list_bins:
        bin_labels = [f'Bin {i+1}' for i in range(len(bins))]
        nb_bins = len(bins)

        # Iterate over each bin and obtain the sizes of the items in the bin
        bin_contents = []
        for bin in bins:
            contents = [item.size for item in bin.items]
            bin_contents.append(contents)

        # Define a list of colors to use for each item in the bin
        colors = ['navy','lightblue','darkgreen']


        # Define a list of positions for each item in the bin
        positions = []
        for contents in bin_contents:
            pos = [0]
            for i in range(len(contents)-1):
                pos.append(pos[-1] + contents[i])
            positions.append(pos)
        
        # Create a vertical bar chart with the item sizes on the x-axis and position in the bin on the y-axis
        axes[ind].set_title(f'{titles[ind]} , T={t[ind]}s, nb_bins = {len(bins)}', fontsize=12, weight='bold')
        for i in range(len(bins)):
            # shuffles the colors
            k=0
            random.shuffle(colors)
            axes[ind].bar(i+1, initial_capacity, width=0.9,bottom=0, color='white',edgecolor='black',hatch=r"\\")
            for j in range(len(bin_contents[i])):
                axes[ind].bar(i+1, bins[i].items[j].size, width=0.9,bottom=k, color=colors[j%len(colors)],edgecolor='black')
                k=k+bins[i].items[j].size

        # axes[ind].set_yticks(list(range(initial_capacity + 1)))
        # axes[ind].set_xticks(list(range(1, nb_bins + 1)))

        # Set the y-axis limits based on the maximum size of the items in all the bins
        axes[ind].set_ylim(0, initial_capacity+0.4)

        # Label the x-axis and y-axis
        axes[ind].set_xlabel('Bins')
        axes[ind].set_ylabel('Position in Bin')
        
        ind += 1
    fig.tight_layout(pad=6.0)
    # Show the plot
    plt.show()

def plot_1_bin(bins,t,title, nb_items, fig_name):
    gs = gridspec.GridSpec(1, 1)

    fig = pl.figure(figsize=(8,8))
    ax = pl.subplot(gs[0, 0])

    initial_capacity=bins[0].initial_capacity

    # Iterate over each bin and obtain the sizes of the items in the bin
    bin_contents = []
    for bin in bins:
        contents = [item.size for item in bin.items]
        bin_contents.append(contents)

    # Define a list of colors to use for each item in the bin
    colors = ['navy','lightblue','darkgreen']


    # Define a list of positions for each item in the bin
    positions = []
    for contents in bin_contents:
        pos = [0]
        for i in range(len(contents)-1):
            pos.append(pos[-1] + contents[i])
        positions.append(pos)
    
    # Create a vertical bar chart with the item sizes on the x-axis and position in the bin on the y-axis
    ax.set_title(f'{title}\n temps pris={t}s\nnombre de bins = {len(bins)}\nnombre de items = {nb_items}', fontsize=12, weight='bold')
    for i in range(len(bins)):
        # shuffles the colors
        k=0
        random.shuffle(colors)
        ax.bar(i+1, initial_capacity, width=0.9,bottom=0, color='white',edgecolor='black',hatch=r"\\")
        for j in range(len(bin_contents[i])):
            ax.bar(i+1, bins[i].items[j].size, width=0.9,bottom=k, color=colors[j%len(colors)],edgecolor='black')
            k=k+bins[i].items[j].size

    # ax.set_yticks(list(range(initial_capacity + 1)))
    # ax.set_xticks(list(range(1, nb_bins + 1)))

    # Set the y-axis limits based on the maximum size of the items in all the bins
    ax.set_ylim(0, initial_capacity+0.4)

    # Label the x-axis and y-axis
    ax.set_xlabel('Bins')
    ax.set_ylabel('Position in Bin')
        
    fig.tight_layout(pad=6.0)
    # Show the plot
    plt.show()
    fig.savefig(f'./plots/{fig_name}.png')

def plot_2_bins(list_bins,t,titles):
    
    gs = gridspec.GridSpec(1, 2)

    fig = pl.figure(figsize=(14,14))
    ax1 = pl.subplot(gs[0, 0]) # row 0, col 0

    ax2 = pl.subplot(gs[0, 1]) # row 0, co



    initial_capacity=list_bins[0][0].initial_capacity
    
    ind=0
    axes = [ax1,ax2]

    for bins in list_bins:
        bin_labels = [f'Bin {i+1}' for i in range(len(bins))]
        nb_bins = len(bins)

        # Iterate over each bin and obtain the sizes of the items in the bin
        bin_contents = []
        for bin in bins:
            contents = [item.size for item in bin.items]
            bin_contents.append(contents)

        # Define a list of colors to use for each item in the bin
        colors = ['navy','lightblue','darkgreen']


        # Define a list of positions for each item in the bin
        positions = []
        for contents in bin_contents:
            pos = [0]
            for i in range(len(contents)-1):
                pos.append(pos[-1] + contents[i])
            positions.append(pos)
        
        # Create a vertical bar chart with the item sizes on the x-axis and position in the bin on the y-axis
        axes[ind].set_title(f'{titles[ind]} , T={t[ind]}s, nb_bins = {len(bins)}', fontsize=12, weight='bold')
        for i in range(len(bins)):
            # shuffles the colors
            k=0
            random.shuffle(colors)
            axes[ind].bar(i+1, initial_capacity, width=0.9,bottom=0, color='white',edgecolor='black',hatch=r"\\")
            for j in range(len(bin_contents[i])):
                axes[ind].bar(i+1, bins[i].items[j].size, width=0.9,bottom=k, color=colors[j%len(colors)],edgecolor='black')
                k=k+bins[i].items[j].size

        # axes[ind].set_yticks(list(range(initial_capacity + 1)))
        # axes[ind].set_xticks(list(range(1, nb_bins + 1)))

        # Set the y-axis limits based on the maximum size of the items in all the bins
        axes[ind].set_ylim(0, initial_capacity+0.4)

        # Label the x-axis and y-axis
        axes[ind].set_xlabel('Bins')
        axes[ind].set_ylabel('Position in Bin')
        
        ind += 1
    fig.tight_layout(pad=6.0)
    # Show the plot
    plt.show()
    
def state_hash(items,bins): 
    set_bins=sorted(bins)
    s=hex(len(items))[2:]
    for bin in set_bins:
        s=s+hex(bin.capacity)[2:] + " "
    return s

def print_result(items,bins):
    print(f"Number of bins: {len(bins)}")
    print(f"Bins capacity: {bins[0].initialcapacity}")
    print(f"Items: {[str(item) for item in items]}")
    for bin in bins:
        print(bin)
        bin.print_items()

def genereate_items(number_items,max_size):
    items = []
    for i in range(number_items):
        items.append(Item(random.randint(1, max_size)))
    return items
