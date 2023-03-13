import matplotlib.pyplot as plt
import random
from classes import Bin, Item

def plot_bins(list_bins,t,titles):
    initial_capacity=list_bins[0][0].initial_capacity
    fig, ax = plt.subplots((1+len(list_bins))//2,2,figsize=(14,14))
    
    ind=0
    axes=[]

    for a in ax:
        if len(a)>1:
            for b in a:
                axes.append(b)
        else:
            axes.append(a)
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
        axes[ind].set_title(f'{titles[ind]} , T={t[ind]}s', fontsize=12, weight='bold')
        for i in range(len(bins)):
            # shuffles the colors
            k=0
            random.shuffle(colors)
            axes[ind].bar(i+1, initial_capacity, width=0.9,bottom=0, color='white',edgecolor='black',hatch=r"\\")
            for j in range(len(bin_contents[i])):
                axes[ind].bar(i+1, bins[i].items[j].size, width=0.9,bottom=k, color=colors[j%len(colors)],edgecolor='black')
                k=k+bins[i].items[j].size

        axes[ind].set_yticks(list(range(initial_capacity + 1)))
        axes[ind].set_xticks(list(range(1, nb_bins + 1)))

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
    set_bins=set(bins)
    s=hex(len(items))[2:]
    for bin in set_bins:
        s=s+hex(bin.capacity)[2:]
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
