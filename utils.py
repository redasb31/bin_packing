import matplotlib.pyplot as plt
import random

def plot_bins(bins):

    # Create a list of labels for each bin based on their index
    bin_labels = [f'Bin {i+1}' for i in range(len(bins))]
    initialcapacity=bins[0].initialcapacity

    # Iterate over each bin and obtain the sizes of the items in the bin
    bin_contents = []
    for bin in bins:
        contents = [item.size for item in bin.items]
        bin_contents.append(contents)

    # Define a list of colors to use for each item in the bin
    colors = ['orange', 'purple', 'brown', 'pink', 'olive', 'cyan']


    # Define a list of positions for each item in the bin
    positions = []
    for contents in bin_contents:
        pos = [0]
        for i in range(len(contents)-1):
            pos.append(pos[-1] + contents[i])
        positions.append(pos)

    # Create a vertical bar chart with the item sizes on the x-axis and position in the bin on the y-axis
    fig, ax = plt.subplots()
    for i in range(len(bins)):
        # shuffles the colors
        random.shuffle(colors)
        ax.bar(i, initialcapacity, width=0.95,bottom=0, color='gray')
        for j in range(len(bin_contents[i])):
            ax.bar(i, bin_contents[i][j], width=0.9,bottom=positions[i][j] + 0.1, color=colors[j%len(colors)])

    # Set the x-axis tick labels to the bin labels
    plt.xticks(range(len(bins)), bin_labels)

    # Set the y-axis limits based on the maximum size of the items in all the bins
    ax.set_ylim(0, initialcapacity)

    # Label the x-axis and y-axis
    plt.xlabel('Bins')
    plt.ylabel('Position in Bin')

    # Show the plot
    plt.show()