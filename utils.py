import matplotlib.pyplot as plt
import random

def plot_bins(list_bins,t,titles):

    initialcapacity=list_bins[0][0].initialcapacity
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
        axes[ind].set_title(f'{titles[ind]} , T={t[ind]}s', fontsize=12)
        for i in range(len(bins)):
            # shuffles the colors
            k=0
            random.shuffle(colors)
            axes[ind].bar(i, initialcapacity, width=0.9,bottom=0, color='white',edgecolor='black',hatch=r"\\")
            for j in range(len(bin_contents[i])):
                axes[ind].bar(i, bins[i].items[j].size, width=0.9,bottom=k, color=colors[j%len(colors)],edgecolor='black')
                k=k+bins[i].items[j].size

        # Set the x-axis tick labels to the bin labels
        #plt.xticks(range(len(bins)), bin_labels)

        # Set the y-axis limits based on the maximum size of the items in all the bins
        axes[ind].set_ylim(0, initialcapacity+0.4)

        # Label the x-axis and y-axis
        plt.xlabel('Bins')
        plt.ylabel('Position in Bin')
        ind+=1

    # Show the plot
    plt.show()