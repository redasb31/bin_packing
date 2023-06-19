import numpy as np
import matplotlib.pyplot as plt
import os


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
    for i in range(len(y)):
        for j in range(len(x)):
            ax.text(j+0.5, i+0.5, f'{Z[i, j]:.2f}', ha='center', va='center', color='black')
    
    # fig.tight_layout(pad=1.0)
    # Afficher le heatmap
    plt.show()
    plt.savefig(os.path.join("plots", f"{graph_name}.png"))



# def plot_3d_surface(x, y, z, x_label, y_label, z_label, graph_name):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_zlabel(z_label)
#     ax.set_title(graph_name)
    
#     plt.savefig(f"plots/{graph_name}.png")
#     plt.show()





# def plot_heatmap(x, y, z, x_label, y_label, graph_name):
#     plt.imshow(z, extent=[min(x), max(x), min(y), max(y)], origin='lower', aspect='auto')
#     plt.colorbar()
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(graph_name)
    
#     plt.savefig(f"plots/{graph_name}.png")
#     plt.show()