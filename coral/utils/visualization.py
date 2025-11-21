import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from scipy.stats import gaussian_kde
import networkx as nx
from torch_geometric.utils import to_networkx
from umap import UMAP
import seaborn as sns

def plot_spatial(
    adata,
    res: float = 0.1,
    size: float = 15,
    use_rep_for_cluster: str | None = None,
    to_plot_var: str | None = None, 
    need_lognormed: bool | None = None,
    color_list: list[str] | None = None,
    figsize: tuple[float, float] = (4, 4),
    legd: bool = False,
    invert_yaxis: bool = True,
    axis_: bool = False,
    file_name: str | None = None,
    bbox_to_anchor: tuple[float, float] = (1.01, 0.8),
    legend_fontsize: int = 10,
    legend_markerscale: float = 5,
    return_cluster: bool = False, 
    n_neighbors: int = 100,
):
    """
    Plot spatial clusters from a spatial transcriptomics AnnData object.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix with `.obsm['spatial']` coordinates.
    res : float, optional
        Resolution parameter for Leiden clustering (default: 0.1).
    size : float, optional
        Marker size in the scatter plot (default: 15).
    use_rep_for_cluster : str or None, optional
        Key in `adata.obsm` to use for neighbor graph computation (e.g. 'X_pca').
        If None, PCA will be computed automatically.
    need_lognormed : bool or None, optional
        If True, applies Scanpy’s total count normalization and log1p.
        If None, assumes pre-normalized input.
    color_list : list of str, optional
        List of colors for clusters. If None, a default color palette will be used.
    figsize : tuple, optional
        Figure size in inches (default: (4, 4)).
    legd : bool, optional
        Whether to display the legend (default: False).
    return_ : bool, optional
        If True, returns `(fig, ax, adata_)` instead of displaying only.
    invert_yaxis : bool, optional
        Whether to invert y-axis (default: True, consistent with spatial plots).
    axis_ : bool, optional
        Whether to display axis ticks and frames (default: False).
    file_name : str or None, optional
        Optional path to save the figure (e.g. "spatial_cluster.pdf").
    bbox_to_anchor : tuple, optional
        Anchor position of the legend (default: (1.01, 0.8)).
    legend_fontsize : int, optional
        Font size of legend labels (default: 10).
    legend_markerscale : float, optional
        Scale factor of legend markers (default: 5).

    Returns
    -------
    fig, ax, adata_ : tuple
        Returned only if `return_` is True.
    """

    # Copy input to avoid modifying the original AnnData
    adata_ = adata.copy()

    # Optional normalization
    if need_lognormed:
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)

    if to_plot_var=='cluster' and "cluster" not in adata_.obsm:
                
        if use_rep_for_cluster=='X_pca':
            sc.tl.pca(adata_, svd_solver="arpack")
            use_rep_for_cluster = "X_pca"
    
        # Compute neighbors and Leiden clusters
        sc.pp.neighbors(adata_, n_neighbors=n_neighbors, use_rep=use_rep_for_cluster)
        sc.tl.leiden(adata_, resolution=res, random_state=0, flavor="igraph")
        #sc.tl.leiden(adata_, resolution=0.8)
        adata_.obs["cluster"] = adata_.obs["leiden"].astype(str)

    # Choose colors
    if color_list is None:
        color_list = [
            "#f94144", "#f3722c", "#f8961e", "#f9c74f",
            "#90be6d", "#43aa8b", "#577590", "#226CE0", "#534B62"
        ]
    colors = (color_list * ((len(adata_.obs[to_plot_var].unique()) // len(color_list)) + 1))[
        : len(adata_.obs[to_plot_var].unique())
    ]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=500)

    for j, cluster in enumerate(sorted(adata_.obs[to_plot_var].unique())):
        coords = adata_.obsm["spatial"][adata_.obs[to_plot_var] == cluster]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=size,
            color=colors[j],
            rasterized=True,
            label=cluster,
        )

    # Legend
    if legd:
        ax.legend(
            bbox_to_anchor=bbox_to_anchor,
            prop={"size": legend_fontsize},
            markerscale=legend_markerscale,
            frameon=False,
        )

    if invert_yaxis:
        ax.invert_yaxis()
    if not axis_:
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, bbox_inches="tight", transparent=True)

    
    if return_cluster:
        return adata_.obs["leiden"].astype(str)
    else:
        plt.show()


def plot_latent_umap(adata, rep = 'X',to_plot_var = 'cluster',custom_palette = ['#f4f1de',  '#81b29a', '#f2cc8f', '#e07a5f','#3d405b',]):

    latent_rep = adata.obsm[rep]
    umap_model = UMAP(n_neighbors=30, min_dist=0.2, n_components=2, random_state=42)
    
    # Fit the model to your data and transform it
    z_umap = umap_model.fit_transform(latent_rep)
    adata.obsm['umap'] = z_umap 

    plt.figure(dpi=500,figsize=(3.4,3))
    sc.pl.embedding(adata, basis='umap', title='Coral', color=to_plot_var, s=70, show=False, palette=custom_palette)
    plt.xlabel('')
    plt.ylabel('')

def plot_umap(
    adata,
    res: float = 0.1,
    size: float = 15,
    use_rep_for_cluster: str | None = None,
    to_plot_var: str | None = None, 
    need_lognormed: bool | None = None,
    color_list: list[str] | None = None,
    figsize: tuple[float, float] = (4, 4),
    legd: bool = False,
    return_: bool = False,
    invert_yaxis: bool = True,
    axis_: bool = False,
    file_name: str | None = None,
    bbox_to_anchor: tuple[float, float] = (1.01, 0.8),
    legend_fontsize: int = 10,
    legend_markerscale: float = 5,

):
    """
    Plot spatial clusters from a spatial transcriptomics AnnData object.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix with `.obsm['spatial']` coordinates.
    res : float, optional
        Resolution parameter for Leiden clustering (default: 0.1).
    size : float, optional
        Marker size in the scatter plot (default: 15).
    use_rep : str or None, optional
        Key in `adata.obsm` to use for neighbor graph computation (e.g. 'X_pca').
        If None, PCA will be computed automatically.
    need_lognormed : bool or None, optional
        If True, applies Scanpy’s total count normalization and log1p.
        If None, assumes pre-normalized input.
    color_list : list of str, optional
        List of colors for clusters. If None, a default color palette will be used.
    figsize : tuple, optional
        Figure size in inches (default: (4, 4)).
    legd : bool, optional
        Whether to display the legend (default: False).
    return_ : bool, optional
        If True, returns `(fig, ax, adata_)` instead of displaying only.
    invert_yaxis : bool, optional
        Whether to invert y-axis (default: True, consistent with spatial plots).
    axis_ : bool, optional
        Whether to display axis ticks and frames (default: False).
    file_name : str or None, optional
        Optional path to save the figure (e.g. "spatial_cluster.pdf").
    bbox_to_anchor : tuple, optional
        Anchor position of the legend (default: (1.01, 0.8)).
    legend_fontsize : int, optional
        Font size of legend labels (default: 10).
    legend_markerscale : float, optional
        Scale factor of legend markers (default: 5).

    Returns
    -------
    fig, ax, adata_ : tuple
        Returned only if `return_` is True.
    """

    # Copy input to avoid modifying the original AnnData
    adata_ = adata.copy()

    # Optional normalization
    if need_lognormed:
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)

    if to_plot_var=='cluster' and "cluster" not in adata_.obsm:
                
        if use_rep_for_cluster=='X_pca':
            sc.tl.pca(adata_, svd_solver="arpack")
            use_rep_for_cluster = "X_pca"
    
        # Compute neighbors and Leiden clusters
        sc.pp.neighbors(adata_, n_neighbors=100, use_rep=use_rep_for_cluster)
        sc.tl.leiden(adata_, resolution=res, random_state=0, flavor="igraph")
    
        adata_.obs["cluster"] = adata_.obs["leiden"].astype(str)

    if "X_pca" not in adata_.obsm:
        sc.tl.pca(adata_, svd_solver="arpack")
    if "neighbors" not in adata_.uns:
        sc.pp.neighbors(adata_, n_neighbors=100)
    # Choose colors
    if color_list is None:
        color_list = [
            "#f94144", "#f3722c", "#f8961e", "#f9c74f",
            "#90be6d", "#43aa8b", "#577590", "#226CE0", "#534B62"
        ]
    colors = (color_list * ((len(adata_.obs[to_plot_var].unique()) // len(color_list)) + 1))[
        : len(adata_.obs[to_plot_var].unique())
    ]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=500)

    sc.tl.umap(adata_,min_dist=0.5) 
    for j, cluster in enumerate(sorted(adata_.obs[to_plot_var].unique().astype(str))):
        coords = adata_.obsm["X_umap"][adata_.obs[to_plot_var] == cluster]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=size,
            edgecolor='none',
            alpha = 0.5,    
            color=colors[j],
            rasterized=True,
            label=cluster,
        )

    # Legend
    if legd:
        ax.legend(
            bbox_to_anchor=bbox_to_anchor,
            prop={"size": legend_fontsize},
            markerscale=legend_markerscale,
            frameon=False,
        )

    if invert_yaxis:
        ax.invert_yaxis()
    if not axis_:
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, bbox_inches="tight", transparent=True)

    if return_:
        return fig, ax, adata_
    else:
        plt.show()


def plot_umap_gene(
    adata,
    res: float = 0.1,
    size: float = 15,
    use_rep_for_cluster: str | None = None,
    to_plot_gene: str | None = None, 
    need_lognormed: bool | None = None,
    color_list: list[str] | None = None,
    figsize: tuple[float, float] = (4, 4),
    legd: bool = False,
    return_: bool = False,
    invert_yaxis: bool = True,
    axis_: bool = False,
    file_name: str | None = None,
    bbox_to_anchor: tuple[float, float] = (1.01, 0.8),
    legend_fontsize: int = 10,
    legend_markerscale: float = 5,
        vmin: float = 0,
    vmax: float | None = None,
):
    # Copy input to avoid modifying the original AnnData
    adata_ = adata.copy()

    # Optional normalization
    if need_lognormed:
        sc.pp.normalize_total(adata_, target_sum=1e4)
        sc.pp.log1p(adata_)


    if "X_pca" not in adata_.obsm:
        sc.tl.pca(adata_, svd_solver="arpack")
    if "neighbors" not in adata_.uns:
        sc.pp.neighbors(adata_, n_neighbors=100)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=500)

    sc.tl.umap(adata_,min_dist=0.5) 
    coords = adata_.obsm["X_umap"]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=size,
        c = adata_[:,to_plot_gene].to_df().values,
        cmap = 'Spectral_r',
        edgecolor='none',
        alpha = 0.5,    
        rasterized=True,
        vmin=vmin,
        vmax=vmax
    )

    # Legend
    if legd:
        ax.legend(
            bbox_to_anchor=bbox_to_anchor,
            prop={"size": legend_fontsize},
            markerscale=legend_markerscale,
            frameon=False,
        )

    if invert_yaxis:
        ax.invert_yaxis()
    if not axis_:
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, bbox_inches="tight", transparent=True)

    if return_:
        return fig, ax, adata_
    else:
        plt.show()    
def display_reconst(df_true,
                    df_pred,
                    density=False,
                    marker_genes=None,
                    sample_rate=0.01,
                    size=(4, 4),
                    spot_size=1,
                    title=None,
                    x_label='',
                    y_label='',
                    min_val=None,
                    max_val=None,
                    ):
    """
    Scatter plot - raw gexp vs. reconstructed gexp
    """
    assert 0 < sample_rate <= 1, \
        "Invalid downsampling rate for reconstruct scatter plot: {}".format(sample_rate)

    if marker_genes is not None:
        marker_genes = set(marker_genes)

    df_true_sample = df_true.sample(frac=sample_rate, random_state=0)
    df_pred_sample = df_pred.loc[df_true_sample.index]

    plt.rcParams["figure.figsize"] = size
    plt.figure(dpi=800)
    ax = plt.gca()

    xx = df_true_sample.T.to_numpy().flatten()
    yy = df_pred_sample.T.to_numpy().flatten()

    if density:
        for gene in df_true_sample.columns:
            try:
                gene_true = df_true_sample[gene].values
                gene_pred = df_pred_sample[gene].values
                gexp_stacked = np.vstack([df_true_sample[gene].values, df_pred_sample[gene].values])

                z = gaussian_kde(gexp_stacked)(gexp_stacked)
                ax.scatter(gene_true, gene_pred, c=z, s=spot_size, alpha=0.5)
            except np.linalg.LinAlgError as e:
                pass

    elif marker_genes is not None:
        color_dict = {True: 'red', False: 'green'}
        gene_colors = np.vectorize(
            lambda x: color_dict[x in marker_genes]
        )(df_true_sample.columns)
        colors = np.repeat(gene_colors, df_true_sample.shape[0])

        ax.scatter(xx, yy, c=colors, s=spot_size, alpha=0.5)

    else:
        ax.scatter(xx, yy, s=spot_size, alpha=0.5)

    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='y = x')

    #min_val = min(xx.min(), yy.min())
    #max_val = max(xx.max(), yy.max())
    #ax.set_xlim(min_val, 400)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    #ax.set_ylim(min_val, 400)

    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()

def visualize_subgraph(data):
    # Convert to NetworkX graph with node attributes
    G = to_networkx(data, node_attrs=['x', 'spatial_coords', 'center_cell'])

    # Extract spatial coordinates for positioning nodes
    pos = {i: data.spatial_coords[i].cpu().numpy() for i in range(data.num_nodes)}

    # Extract the center cell index
    center_cell_idx = data.center_cell.nonzero(as_tuple=True)[0].item()

    # Define node colors
    node_colors = ['red' if i == center_cell_idx else 'skyblue' for i in range(data.num_nodes)]

    # Draw the graph
    plt.figure(dpi=300)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors, font_size=10, font_color='black', edge_color='gray')

    # Show the plot
    plt.show()
    
def visualize_attention_graph(edge_index, attn_weights):
    G = nx.Graph()
    
    # Average attention weights across all heads
    attn_weights_avg = np.mean(attn_weights, axis=1)
    
    # Add edges with averaged attention weights as edge attributes
    for i, (src, dst) in enumerate(edge_index.T):
        G.add_edge(src.item(), dst.item(), weight=attn_weights_avg[i])
    
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color=weights, edge_cmap=plt.cm.Blues, width=2)
    plt.show()



def compare_gene_expression_all(original_data, generated_data, gene_indices, gene_names, mean_bool = False):
    plt.figure(figsize=(6.9,3.5),dpi=800)

    if mean_bool: 
        original_flattened = original_data[:, gene_indices].mean(axis=0).flatten()
        generated_flattened = generated_data[:, gene_indices].mean(axis=0).flatten()
        
    else:
    # Flatten the data across all selected genes
        original_flattened = original_data[:, gene_indices].flatten()
        generated_flattened = generated_data[:, gene_indices].flatten()

    # Plot the distributions
    sns.kdeplot(original_flattened, label='Ground truth', shade=True)
    sns.kdeplot(generated_flattened, label='Generated', shade=True)

    plt.title('Mean expression distribution for all genes')
    plt.legend()
    plt.xlabel('Expression levels')
    plt.ylabel('Density')
    plt.show()

def compare_gene_expression(original_data, generated_data, gene_indices, gene_names):
    fig, axes = plt.subplots(len(gene_indices), 1, figsize=(10, 5 * len(gene_indices)))
    for i, (gene_idx, gene_name) in enumerate(zip(gene_indices, gene_names)):
        sns.kdeplot(original_data[:, gene_idx], label='Original', ax=axes[i])
        sns.kdeplot(generated_data[:, gene_idx], label='Generated', ax=axes[i])
        axes[i].set_title(f'Expression distribution for {gene_name}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()