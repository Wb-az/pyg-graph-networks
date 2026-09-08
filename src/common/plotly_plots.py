from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.express as px



def plot_2d_projections(features, color, label, method="tsne", random_state=0):
    """Reduce `features` to 2D with t-SNE or UMAP and render an interactive scatter."""
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, n_jobs=1, random_state=random_state)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'tsne' or 'umap'.")

    projections = reducer.fit_transform(features)

    fig = px.scatter(
        projections, x=0, y=1,
        color=color, labels={'color': label}
    )
    fig.show()
    return projections


def plot_3d_projection(features, color, method="tsne", pca_components=None, title=None, random_state=123,
                       label_name="Publication type"):
    """Reduce `features` to 3D with t-SNE or UMAP, optionally PCA-denoising first, and render an interactive scatter."""
    x = features
    if pca_components is not None:
        x = PCA(n_components=pca_components, random_state=random_state).fit_transform(x)

    if method == "tsne":
        reducer = TSNE(n_components=3, random_state=random_state)
    elif method == "umap":
        reducer = umap.UMAP(n_components=3, n_jobs =1, random_state=random_state)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'tsne' or 'umap'.")

    projections = reducer.fit_transform(x)

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=color, labels={'color': label_name},
        title=title,
    )
    fig.update_traces(marker_size=6)
    fig.show()

    return projections