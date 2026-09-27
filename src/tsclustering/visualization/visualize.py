"""
Visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Categorical slots from a colorblind-validated palette: blue, orange, aqua, yellow.
# Slots 1-3 stay distinguishable even when every pair is on screen; use slot 4
# only where panels or labels also carry identity.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED, MEMBER_GREY, SURFACE = "#0b0b0b", "#52514e", "#b8b7b1", "#fcfcfb"


def set_style() -> None:
    """Apply the shared figure style: light surface, recessive grid, thin lines."""
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#d9d8d3",
            "grid.color": "#ebeae6",
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "lines.linewidth": 2,
        },
    )


def plot_distribution(
    data: pd.Series | np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 30,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the distribution of a numerical variable.

    Parameters
    ----------
    data : pandas.Series or numpy.ndarray
        Input data.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    bins : int, optional
        Number of histogram bins (default is 30).
    figsize : tuple, optional
        Figure size (default is (10, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=data, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a correlation matrix heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    title : str, optional
        Plot title (default is 'Correlation Matrix').
    figsize : tuple, optional
        Figure size (default is (12, 8)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True
    )
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_time_series(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    title: str,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    time_column : str
        Name of the time column.
    value_column : str
        Name of the value column.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size (default is (12, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    plt.plot(df[time_column], df[value_column])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Box Plots",
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot box plots for multiple columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to plot.
    title : str, optional
        Plot title (default is 'Box Plots').
    figsize : tuple, optional
        Figure size (default is (12, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    df[columns].boxplot()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Scatter Matrix",
    figsize: tuple[int, int] = (12, 12),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a scatter matrix for multiple columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to plot.
    title : str, optional
        Plot title (default is 'Scatter Matrix').
    figsize : tuple, optional
        Figure size (default is (12, 12)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    pd.plotting.scatter_matrix(df[columns], diagonal="kde", figsize=figsize)
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray | None = None,
    title: str | None = None,
    color: str = PALETTE[0],
    axes: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot one panel per cluster: members in grey, center in `color`.

    Parameters
    ----------
    X : numpy.ndarray
        Series of shape `(n_series, length)` or `(n_series, length, 1)`.
    labels : numpy.ndarray
        Cluster assignment per series.
    centers : numpy.ndarray, optional
        One center (barycenter/centroid) per cluster, e.g. `model.cluster_centers_`.
    title : str, optional
        Figure title (only used when this function creates the figure).
    color : str, optional
        Color of the center line (default is the first palette slot).
    axes : numpy.ndarray of matplotlib Axes, optional
        One axis per cluster, to draw into an existing figure (e.g. one row of a
        grid). When None, a new figure is created and shown.
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).
    """
    clusters = np.unique(labels)
    fig = None
    if axes is None:
        fig, grid = plt.subplots(
            1,
            len(clusters),
            figsize=(3 * len(clusters), 2.6),
            sharey=True,
            squeeze=False,
        )
        axes = grid[0]
    for ax, k in zip(axes, clusters, strict=True):
        for series in X[labels == k]:
            ax.plot(series.ravel(), color=MEMBER_GREY, alpha=0.35, lw=0.8)
        if centers is not None:
            ax.plot(centers[k].ravel(), color=color, lw=2)
        ax.set_title(f"Cluster {k} (n={np.sum(labels == k)})", fontsize=10)
    if fig is not None:
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
