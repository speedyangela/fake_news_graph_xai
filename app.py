"""Démo Streamlit : graphes fake, pas le vrai pipeline UPFD."""

from __future__ import annotations

import time
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
from matplotlib import colors as mcolors

SIM_ORGANIC = "Propagation Organique (Vraie info)"
SIM_BOTS = "Attaque Coordonnée (Bots)"


def simulate_organic_graph() -> nx.Graph:
    """Étoile : un gros hub, beaucoup de feuilles."""
    return nx.star_graph(60)


def simulate_bot_coordinated_graph() -> nx.Graph:
    """Plusieurs hubs reliés + satellites (look “coordination”)."""
    G = nx.Graph()
    hubs = [0, 1, 2, 3]
    G.add_nodes_from(hubs)
    for u in hubs:
        for v in hubs:
            if u < v:
                G.add_edge(u, v)
    next_id = 4
    leaves_per_hub = 17
    for h in hubs:
        for _ in range(leaves_per_hub):
            G.add_edge(h, next_id)
            next_id += 1
    return G


def generate_graph(simulation_label: str) -> nx.Graph:
    if simulation_label == SIM_ORGANIC:
        return simulate_organic_graph()
    return simulate_bot_coordinated_graph()


def layout_positions(G: nx.Graph, seed: int = 42) -> dict:
    return nx.spring_layout(
        G,
        seed=seed,
        k=2 / np.sqrt(max(G.number_of_nodes(), 1)),
    )


def degree_to_size_and_color(
    G: nx.Graph,
    deg_max: float,
    size_min: float = 120.0,
    size_max: float = 2200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    node_list = list(G.nodes())
    degs = np.array([G.degree(n) for n in node_list], dtype=np.float64)
    denom = max(float(deg_max), 1e-6)
    alpha = degs / denom
    sizes = size_min + (size_max - size_min) * alpha
    return sizes, degs


def draw_xai_graph(
    G: nx.Graph,
    title: str,
    cmap_name: str = "viridis",
    figsize: Tuple[float, float] = (10, 7),
) -> plt.Figure:
    """Graphe matplotlib : taille + couleur selon le degré."""
    pos = layout_positions(G, seed=42)
    deg_max = max(dict(G.degree()).values()) if G.number_of_nodes() else 1.0
    node_list = list(G.nodes())
    sizes, degs = degree_to_size_and_color(G, float(deg_max))

    norm = mcolors.Normalize(vmin=0.0, vmax=float(deg_max))
    cmap = plt.get_cmap(cmap_name)
    node_colors = cmap(norm(degs))

    fig, ax = plt.subplots(figsize=figsize, facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        width=1.0,
        alpha=0.55,
        arrows=False,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=node_list,
        node_size=sizes,
        node_color=node_colors,
        linewidths=0.6,
        edgecolors="#1e293b",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a", pad=16)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Degré (centralité locale)", rotation=270, labelpad=20, fontsize=10)

    fig.tight_layout()
    return fig


def run_gnn_progress_placeholder() -> None:
    """Barre de progression décorative."""
    progress = st.progress(0, text="Initialisation du modèle…")
    stages = [
        "Lecture du graphe…",
        "Propagation des messages (GCN)…",
        "Agrégation globale (pooling)…",
        "Tête de classification…",
    ]
    for i in range(10):
        progress.progress(
            (i + 1) / 10,
            text=stages[min(i // 3, len(stages) - 1)],
        )
        time.sleep(0.06)
    progress.empty()


def main() -> None:
    st.set_page_config(
        page_title="A.R.I.A.N.E",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1100px; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# A.R.I.A.N.E")
    st.markdown(
        '<p style="font-size:1.2rem;color:#64748b;margin-top:-0.5rem;margin-bottom:1.2rem;">'
        "XAI (Fake News Detector)</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Cette démo illustre comment un **Graph Neural Network** peut juger la **géométrie**
        de la diffusion (**organique**, une source qui rayonne, ou **artificielle**, réseau
        d’amplificateurs / astroturfing), **sans lire le texte** de l’article.
        Les features sémantiques (ex. BERT) peuvent se combiner à cette structure dans
        le modèle complet ; ici, l’accent est mis sur la **forme du réseau**.
        """
    )
    st.info(
        "**Démo pédagogique (données mockées)** : les graphes sont générés pour l’exemple, "
        "la barre de progression et les pourcentages de confiance sont **simulés**, et aucun "
        "modèle entraîné sur UPFD n’est exécuté ici. Le code du projet (chargement UPFD, GCN, "
        "métriques réelles) vit dans les scripts Python du dépôt."
    )

    st.sidebar.header("Simulation")
    simulation = st.sidebar.radio(
        "Scénario de propagation",
        options=[SIM_ORGANIC, SIM_BOTS],
        index=0,
        help="Graphes NetworkX générés pour la démo (pas des extraits du corpus UPFD).",
    )

    analyze = st.sidebar.button(
        "Lancer l'analyse du réseau",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        run_gnn_progress_placeholder()
        G = generate_graph(simulation)
        max_deg = max(dict(G.degree()).values()) if G.number_of_nodes() else 0

        if simulation == SIM_ORGANIC:
            verdict = "Organique"
            confidence = 94
            artificial = False
            cmap = "YlGnBu"
            fig_title = "Réseau simulé (propagation organique, étoile)"
        else:
            verdict = "Artificiel (attaque coordonnée)"
            confidence = 91
            artificial = True
            cmap = "YlOrRd"
            fig_title = "Réseau simulé (multi-hubs, amplification coordonnée)"

        st.session_state["result"] = {
            "G": G,
            "verdict": verdict,
            "confidence": confidence,
            "max_deg": max_deg,
            "artificial": artificial,
            "cmap": cmap,
            "fig_title": fig_title,
            "simulation": simulation,
        }

    res = st.session_state.get("result")
    if res is None:
        st.info("Choisissez un scénario dans la barre latérale, puis lancez l’analyse.")
        return

    st.caption(f"Scénario analysé : **{res['simulation']}**")
    st.subheader("Résultats de l’analyse")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Verdict", res["verdict"])
    with c2:
        st.metric("Confiance du modèle", f"{res['confidence']} %")
    with c3:
        st.metric("Degré maximum (Δ_max)", res["max_deg"])

    if res["artificial"]:
        st.error(
            "Structure compatible avec une diffusion coordonnée : plusieurs hubs denses "
            "et interconnectés (signature souvent associée à l’amplification artificielle). "
            "Démo pédagogique (pas une preuve juridique)."
        )
    else:
        st.success(
            "Structure compatible avec une diffusion organique : un hub principal dominant, "
            "périphérie peu connectée entre elle (schéma type étoile)."
        )

    st.subheader("Explication visuelle (XAI)")
    st.caption(
        "Taille et couleur des nœuds = **degré** (même logique que `visualize.py`). "
        "Les **hubs** ressortent immédiatement à l’œil."
    )

    fig = draw_xai_graph(res["G"], title=res["fig_title"], cmap_name=res["cmap"])
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
