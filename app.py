"""
app.py 
SolaraViz interactive dashboard
Run locally with: solara run app.py

Shows agents moving in real time across the 20×20 problem space.
Problems are colored by definition level (red = undefined → green = solved).
Agents are colored by type.

Requires: pip install solara mesa matplotlib
"""

import solara
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mesa.visualization import SolaraViz, make_plot_component

from simulation import GenerativeOrgModel, GRID_SIZE

# ── Color scheme (consistent with simulation.py) ──────────────
AGENT_COLORS = {
    "crystallizer": "#E07B54",
    "propagator":   "#5B8DB8",
    "diffuser":     "#7BAF7A",
}
AGENT_MARKERS = {
    "crystallizer": "o",
    "propagator":   "^",
    "diffuser":     "D",
}

# ── Custom space component ────────────────────────────────────
# Mesa's built-in SpaceRenderer works with discrete grids.
# Our simulation uses a continuous 2D space, so we draw it
# with a custom matplotlib component.

@solara.component
def SpacePlot(model):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Problems, colored by definition level (red → green)
    for p in model.problems:
        if not p.solved:
            rgba = cm.RdYlGn(p.definition_level)
            ax.scatter(p.x, p.y, color=[rgba], s=180, alpha=0.75,
                       marker="s", zorder=1)

    # Agents — colored and shaped by type
    for agent in model.agents:
        color  = AGENT_COLORS.get(agent.agent_type, "#999")
        marker = AGENT_MARKERS.get(agent.agent_type, "o")
        ax.scatter(agent.x, agent.y, c=color, s=90, zorder=2,
                   marker=marker, edgecolors="white", linewidth=0.8)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_title(f"Step {model._step_count} / 100", fontsize=11)
    ax.set_xlabel("Space", fontsize=9)
    ax.set_ylabel("Space", fontsize=9)
    ax.grid(True, alpha=0.1)

    # Agent legend
    agent_handles = [
        mpatches.Patch(color=AGENT_COLORS[t], label=t.capitalize())
        for t in ["crystallizer", "propagator", "diffuser"]
    ]
    # Problem color scale note
    problem_note = mpatches.Patch(
        facecolor="white", edgecolor="gray", linestyle="--",
        label="■ problems: red=formless → green=solved"
    )
    ax.legend(handles=agent_handles + [problem_note],
              loc="upper right", fontsize=7, framealpha=0.9)

    solara.FigureMatplotlib(fig)
    plt.close(fig)


# ── Chart components ─────────────────────────────────────────
ContributionPlot = make_plot_component({
    "Contribution_Crystallizer": "#E07B54",
    "Contribution_Propagator":   "#5B8DB8",
    "Contribution_Diffuser":     "#7BAF7A",
})

MetricsPlot = make_plot_component({
    "Avg_Problem_Definition": "#5B8DB8",
})

BacklogPlot = make_plot_component({
    "Formless_Backlog": "#E07B54",
    "Solved_Count":     "#7BAF7A",
})


# ── Model parameters (UI controls) ───────────────────────────
model_params = {
    "sim_type": {
        "type":   "Select",
        "value":  "balanced",
        "values": ["crystallizer_heavy", "balanced", "propagator_heavy"],
        "label":  "Team Composition",
    },
}

# ── SolaraViz app ─────────────────────────────────────────────
page = SolaraViz(
    GenerativeOrgModel,
    components=[SpacePlot, ContributionPlot, MetricsPlot, BacklogPlot],
    model_params=model_params,
    name="Simulating People-Shapes in Organizations | Animacy",
)

page
