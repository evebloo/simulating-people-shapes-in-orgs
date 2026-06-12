"""
app_hf.py
Gradio app for HuggingFace Spaces
Embeddable in Substack via iframe.

Deploy to HuggingFace Spaces:
  1. Create a new Space at huggingface.co/spaces
  2. Choose "Gradio" as the SDK
  3. Upload this file as app.py, plus simulation.py and requirements_hf.txt

The app runs the simulation and shows:
  - The 2D space with agents and problems
  - Contribution curves per agent type
  - Formless backlog and avg problem definition over time
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from simulation import GenerativeOrgModel, GRID_SIZE

# ── Colors ────────────────────────────────────────────────────
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
SCENARIO_LABELS = {
    "crystallizer_heavy": "Crystallizer-Heavy (8 cryst · 4 prop · 3 diff)",
    "balanced":           "Balanced (5 · 5 · 5)",
    "propagator_heavy":   "Propagator-Heavy (3 cryst · 8 prop · 4 diff)",
}


def run_simulation(scenario: str, seed: int = 42) -> tuple:
    """
    Run the full simulation and return figures.
    Returns: (space_fig, metrics_fig)
    """
    model = GenerativeOrgModel(sim_type=scenario, rng=seed)
    history = []

    for _ in range(100):
        model.step()
        row = {
            "step": model._step_count,
            "avg_def":  model._avg_problem_definition(),
            "backlog":  model._formless_backlog(),
            "solved":   model.solved_count,
            "cryst":    model._avg_contribution("crystallizer"),
            "prop":     model._avg_contribution("propagator"),
            "diff":     model._avg_contribution("diffuser"),
        }
        history.append(row)

    # ── Space plot ────────────────────────────────────────────
    fig_space, ax_s = plt.subplots(figsize=(5, 5))

    for p in model.problems:
        if not p.solved:
            rgba = cm.RdYlGn(p.definition_level)
            ax_s.scatter(p.x, p.y, color=[rgba], s=180, alpha=0.75,
                         marker="s", zorder=1)

    for agent in model.agents:
        color  = AGENT_COLORS[agent.agent_type]
        marker = AGENT_MARKERS[agent.agent_type]
        ax_s.scatter(agent.x, agent.y, c=color, s=90, zorder=2,
                     marker=marker, edgecolors="white", linewidth=0.8)

    ax_s.set_xlim(0, GRID_SIZE)
    ax_s.set_ylim(0, GRID_SIZE)
    ax_s.set_aspect("equal")
    ax_s.set_title(f"Final state — {SCENARIO_LABELS[scenario]}", fontsize=9)
    ax_s.grid(True, alpha=0.1)

    handles = [mpatches.Patch(color=AGENT_COLORS[t], label=t.capitalize())
               for t in ["crystallizer", "propagator", "diffuser"]]
    ax_s.legend(handles=handles, fontsize=7, loc="upper right")
    plt.tight_layout()

    # ── Metrics plots ─────────────────────────────────────────
    fig_m, axes = plt.subplots(1, 3, figsize=(14, 4))
    steps = [r["step"] for r in history]

    # Contribution per type
    axes[0].plot(steps, [r["cryst"] for r in history],
                 color="#E07B54", linewidth=2, label="Crystallizer")
    axes[0].plot(steps, [r["prop"]  for r in history],
                 color="#5B8DB8", linewidth=2, label="Propagator")
    axes[0].plot(steps, [r["diff"]  for r in history],
                 color="#7BAF7A", linewidth=2, label="Diffuser")
    axes[0].set_title("Avg Cumulative Contribution", fontsize=10)
    axes[0].set_xlabel("Time Steps")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Avg problem definition
    axes[1].plot(steps, [r["avg_def"] for r in history],
                 color="#5B8DB8", linewidth=2)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Avg Problem Definition Level", fontsize=10)
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("0 = formless → 1 = solved")
    axes[1].grid(True, alpha=0.2)

    # Formless backlog + solved
    ax2 = axes[2].twinx()
    axes[2].plot(steps, [r["backlog"] for r in history],
                 color="#E07B54", linewidth=2, label="Formless backlog")
    ax2.plot(steps, [r["solved"] for r in history],
             color="#7BAF7A", linewidth=2, linestyle="--", label="Solved count")
    axes[2].set_title("Formless Backlog & Throughput", fontsize=10)
    axes[2].set_xlabel("Time Steps")
    axes[2].set_ylabel("Untouched undefined problems", color="#E07B54")
    ax2.set_ylabel("Problems solved", color="#7BAF7A")
    axes[2].grid(True, alpha=0.2)
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig_m.suptitle(
        f"Simulating People-Shapes in Organizations — {SCENARIO_LABELS[scenario]}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    return fig_space, fig_m


# ── Gradio interface ──────────────────────────────────────────
with gr.Blocks(title="Simulating People-Shapes in Organizations") as demo:

    gr.Markdown("""
# Simulating People-Shapes in Organizations
### *Animacy | Eve-Marie Blouin-Hudon*

An agent-based simulation of three organizational shapes from the essay
[Built for Ambiguity](https://www.eveblou.com/p/built-for-ambiguity-are-these-the).

**Three agent types** move through a space of 30 problems, each with a definition level (0 = formless → 1 = defined/solved).
- 🔶 **Crystallizer** seeks undefined problems and gives them form
- 🔵 **Propagator** finds stuck agents and reframes them; works on ambiguous problems
- 🟢 **Diffuser** finds drifting agents and restores coordination across many neighboring agents

**Three compositions** tested across 100 time steps.
""")

    with gr.Row():
        scenario_input = gr.Dropdown(
            choices=list(SCENARIO_LABELS.keys()),
            value="balanced",
            label="Team Composition",
        )
        seed_input = gr.Slider(
            minimum=0, maximum=99, value=42, step=1,
            label="Random Seed (change to see variation)"
        )
        run_btn = gr.Button("▶ Run Simulation", variant="primary")

    with gr.Row():
        space_output   = gr.Plot(label="Final Agent & Problem State")
        metrics_output = gr.Plot(label="Metrics over 100 Steps")

    run_btn.click(
        fn=run_simulation,
        inputs=[scenario_input, seed_input],
        outputs=[space_output, metrics_output],
    )

    gr.Markdown("""
---
Built with [Mesa](https://github.com/projectmesa/mesa) (Python agent-based modeling) and [Gradio](https://gradio.app).
Code on GitHub: [simulating-people-shapes-in-orgs](https://github.com/evebloo/simulating-people-shapes-in-orgs)
Read the essay series at [eveblou.substack.com](https://eveblou.substack.com)
""")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
