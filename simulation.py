"""
animacy_generative_simulation.py — v3

Proper generative social science simulation.
Based on Epstein & Axtell (1996): Growing Artificial Societies.

DESIGN PIVOT from v1 → v3:

v1 had:
  - Pre-defined environments (hierarchical/transitioning)
  - Impact compounding multiplier (Wright learning curve)
  - Stagnation erosion, toil mechanics
  - I-shape and T-shape agents
  This was a comparative experiment, not a generative model.
  Environments were inputs, not outputs.

v2 introduced proper generative mechanics (problems as objects
in 2D space, agents move, organizational patterns emerge) but
carried over the compounding machinery from v1.

v3 simplifies to the right level:
  - Three agent types only: crystallizer, propagator, diffuser
  - Contribution = simple count of successful interactions
    (Sugarscape equivalent: wealth = sugar accumulated)
  - No compounding multiplier, no toil
    (Sugarscape had metabolism as baseline drain; we removed
    that too — agents don't die, they just contribute less)
  - Organizational patterns emerge purely from movement
    and interaction rules

THREE SCENARIOS compared:
  1. Crystallizer-heavy  (8 crystallizers, 4 propagators, 3 diffusers)
  2. Balanced            (5 of each)
  3. Propagator-heavy    (3 crystallizers, 8 propagators, 4 diffusers)

KEY EMERGENT METRICS:
  - formless_backlog: % of undefined problems sitting untouched > 5 steps
  - avg_problem_definition: is the problem space transforming?
  - solved_count: total throughput

Usage:
    python simulation.py
"""

import random
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

GRID_SIZE          = 20
N_PROBLEMS         = 30
STEPS              = 100
INTERACTION_RADIUS = 2.0
N_SEEDS            = 10


# ─────────────────────────────────────────────────────────────
# PROBLEM
# ─────────────────────────────────────────────────────────────

class Problem:
    """
    A problem in the organizational space.

    definition_level: 0.0 = pure formlessness, 1.0 = fully defined/solved.

    Problems are persistent objects. Their definition_level evolves
    through agent interaction. The distribution of definition levels
    across all problems at any given step IS the organizational pattern, 
    it emerges from who is in the system.

    Sugarscape equivalent: sugar patches with varying sugar levels.
    Agents move toward them; interaction depletes/advances them.
    """

    def __init__(self, x, y, definition_level=None):
        self.x = x
        self.y = y
        self.definition_level = (
            definition_level if definition_level is not None
            else random.uniform(0, 1)
        )
        self.steps_without_interaction = 0
        self.solved = False

    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def age(self):
        self.steps_without_interaction += 1


# ─────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────

class OrgAgent(Agent):
    """
    Base class. Agents have positions in 2D space and move toward targets.

    contribution: count of successful interactions.
      Sugarscape equivalent: wealth (sugar accumulated over time).
      Simple, honest. No compounding multiplier, no decay.

    steps_without_progress: how long since the agent last contributed.
      Used by the propagator to find stuck agents.
      Sugarscape equivalent: there wasn't one. This is specific
      to my model's social interaction layer.
    """

    def __init__(self, model, agent_type):
        super().__init__(model)
        self.agent_type = agent_type
        self.x = random.uniform(0, GRID_SIZE)
        self.y = random.uniform(0, GRID_SIZE)
        self.contribution = 0.0      # cumulative successful contributions
        self.steps_without_progress = 0

    def move_toward(self, tx, ty):
        dx, dy = tx - self.x, ty - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist > 0:
            step = min(1.0, dist)
            self.x += (dx / dist) * step
            self.y += (dy / dist) * step
        self.x = max(0, min(GRID_SIZE, self.x))
        self.y = max(0, min(GRID_SIZE, self.y))

    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def nearest_problem(self, condition=None):
        active = [p for p in self.model.problems if not p.solved]
        pool = [p for p in active if condition(p)] if condition else active
        if not pool:
            pool = active
        return min(pool, key=lambda p: self.distance_to(p.x, p.y)) if pool else None

    def nearest_agent(self, condition):
        others = [a for a in self.model.agents if a is not self and condition(a)]
        return min(others, key=lambda a: self.distance_to(a.x, a.y)) if others else None

    def contribute(self, amount):
        """Record a successful interaction."""
        self.contribution += amount
        self.steps_without_progress = 0

    def try_solve(self, problem):
        if problem.definition_level >= 0.95:
            problem.solved = True
            self.model.solved_count += 1

    def step(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# CRYSTALLIZER
# ─────────────────────────────────────────────────────────────

class Crystallizer(OrgAgent):
    """
    Moves toward undefined problems (definition < 0.4).
    Raises their definition level and gives form to formlessness.

    Contribution = amount of definition level raised per interaction.
    Sugarscape equivalent: an agent with vision tuned to low-sugar patches,
    metabolizing them into something usable.

    Emergent effect on formless backlog:
      When crystallizers are present → undefined problems get worked on
      → formless backlog stays low.
      When absent → formless problems pile up untouched
      → formless backlog rises.
    """

    def __init__(self, model):
        super().__init__(model, 'crystallizer')

    def step(self):
        target = self.nearest_problem(lambda p: p.definition_level < 0.4)
        if not target:
            target = self.nearest_problem()
        if not target:
            self.steps_without_progress += 1
            return

        if self.distance_to(target.x, target.y) <= INTERACTION_RADIUS:
            delta = 0.25
            target.definition_level = min(1.0, target.definition_level + delta)
            target.steps_without_interaction = 0
            self.contribute(1.0)
            self.model.team_momentum = min(5.0, self.model.team_momentum + 0.2)
            self.try_solve(target)
        else:
            self.move_toward(target.x, target.y)
            self.steps_without_progress += 1


# ─────────────────────────────────────────────────────────────
# PROPAGATOR
# ─────────────────────────────────────────────────────────────

class Propagator(OrgAgent):
    """
    First priority: finds stuck agents (no progress > 5 steps).
    Second: works on ambiguous problems (0.3–0.7 definition).

    Domain-general: not drawn to any specific definition range exclusively.

    Contribution = 1.0 per stuck agent reframed, or delta raised on problems.
    Sugarscape equivalent: an agent with broader vision, able to see
    across the landscape for blockages rather than just resources.
    """

    def __init__(self, model):
        super().__init__(model, 'propagator')

    def step(self):
        stuck = self.nearest_agent(lambda a: a.steps_without_progress > 5)

        if stuck:
            if self.distance_to(stuck.x, stuck.y) <= INTERACTION_RADIUS:
                stuck.steps_without_progress = 0
                self.contribute(1.0)
            else:
                self.move_toward(stuck.x, stuck.y)
                self.steps_without_progress += 1
            return

        target = self.nearest_problem(lambda p: 0.3 < p.definition_level < 0.7)
        if not target:
            target = self.nearest_problem()
        if not target:
            self.steps_without_progress += 1
            return

        if self.distance_to(target.x, target.y) <= INTERACTION_RADIUS:
            delta = 0.15
            target.definition_level = min(1.0, target.definition_level + delta)
            target.steps_without_interaction = 0
            self.contribute(1.0)
            self.try_solve(target)
        else:
            self.move_toward(target.x, target.y)
            self.steps_without_progress += 1


# ─────────────────────────────────────────────────────────────
# DIFFUSER
# ─────────────────────────────────────────────────────────────

class Diffuser(OrgAgent):
    """
    Moves toward agents who are drifting (steps_without_progress > 3).
    Resets their stuck counter and contributes to collective momentum.

    Doesn't directly raise problem definition levels.
    Their contribution is coordination: keeping agents from stalling.

    Contribution = 1.0 per coordination event (per agent helped).
    Sugarscape equivalent: no direct analogue. Sugarscape didn't have
    agents that boosted other agents. This is the social layer specific
    to my organizational simulation.
    """

    def __init__(self, model):
        super().__init__(model, 'diffuser')

    def step(self):
        drifting = self.nearest_agent(lambda a: a.steps_without_progress > 3)

        if drifting:
            if self.distance_to(drifting.x, drifting.y) <= INTERACTION_RADIUS:
                # Organize nearby agents
                nearby = [
                    a for a in self.model.agents
                    if self.distance_to(a.x, a.y) <= INTERACTION_RADIUS * 1.5
                    and a is not self
                ]
                helped = 0
                for a in nearby[:4]:
                    if a.steps_without_progress > 0:
                        a.steps_without_progress = max(0, a.steps_without_progress - 3)
                        helped += 1
                if helped:
                    self.contribute(1.0)
                else:
                    self.steps_without_progress += 1
            else:
                self.move_toward(drifting.x, drifting.y)
                self.steps_without_progress += 1
        else:
            # No one drifting — ambient presence
            self.contribute(0.3)


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

class GenerativeOrgModel(Model):
    """
    No pre-defined environments. Three scenarios test different
    compositions of crystallizers, propagators, and diffusers.

    Scenarios:
      'crystallizer_heavy' : 8 crystallizers + 4 propagators + 3 diffusers
      'balanced'           : 5 of each
      'propagator_heavy'   : 3 crystallizers + 8 propagators + 4 diffusers
    """

    def __init__(self, sim_type='balanced', rng=42):
        super().__init__(rng=rng)
        self.sim_type = sim_type
        self.team_momentum = 0.0
        self.solved_count = 0
        self._step_count = 0

        self.problems = [
            Problem(
                x=random.uniform(0, GRID_SIZE),
                y=random.uniform(0, GRID_SIZE),
            )
            for _ in range(N_PROBLEMS)
        ]

        if sim_type == 'crystallizer_heavy':
            for _ in range(8): Crystallizer(self)
            for _ in range(4): Propagator(self)
            for _ in range(3): Diffuser(self)
        elif sim_type == 'propagator_heavy':
            for _ in range(3): Crystallizer(self)
            for _ in range(8): Propagator(self)
            for _ in range(4): Diffuser(self)
        else:  # balanced
            for _ in range(5): Crystallizer(self)
            for _ in range(5): Propagator(self)
            for _ in range(5): Diffuser(self)

        self.datacollector = DataCollector(
            model_reporters={
                "Formless_Backlog":        lambda m: m._formless_backlog(),
                "Avg_Problem_Definition":  lambda m: m._avg_problem_definition(),
                "Solved_Count":            lambda m: m.solved_count,
                "Contribution_Crystallizer": lambda m: m._avg_contribution('crystallizer'),
                "Contribution_Propagator":   lambda m: m._avg_contribution('propagator'),
                "Contribution_Diffuser":     lambda m: m._avg_contribution('diffuser'),
            }
        )

    def _avg_contribution(self, t):
        ag = [a for a in self.agents if a.agent_type == t]
        return sum(a.contribution for a in ag) / len(ag) if ag else 0

    def _avg_problem_definition(self):
        active = [p for p in self.problems if not p.solved]
        return sum(p.definition_level for p in active) / len(active) if active else 1.0

    def _formless_backlog(self):
        """
        Absolute count of undefined problems (definition < 0.3)
        sitting untouched for more than 5 steps.

        How many formless problems are piling up
        that nobody is picking up? 
        
        Using absolute count rather than percentage to avoid
        statistical noise when the pool of undefined problems is small.
        """
        return len([
            p for p in self.problems
            if not p.solved
            and p.definition_level < 0.3
            and p.steps_without_interaction > 5
        ])

    def _respawn_solved(self):
        for p in self.problems:
            if p.solved:
                p.x = random.uniform(0, GRID_SIZE)
                p.y = random.uniform(0, GRID_SIZE)
                p.definition_level = random.uniform(0.0, 1.0)  # matches initial setup
                p.steps_without_interaction = 0
                p.solved = False

    def step(self):
        self.datacollector.collect(self)
        for p in self.problems:
            p.age()
        self.team_momentum *= 0.9
        self.agents.shuffle_do('step')
        self._respawn_solved()
        self._step_count += 1


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

def run_multi_seed(sim_type, steps=STEPS, n_seeds=N_SEEDS):
    import pandas as pd
    all_data = []
    for seed in range(n_seeds):
        model = GenerativeOrgModel(sim_type=sim_type, rng=seed)
        for _ in range(steps):
            model.step()
        all_data.append(model.datacollector.get_model_vars_dataframe())
    stacked = pd.concat(all_data)
    return (
        stacked.groupby(stacked.index).mean(),
        stacked.groupby(stacked.index).std()
    )


def print_summary(mean, std, sim_type):
    print(f"\n{'='*55}")
    print(f"  {sim_type.upper().replace('_',' ')}  |  {STEPS} steps  |  {N_SEEDS} seeds")
    print(f"{'='*55}")
    f, fs = mean.iloc[-1], std.iloc[-1]
    print(f"  {'Formless backlog':25s}: {f['Formless_Backlog']:.2f}  (±{fs['Formless_Backlog']:.2f})")
    print(f"  {'Avg problem definition':25s}: {f['Avg_Problem_Definition']:.2f}  (±{fs['Avg_Problem_Definition']:.2f})")
    print(f"  {'Problems solved':25s}: {f['Solved_Count']:.0f}")
    for col, name in [
        ("Contribution_Crystallizer", "Crystallizer"),
        ("Contribution_Propagator",   "Propagator"),
        ("Contribution_Diffuser",     "Diffuser"),
    ]:
        print(f"  {name:25s}: {f[col]:.1f}  (±{fs[col]:.2f})")


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────

def plot_results(results):
    COLORS = {
        'crystallizer': '#E07B54',
        'propagator':   '#5B8DB8',
        'diffuser':     '#7BAF7A',
    }
    SCENE_COLORS = ['#E07B54', '#5B8DB8', '#7BAF7A']
    SCENE_LABELS = ['Crystallizer-heavy', 'Balanced', 'Propagator-heavy']

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "Generative Organization Simulation  |  10 seeds × 100 steps\n"
        "Three compositions of crystallizers, propagators, and diffusers",
        fontsize=12, fontweight='bold'
    )

    steps_x = range(STEPS)

    CONTRIB_PAIRS = [
        ('Contribution_Crystallizer', 'crystallizer'),
        ('Contribution_Propagator',   'propagator'),
        ('Contribution_Diffuser',     'diffuser'),
    ]

    scenarios_data = [
        (results['crystallizer_heavy'], 'Crystallizer-Heavy\n(8 cryst · 4 prop · 3 diff)'),
        (results['balanced'],           'Balanced\n(5 of each)'),
        (results['propagator_heavy'],   'Propagator-Heavy\n(3 cryst · 8 prop · 4 diff)'),
    ]

    # Row 1: contribution curves per scenario
    for ax, ((mean, std), title) in zip(axes[0], scenarios_data):
        for col, atype in CONTRIB_PAIRS:
            if mean[col].max() > 0.1:
                ax.plot(steps_x, mean[col], label=atype.capitalize(),
                        color=COLORS[atype], linewidth=2.5)
                ax.fill_between(steps_x,
                    mean[col] - std[col], mean[col] + std[col],
                    color=COLORS[atype], alpha=0.12)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=9)
        ax.set_ylabel('Avg Cumulative Contribution', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Row 2, col 1: Formless backlog
    for (mean, std), color, label in zip(
        [r for r, _ in scenarios_data],
        SCENE_COLORS, SCENE_LABELS
    ):
        axes[1,0].plot(steps_x, mean['Formless_Backlog'],
                       label=label, color=color, linewidth=2.5)
        axes[1,0].fill_between(steps_x,
            mean['Formless_Backlog'] - std['Formless_Backlog'],
            mean['Formless_Backlog'] + std['Formless_Backlog'],
            color=color, alpha=0.12)
    axes[1,0].set_title(
        'Formless Backlog\n(count of undefined problems sitting untouched > 5 steps)',
        fontsize=10, fontweight='bold'
    )
    axes[1,0].set_xlabel('Time Steps', fontsize=9)
    axes[1,0].set_ylabel('Number of untouched undefined problems', fontsize=9)
    axes[1,0].legend(fontsize=9)
    axes[1,0].grid(True, alpha=0.2)

    # Row 2, col 2: Avg problem definition
    for (mean, std), color, label in zip(
        [r for r, _ in scenarios_data],
        SCENE_COLORS, SCENE_LABELS
    ):
        axes[1,1].plot(steps_x, mean['Avg_Problem_Definition'],
                       label=label, color=color, linewidth=2.5)
        axes[1,1].fill_between(steps_x,
            mean['Avg_Problem_Definition'] - std['Avg_Problem_Definition'],
            mean['Avg_Problem_Definition'] + std['Avg_Problem_Definition'],
            color=color, alpha=0.12)
    axes[1,1].set_title(
        'Avg Problem Definition Level\n(is the problem space transforming?)',
        fontsize=10, fontweight='bold'
    )
    axes[1,1].set_xlabel('Time Steps', fontsize=9)
    axes[1,1].set_ylabel('Definition (0=formless, 1=solved)', fontsize=9)
    axes[1,1].legend(fontsize=9)
    axes[1,1].grid(True, alpha=0.2)

    # Row 2, col 3: Solved count
    for (mean, std), color, label in zip(
        [r for r, _ in scenarios_data],
        SCENE_COLORS, SCENE_LABELS
    ):
        axes[1,2].plot(steps_x, mean['Solved_Count'],
                       label=label, color=color, linewidth=2.5)
    axes[1,2].set_title(
        'Problems Solved\n(total throughput)',
        fontsize=10, fontweight='bold'
    )
    axes[1,2].set_xlabel('Time Steps', fontsize=9)
    axes[1,2].set_ylabel('Cumulative problems solved', fontsize=9)
    axes[1,2].legend(fontsize=9)
    axes[1,2].grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    out = 'output/simulation_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nRunning generative simulation — 3 compositions × {N_SEEDS} seeds × {STEPS} steps\n")

    results = {}
    for sim_type in ['crystallizer_heavy', 'balanced', 'propagator_heavy']:
        print(f"Running {sim_type}...")
        results[sim_type] = run_multi_seed(sim_type)

    for sim_type in ['crystallizer_heavy', 'balanced', 'propagator_heavy']:
        print_summary(*results[sim_type], sim_type)

    plot_results({
        k: (mean, std) for k, (mean, std) in results.items()
    })
