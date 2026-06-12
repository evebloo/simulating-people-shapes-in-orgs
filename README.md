# simulating-people-shapes-in-orgs

Agent-based simulation for *Thinkering Post #2* of [Animacy](https://eveblou.substack.com) by Eve-Marie Blouin-Hudon.

Built alongside the essay *"Built for Ambiguity: Are These the People the AI Era Needs?"*

Based on Epstein & Axtell (1996): *Growing Artificial Societies.*

---

## What This Is

A proper generative social science simulation. No pre-defined environments. Three types of agents (crystallizers, propagators, diffusers) move through a space of 30 problems with evolving definition levels. Organizational patterns emerge from their interactions.

**Three compositions tested:**
- Crystallizer-heavy (8 · 4 · 3)
- Balanced (5 · 5 · 5)
- Propagator-heavy (3 · 8 · 4)

---

## Two Ways to Run It

### 1. Batch simulation (charts)

```bash
pip install -r requirements.txt
python simulation.py
```

Runs 3 scenarios × 10 seeds × 100 steps. Saves charts to `output/simulation_results.png`.

### 2. Interactive dashboard (SolaraViz)

```bash
pip install -r requirements.txt
solara run app.py
```

Opens a live browser dashboard with the 2D agent space, real-time contribution charts, and a dropdown to switch team compositions.

---

## HuggingFace Spaces (embeddable)

`app_hf.py` is a Gradio version deployable to HuggingFace Spaces. Use `requirements_hf.txt`.

---

## Agent Types

| Type | Behavior |
|------|----------|
| Crystallizer | Seeks undefined problems (< 0.4). Raises definition by 0.25. |
| Propagator | Finds stuck agents first. Then works on ambiguous problems (0.3–0.7). |
| Diffuser | Finds drifting agents. Restores coordination. |

---

## Key Emergent Metrics

- **Formless backlog** count of undefined problems sitting untouched > 5 steps
- **Avg problem definition** is the problem space transforming?
- **Solved count** throughput

---

## Related

- [Animacy Substack](https://eveblou.substack.com)
- Essay 3: [Built for Ambiguity](https://www.eveblou.com/p/built-for-ambiguity-are-these-the)
