# External Cross-Check: Portfolio Projects for an AI Engineer Profile (June 2026)

Research target: project ideas that get an AI/GenAI engineer hired, cross-checked against
Aravind Pradeep's profile (RAG, GraphRAG, agents, LLM fine-tuning/quantization, knowledge
graphs, FastAPI, PyTorch). Sourced from current hiring/portfolio guides and curated repos.

## What hiring managers actually reward (2026)

- **Deployed live demo > GitHub repo.** A working URL is repeatedly cited as ~10x more
  impressive than source alone. Every project should ship with a live demo link.
- **3–5 complete, deployed projects** beat many half-finished ones. Recommended spread:
  one RAG app, one stateful AI agent, one full-stack GenAI app.
- **GenAI over classical ML.** RAG, agents, and LLM apps are wanted; image classifiers and
  recommender demos read as dated.
- **Production instincts are the real signal:** error handling, evaluation, deployment,
  monitoring, cost tracking, structured READMEs. "Completed the LLM course" portfolios are
  filtered out; production-shaped ones get interviews.
- **Stateful agents that recover from failure**, not one-shot demos.
- **Eval + guardrails is a recognized differentiator** — a named category in 2026 hiring
  (LLM-as-judge, jailbreak/PII/injection scanners, programmable safety rails).
- Delivery: FastAPI + a real UI (Streamlit/Gradio) + Docker beats notebooks every time.

## Mapping to Aravind's edge

- GraphRAG agent already exists and is the lead card → it covers the "agent + RAG" tier IF
  it gets a **live hosted demo** (currently repo-only — biggest gap vs the research).
- The win-condition stack for the warm-outreach startup targets (agentic AI, knowledge
  graphs, eval/guardrails, TypeScript/Next.js) lines up exactly with what the external
  research says is rewarded — so the project bets reinforce the channel pivot.
- Quantization/fine-tuning work (Pruna-style roles) is a real differentiator but needs a
  crisp deployed artifact, not just a training notebook.

## Sources

- [Best GenAI Project Ideas for AI Engineers (2026) — Careery](https://careery.pro/blog/ai-careers/ai-engineer-project-ideas)
- [5 AI Portfolio Projects That Actually Get You Hired in 2026 — DEV](https://dev.to/klement_gunndu/5-ai-portfolio-projects-that-actually-get-you-hired-in-2026-5bpl)
- [Top Generative AI Projects to Build in 2026 — Scaler](https://www.scaler.com/blog/top-generative-ai-projects-to-build-to-get-you-hired/)
- [10 AI Portfolio Projects to Land Your Dream Job (2026) — Scaler](https://www.scaler.com/blog/10-ai-portfolio-projects-to-land-your-dream-job-2026/)
- [AI Engineer Resume Guide 2026 — MirrorCV](https://mirrorcv.com/resume-guide/ai-ml-engineer)
- [ai-engineering-hub — agentic RAG tutorials](https://github.com/patchy631/ai-engineering-hub)
- [awesome-llm-apps — 100+ runnable agent & RAG apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- [agentic-guardrails — layered guardrails](https://github.com/FareedKhan-dev/agentic-guardrails)
- [Best AI Agent Evaluation Tools for Production Teams (2026) — Augment](https://www.augmentcode.com/tools/best-ai-agent-evaluation-tools)
