# Dynamic RL Stock Trading

This project provides a Streamlit application that simulates a live stock market and visualises how
multiple reinforcement learning (RL) agents adapt their strategies in real time. The app creates a
synthetic price series, lets agents trade on it, and surfaces detailed metrics such as per-step
rewards, win rates, and cumulative performance.

## Features

- 📈 **Real-time price simulation** using a configurable geometric random walk.
- 🤖 **Multiple RL agents** (Q-Learning, SARSA, and a random baseline) trade simultaneously.
- 📊 **Live metrics dashboard** showing cumulative rewards, win rates, last actions, and more.
- ⚙️ **Customisable parameters** including drift, volatility, transaction cost, and refresh delay.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**

   ```bash
   streamlit run app.py
   ```

3. Open the provided local URL in your browser. Adjust settings from the sidebar and click *Run
   Simulation* to watch the agents learn from the evolving market.

## Project structure

```
.
├── app.py                         # Streamlit entry point
├── dynamic_rl_stock_trading       # Simulation package
│   ├── __init__.py
│   ├── agents.py                  # RL agent implementations
│   ├── environment.py             # Price process and trading environment
│   └── simulation.py              # Simulation manager and helpers
├── requirements.txt
└── README.md
```

## Notes

- Rewards and price changes are presented in relative (%) terms for easier comparison across runs.
- The simulation is intentionally lightweight for educational purposes; agents use tabular methods
  with discretised states rather than deep neural networks.
