"""Streamlit application for dynamic RL stock trading simulation."""
from __future__ import annotations

import time
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from dynamic_rl_stock_trading.agents import QLearningAgent, RandomAgent, SarsaAgent
from dynamic_rl_stock_trading.environment import PriceSimulator
from dynamic_rl_stock_trading.simulation import SimulationManager

st.set_page_config(page_title="Dynamic RL Stock Trading", layout="wide")


def create_agents(selected: List[str]) -> List:
    mapping = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent,
        "Random": RandomAgent,
    }
    return [mapping[name]() for name in selected]


def render_price_chart(price_frame: pd.DataFrame) -> None:
    line_chart = (
        alt.Chart(price_frame)
        .mark_line()
        .encode(x="step", y="price")
        .properties(height=300)
    )
    st.altair_chart(line_chart, use_container_width=True)


def render_rewards_chart(reward_frame: pd.DataFrame) -> None:
    if reward_frame.empty:
        st.info("Run the simulation to see reward trajectories.")
        return

    melted = reward_frame.reset_index().melt(id_vars="index", var_name="Agent", value_name="Cumulative Reward")
    reward_chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(x="index", y="Cumulative Reward", color="Agent")
        .properties(height=300)
    )
    st.altair_chart(reward_chart, use_container_width=True)


st.title("📈 Dynamic RL Stock Trading Playground")
st.markdown(
    """
    Explore how different reinforcement learning agents react to a live simulated market. Use the
    controls on the sidebar to configure the price dynamics and simulation speed, then watch the
    agents learn and adapt in real time.
    """
)

with st.sidebar:
    st.header("Simulation Settings")
    num_steps = st.slider("Simulation steps", min_value=10, max_value=500, value=120, step=10)
    drift = st.slider("Drift", min_value=-0.005, max_value=0.01, value=0.0005, step=0.0005)
    volatility = st.slider("Volatility", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
    transaction_cost = st.slider("Transaction cost", min_value=0.0, max_value=0.005, value=0.0005, step=0.0005)
    update_delay = st.slider("Update delay (seconds)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    selected_agents = st.multiselect(
        "Agents",
        options=["Q-Learning", "SARSA", "Random"],
        default=["Q-Learning", "SARSA", "Random"],
    )

if not selected_agents:
    st.warning("Select at least one agent to start the simulation.")
    st.stop()

run_button = st.button("Run Simulation", type="primary")

if run_button:
    agents = create_agents(selected_agents)
    price_simulator = PriceSimulator(drift=drift, volatility=volatility)
    manager = SimulationManager(agents=agents, price_simulator=price_simulator, transaction_cost=transaction_cost)

    price_placeholder = st.empty()
    metrics_placeholder = st.empty()
    reward_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for step in range(1, num_steps + 1):
        metrics_snapshot: Dict[str, Dict[str, float]] = manager.step()
        price_frame = manager.price_history_frame()
        metrics_frame = manager.metrics_frame()
        reward_frame = manager.rewards_over_time()

        with price_placeholder.container():
            st.subheader("Simulated Price")
            render_price_chart(price_frame)

        with metrics_placeholder.container():
            st.subheader("Agent Metrics")
            st.dataframe(metrics_frame.style.format({"Cumulative Reward": "{:.4f}", "Average Reward": "{:.4f}", "Win Rate": "{:.2%}"}))

        with reward_placeholder.container():
            st.subheader("Cumulative Reward Trajectories")
            render_rewards_chart(reward_frame)

        progress_bar.progress(step / num_steps)
        status_lines = [
            f"**{name}** → Reward: {values['reward']:.4f}, Cumulative: {values['cumulative_reward']:.4f}, Action: {values['last_action']}"
            for name, values in metrics_snapshot.items()
        ]
        status_placeholder.markdown("\n".join(status_lines))
        time.sleep(update_delay)

    st.success("Simulation complete!")
else:
    st.info("Configure settings and press 'Run Simulation' to start.")
