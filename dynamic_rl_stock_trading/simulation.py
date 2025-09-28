"""Simulation utilities to run trading agents in real time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .agents import BaseAgent
from .environment import ACTION_MAP, PriceSimulator, PortfolioState, TradingEnvironment


@dataclass
class AgentContext:
    """Hold per-agent state used during a simulation run."""

    agent: BaseAgent
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    state: Tuple[int, ...] | None = None


class SimulationManager:
    """Coordinate price simulation and agent learning."""

    def __init__(
        self,
        agents: Iterable[BaseAgent],
        price_simulator: PriceSimulator,
        window: int = 5,
        transaction_cost: float = 0.0005,
    ) -> None:
        self.price_simulator = price_simulator
        self.environment = TradingEnvironment(
            price_simulator=price_simulator, window=window, transaction_cost=transaction_cost
        )
        self.agent_contexts: Dict[str, AgentContext] = {
            agent.name: AgentContext(agent=agent, state=self.environment.initial_state())
            for agent in agents
        }
        self.history: List[Dict[str, float]] = []

    def step(self) -> Dict[str, Dict[str, float]]:
        """Perform a single simulation step for all agents."""

        metrics_snapshot: Dict[str, Dict[str, float]] = {}
        chosen_actions: Dict[str, int] = {}

        for name, context in self.agent_contexts.items():
            assert context.state is not None
            chosen_actions[name] = context.agent.select_action(context.state)

        previous_price, new_price = self.environment.advance_price()
        next_state = self.environment.next_state()

        for name, context in self.agent_contexts.items():
            action = chosen_actions[name]
            reward = self.environment.compute_reward(
                context.portfolio, action, previous_price=previous_price, new_price=new_price
            )
            context.agent.learn(context.state, action, reward, next_state)
            context.agent.update_metrics(reward, action)
            context.state = next_state
            diagnostics = context.agent.training_details()
            metrics_snapshot[name] = {
                "reward": reward,
                "cumulative_reward": context.portfolio.cumulative_reward,
                "win_rate": context.portfolio.win_rate,
                "average_reward": context.portfolio.average_reward,
                "last_action": ACTION_MAP[action],
                "experience_steps": context.agent.metrics.steps,
                "states_tracked": diagnostics.get("States Tracked", 0),
                "exploration": diagnostics.get("Exploration", "N/A"),
            }

        self.history.append({name: values["cumulative_reward"] for name, values in metrics_snapshot.items()})
        return metrics_snapshot

    def price_history_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"step": range(len(self.price_simulator.history())), "price": self.price_simulator.history()})

    def metrics_frame(self) -> pd.DataFrame:
        records = []
        for name, context in self.agent_contexts.items():
            records.append(
                {
                    "Agent": name,
                    "Cumulative Reward": context.portfolio.cumulative_reward,
                    "Average Reward": context.portfolio.average_reward,
                    "Win Rate": context.portfolio.win_rate,
                    "Trades": len(context.portfolio.realized_rewards),
                    "Last Reward": context.agent.metrics.rewards[-1] if context.agent.metrics.rewards else 0.0,
                    "Last Action": context.agent.metrics.last_action_name,
                    "Experience Steps": context.agent.metrics.steps,
                }
            )
        return pd.DataFrame(records)

    def rewards_over_time(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def training_details_frame(self) -> pd.DataFrame:
        records = []
        for name, context in self.agent_contexts.items():
            details = dict(context.agent.training_details())
            details["Agent"] = name
            details.setdefault("Experience Steps", context.agent.metrics.steps)
            records.append(details)

        if not records:
            return pd.DataFrame()

        frame = pd.DataFrame(records)
        preferred_order = [
            "Agent",
            "Strategy",
            "Experience Steps",
            "States Tracked",
            "Exploration",
            "Learning Rate",
            "Discount",
        ]
        ordered_columns = [col for col in preferred_order if col in frame.columns]
        remaining_columns = [col for col in frame.columns if col not in ordered_columns]
        return frame[ordered_columns + remaining_columns]
