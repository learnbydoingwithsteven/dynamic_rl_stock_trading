"""Trading environment and price simulation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class PriceSimulator:
    """Generate synthetic price data using a geometric random walk."""

    initial_price: float = 100.0
    drift: float = 0.0005
    volatility: float = 0.01
    seed: int | None = None

    prices: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if not self.prices:
            self.prices = [self.initial_price]

    @property
    def current_price(self) -> float:
        return self.prices[-1]

    def step(self) -> float:
        last_price = self.current_price
        shock = self.rng.normal(self.drift, self.volatility)
        # Ensure the price remains positive
        new_price = max(1e-3, last_price * (1.0 + shock))
        self.prices.append(new_price)
        return new_price

    def history(self) -> Sequence[float]:
        return self.prices


@dataclass
class PortfolioState:
    """Track the trading position for an agent."""

    position: int = 0  # -1 short, 0 flat, 1 long
    cumulative_reward: float = 0.0
    realized_rewards: List[float] = field(default_factory=list)

    def register_reward(self, reward: float) -> None:
        self.cumulative_reward += reward
        self.realized_rewards.append(reward)

    @property
    def win_rate(self) -> float:
        if not self.realized_rewards:
            return 0.0
        wins = sum(1 for r in self.realized_rewards if r > 0)
        return wins / len(self.realized_rewards)

    @property
    def average_reward(self) -> float:
        if not self.realized_rewards:
            return 0.0
        return float(np.mean(self.realized_rewards))


def categorize_return(price_return: float) -> int:
    """Discretize returns into bins for tabular RL agents."""

    if price_return < -0.01:
        return -2
    if price_return < -0.002:
        return -1
    if price_return < 0.002:
        return 0
    if price_return < 0.01:
        return 1
    return 2


def build_state_from_history(prices: Sequence[float], window: int) -> tuple[int, ...]:
    """Return a discrete state representation based on trailing returns."""

    if len(prices) < 2:
        return tuple(0 for _ in range(window))

    relevant_prices = prices[-(window + 1) :]
    returns = []
    for before, after in zip(relevant_prices[:-1], relevant_prices[1:]):
        returns.append((after / before) - 1.0)

    while len(returns) < window:
        returns.insert(0, 0.0)

    categorized = tuple(categorize_return(r) for r in returns[-window:])
    return categorized


ACTIONS = ["Hold", "Go Long", "Go Short"]
ACTION_MAP = {0: "Hold", 1: "Go Long", 2: "Go Short"}


class TradingEnvironment:
    """A light-weight trading environment using shared price simulation."""

    def __init__(
        self,
        price_simulator: PriceSimulator,
        window: int = 5,
        transaction_cost: float = 0.0005,
    ) -> None:
        self.price_simulator = price_simulator
        self.window = window
        self.transaction_cost = transaction_cost

    def initial_state(self) -> tuple[int, ...]:
        return build_state_from_history(self.price_simulator.history(), self.window)

    def advance_price(self) -> tuple[float, float]:
        """Advance the underlying price process by one step."""

        previous_price = self.price_simulator.current_price
        new_price = self.price_simulator.step()
        return previous_price, new_price

    def compute_reward(
        self, portfolio: PortfolioState, action: int, previous_price: float, new_price: float
    ) -> float:
        """Compute the reward for an action given the price transition."""

        reward = 0.0
        price_return = (new_price - previous_price) / previous_price
        if action == 1:  # Go long
            portfolio.position = 1
            reward = price_return
        elif action == 2:  # Go short
            portfolio.position = -1
            reward = -price_return
        else:  # Hold position maintains existing exposure
            reward = portfolio.position * price_return

        reward -= self.transaction_cost if action != 0 else 0.0
        portfolio.register_reward(reward)
        return reward

    def next_state(self) -> tuple[int, ...]:
        return build_state_from_history(self.price_simulator.history(), self.window)
