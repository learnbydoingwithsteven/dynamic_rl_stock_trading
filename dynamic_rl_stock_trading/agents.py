"""Implementations of simple reinforcement learning agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .environment import ACTIONS


@dataclass
class AgentMetrics:
    """Track metrics for an agent over time."""

    name: str
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)

    def register(self, reward: float, action: int) -> None:
        self.rewards.append(reward)
        self.actions.append(action)

    @property
    def cumulative_reward(self) -> float:
        return float(np.sum(self.rewards)) if self.rewards else 0.0

    @property
    def average_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def win_rate(self) -> float:
        if not self.rewards:
            return 0.0
        wins = sum(1 for r in self.rewards if r > 0)
        return wins / len(self.rewards)

    @property
    def last_action_name(self) -> str:
        if not self.actions:
            return "N/A"
        return ACTIONS[self.actions[-1]]

    @property
    def steps(self) -> int:
        return len(self.rewards)


class BaseAgent:
    """Interface implemented by trading agents."""

    def __init__(self, name: str, action_space: int = 3) -> None:
        self.name = name
        self.action_space = action_space
        self.metrics = AgentMetrics(name=name)

    def select_action(self, state: Tuple[int, ...]) -> int:
        raise NotImplementedError

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        raise NotImplementedError

    def update_metrics(self, reward: float, action: int) -> None:
        self.metrics.register(reward=reward, action=action)

    def training_details(self) -> Dict[str, Any]:
        """Return a dictionary with algorithm specific diagnostics."""

        return {
            "Strategy": self.name,
            "States Tracked": 0,
            "Exploration": "N/A",
            "Experience Steps": self.metrics.steps,
        }


class QLearningAgent(BaseAgent):
    """Basic tabular Q-learning agent."""

    def __init__(
        self,
        name: str = "Q-Learning",
        action_space: int = 3,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(name=name, action_space=action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure_state(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space, dtype=float)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...]) -> int:
        q_values = self._ensure_state(state)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_space))
        return int(np.argmax(q_values))

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        q_values = self._ensure_state(state)
        next_q_values = self._ensure_state(next_state)
        td_target = reward + self.gamma * np.max(next_q_values)
        q_values[action] += self.alpha * (td_target - q_values[action])

    def training_details(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "Strategy": self.name,
            "States Tracked": len(self.q_table),
            "Exploration": f"ε-greedy (ε={self.epsilon:.2f})",
            "Learning Rate": self.alpha,
            "Discount": self.gamma,
            "Experience Steps": self.metrics.steps,
        }


class SarsaAgent(BaseAgent):
    """Expected SARSA implementation for smoother policy evaluation."""

    def __init__(
        self,
        name: str = "SARSA",
        action_space: int = 3,
        alpha: float = 0.25,
        gamma: float = 0.95,
        epsilon: float = 0.15,
    ) -> None:
        super().__init__(name=name, action_space=action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure_state(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space, dtype=float)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...]) -> int:
        q_values = self._ensure_state(state)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_space))
        return int(np.argmax(q_values))

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        current_q_values = self._ensure_state(state)
        next_q_values = self._ensure_state(next_state)
        greedy_action = int(np.argmax(next_q_values))
        policy = np.ones(self.action_space) * (self.epsilon / self.action_space)
        policy[greedy_action] += 1 - self.epsilon
        expected_q = float(np.dot(policy, next_q_values))
        td_target = reward + self.gamma * expected_q
        current_q_values[action] += self.alpha * (td_target - current_q_values[action])

    def training_details(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "Strategy": self.name,
            "States Tracked": len(self.q_table),
            "Exploration": f"ε-greedy (ε={self.epsilon:.2f})",
            "Learning Rate": self.alpha,
            "Discount": self.gamma,
            "Experience Steps": self.metrics.steps,
        }


class RandomAgent(BaseAgent):
    """Baseline agent that selects actions uniformly at random."""

    def __init__(self, name: str = "Random", action_space: int = 3) -> None:
        super().__init__(name=name, action_space=action_space)

    def select_action(self, state: Tuple[int, ...]) -> int:  # noqa: D401
        return int(np.random.randint(self.action_space))

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:  # noqa: D401
        return

    def training_details(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "Strategy": self.name,
            "States Tracked": 0,
            "Exploration": "Uniform random",
            "Learning Rate": 0.0,
            "Discount": 0.0,
            "Experience Steps": self.metrics.steps,
        }


class DoubleQLearningAgent(BaseAgent):
    """Double Q-learning agent to reduce overestimation bias."""

    def __init__(
        self,
        name: str = "Double Q-Learning",
        action_space: int = 3,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(name=name, action_space=action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_a: Dict[Tuple[int, ...], np.ndarray] = {}
        self.q_table_b: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure_state(self, state: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        if state not in self.q_table_a:
            self.q_table_a[state] = np.zeros(self.action_space, dtype=float)
            self.q_table_b[state] = np.zeros(self.action_space, dtype=float)
        return self.q_table_a[state], self.q_table_b[state]

    def select_action(self, state: Tuple[int, ...]) -> int:
        qa, qb = self._ensure_state(state)
        combined = qa + qb
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_space))
        return int(np.argmax(combined))

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        qa, qb = self._ensure_state(state)
        next_qa, next_qb = self._ensure_state(next_state)

        if np.random.rand() < 0.5:
            best_action = int(np.argmax(qa))
            td_target = reward + self.gamma * next_qb[best_action]
            qa[action] += self.alpha * (td_target - qa[action])
        else:
            best_action = int(np.argmax(qb))
            td_target = reward + self.gamma * next_qa[best_action]
            qb[action] += self.alpha * (td_target - qb[action])

    def training_details(self) -> Dict[str, Any]:  # noqa: D401
        visited_states = set(self.q_table_a) | set(self.q_table_b)
        return {
            "Strategy": self.name,
            "States Tracked": len(visited_states),
            "Exploration": f"ε-greedy (ε={self.epsilon:.2f})",
            "Learning Rate": self.alpha,
            "Discount": self.gamma,
            "Experience Steps": self.metrics.steps,
        }


class BoltzmannQLearningAgent(BaseAgent):
    """Q-learning agent with Boltzmann exploration."""

    def __init__(
        self,
        name: str = "Boltzmann Q-Learning",
        action_space: int = 3,
        alpha: float = 0.25,
        gamma: float = 0.95,
        temperature: float = 0.5,
    ) -> None:
        super().__init__(name=name, action_space=action_space)
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure_state(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space, dtype=float)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...]) -> int:
        q_values = self._ensure_state(state)
        if self.temperature <= 0:
            return int(np.argmax(q_values))
        scaled = q_values / max(self.temperature, 1e-6)
        # Shift for numerical stability before computing softmax
        shifted = scaled - np.max(scaled)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return int(np.random.choice(self.action_space, p=probabilities))

    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        q_values = self._ensure_state(state)
        next_q_values = self._ensure_state(next_state)
        td_target = reward + self.gamma * np.max(next_q_values)
        q_values[action] += self.alpha * (td_target - q_values[action])

    def training_details(self) -> Dict[str, Any]:  # noqa: D401
        return {
            "Strategy": self.name,
            "States Tracked": len(self.q_table),
            "Exploration": f"Boltzmann (τ={self.temperature:.2f})",
            "Learning Rate": self.alpha,
            "Discount": self.gamma,
            "Experience Steps": self.metrics.steps,
        }
