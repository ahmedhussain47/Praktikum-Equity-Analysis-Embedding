"""
risk_manager.py — Position sizing and capital protection for the signal engine.

Position sizing follows a modified fixed-fractional approach where the fraction
risked per trade is scaled down by (a) signal confidence and (b) current drawdown.

Rules enforced
──────────────
1. Never risk more than `max_risk_pct` of current equity per trade (default 1%).
2. Scale position DOWN when confidence < 80% (half-size at 60–69%).
3. Block new trades when current drawdown exceeds `max_drawdown_pct` (default 10%).
4. Require minimum R/R ratio before entering (default 1.5).
5. Never let a single trade lose more than `hard_stop_pct` of account (default 2%).
"""

from dataclasses import dataclass
from typing import Optional

from .signal_engine import Signal


# ── Trade setup dataclass ──────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    asset:             str
    direction:         str
    entry:             float
    stop_loss:         float
    take_profit:       float
    confidence:        int
    rr_ratio:          float

    position_units:    float    # Number of shares / ounces / lots / pips
    risk_amount_usd:   float    # Max dollar loss if SL is hit
    reward_amount_usd: float    # Max dollar gain if TP is hit
    account_risk_pct:  float    # Fraction of account at risk (0–1)

    def __str__(self) -> str:
        return (
            f"{self.direction} {self.asset}  "
            f"entry={self.entry:.5f}  SL={self.stop_loss:.5f}  TP={self.take_profit:.5f}\n"
            f"  Size={self.position_units:.4f}  Risk=${self.risk_amount_usd:.2f} "
            f"({self.account_risk_pct*100:.2f}% of account)  "
            f"Reward=${self.reward_amount_usd:.2f}  R/R={self.rr_ratio:.1f}x"
        )


# ── Risk manager ───────────────────────────────────────────────────────────────

class RiskManager:
    """
    Calculate position sizes and enforce capital-protection rules.

    Parameters
    ──────────
    account_balance:    Starting (or current) account balance in USD.
    max_risk_pct:       Maximum fraction of account to risk per trade (0.01 = 1%).
    hard_stop_pct:      Absolute ceiling on per-trade risk, regardless of sizing.
    max_drawdown_pct:   Pause trading when drawdown exceeds this fraction.
    min_rr_ratio:       Reject signals whose R/R is below this value.
    pip_value:          Dollar value of 1 pip for 1 standard forex lot.
                        Only relevant for pip-based sizing (Forex).
    """

    def __init__(
        self,
        account_balance:    float,
        max_risk_pct:       float = 0.01,
        hard_stop_pct:      float = 0.02,
        max_drawdown_pct:   float = 0.10,
        min_rr_ratio:       float = 1.5,
        pip_value:          float = 10.0,
    ):
        self._initial_balance = account_balance
        self._current_balance = account_balance
        self._peak_balance    = account_balance

        self.max_risk_pct     = max_risk_pct
        self.hard_stop_pct    = hard_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.min_rr_ratio     = min_rr_ratio
        self.pip_value        = pip_value

    # ── Balance management ─────────────────────────────────────────────────────

    @property
    def balance(self) -> float:
        return self._current_balance

    def update_balance(self, new_balance: float) -> None:
        """Call after each closed trade to keep drawdown tracking current."""
        self._current_balance = new_balance
        if new_balance > self._peak_balance:
            self._peak_balance = new_balance

    def record_pnl(self, pnl_usd: float) -> None:
        """Convenience: add/subtract PnL from balance."""
        self.update_balance(self._current_balance + pnl_usd)

    @property
    def current_drawdown(self) -> float:
        """Fraction drawn down from peak equity (0 = no drawdown)."""
        if self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - self._current_balance) / self._peak_balance

    # ── Confidence scaling ─────────────────────────────────────────────────────

    @staticmethod
    def _confidence_multiplier(confidence: int) -> float:
        """
        Scale position size by signal confidence.

        Confidence  Multiplier
        ──────────  ──────────
        < 60        0.00  (blocked)
        60–69       0.50
        70–79       0.75
        80–89       1.00
        90+         1.25  (max allowed premium)
        """
        if confidence >= 90:
            return 1.25
        if confidence >= 80:
            return 1.00
        if confidence >= 70:
            return 0.75
        if confidence >= 60:
            return 0.50
        return 0.00

    # ── Drawdown scaling ───────────────────────────────────────────────────────

    def _drawdown_multiplier(self) -> float:
        """
        Reduce position size as drawdown increases (defensive scaling).

        Drawdown   Multiplier
        ────────   ──────────
        0–4%       1.00
        4–7%       0.75
        7–10%      0.50
        > 10%      0.00  (blocked by guard rail)
        """
        dd = self.current_drawdown
        if dd < 0.04:
            return 1.00
        if dd < 0.07:
            return 0.75
        if dd < 0.10:
            return 0.50
        return 0.00   # Will be caught by the guard in size_position()

    # ── Main sizing method ─────────────────────────────────────────────────────

    def size_position(self, signal: Signal) -> Optional[TradeSetup]:
        """
        Calculate position size for a given signal.

        Returns None when any capital-protection rule blocks the trade.
        Returns a TradeSetup with full sizing details otherwise.
        """
        # Guard: drawdown ceiling
        if self.current_drawdown >= self.max_drawdown_pct:
            print(
                f"[RiskManager] BLOCKED {signal.asset}: "
                f"drawdown {self.current_drawdown:.1%} ≥ limit {self.max_drawdown_pct:.1%}"
            )
            return None

        # Guard: minimum R/R
        if signal.rr_ratio < self.min_rr_ratio:
            print(
                f"[RiskManager] BLOCKED {signal.asset}: "
                f"R/R {signal.rr_ratio:.2f} < min {self.min_rr_ratio:.2f}"
            )
            return None

        # Guard: confidence multiplier
        conf_mult = self._confidence_multiplier(signal.confidence)
        if conf_mult == 0.0:
            print(
                f"[RiskManager] BLOCKED {signal.asset}: "
                f"confidence {signal.confidence}% too low"
            )
            return None

        # Guard: drawdown multiplier
        dd_mult = self._drawdown_multiplier()
        if dd_mult == 0.0:
            print(f"[RiskManager] BLOCKED {signal.asset}: drawdown multiplier = 0")
            return None

        # Effective risk fraction
        effective_risk = self.max_risk_pct * conf_mult * dd_mult

        # Hard stop: never exceed hard_stop_pct regardless of scaling
        effective_risk = min(effective_risk, self.hard_stop_pct)

        # Dollar risk budget
        risk_usd = self._current_balance * effective_risk

        # Distance from entry to stop loss
        sl_dist = signal.sl_distance
        if sl_dist <= 0:
            sl_dist = abs(signal.entry - signal.stop_loss)
        if sl_dist <= 0:
            print(f"[RiskManager] BLOCKED {signal.asset}: SL distance is zero")
            return None

        # Position units (shares, ounces, etc.)
        # For forex lots, divide by pip_value × pips; here we use price units.
        units = risk_usd / sl_dist

        # Actual risk / reward in USD
        risk_amount   = units * sl_dist
        tp_dist       = abs(signal.take_profit - signal.entry)
        reward_amount = units * tp_dist

        return TradeSetup(
            asset             = signal.asset,
            direction         = signal.signal,
            entry             = signal.entry,
            stop_loss         = signal.stop_loss,
            take_profit       = signal.take_profit,
            confidence        = signal.confidence,
            rr_ratio          = signal.rr_ratio,
            position_units    = round(units, 6),
            risk_amount_usd   = round(risk_amount, 2),
            reward_amount_usd = round(reward_amount, 2),
            account_risk_pct  = round(risk_amount / self._current_balance, 5),
        )

    # ── Portfolio-level guard ──────────────────────────────────────────────────

    def max_concurrent_risk(self, open_trades: list) -> float:
        """
        Total account fraction at risk across all currently open trades.
        Pass a list of TradeSetup objects for open positions.
        """
        return sum(t.account_risk_pct for t in open_trades)

    def can_open_trade(
        self,
        signal:       Signal,
        open_trades:  list,
        max_total_risk: float = 0.05,    # 5% total concurrent risk
    ) -> bool:
        """
        Check whether opening a new trade would exceed portfolio risk limits.
        """
        setup = self.size_position(signal)
        if setup is None:
            return False
        current_risk = self.max_concurrent_risk(open_trades)
        if current_risk + setup.account_risk_pct > max_total_risk:
            print(
                f"[RiskManager] Portfolio risk {current_risk + setup.account_risk_pct:.1%} "
                f"would exceed {max_total_risk:.1%} — trade skipped"
            )
            return False
        return True

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "balance":           round(self._current_balance, 2),
            "peak_balance":      round(self._peak_balance, 2),
            "current_drawdown":  f"{self.current_drawdown:.2%}",
            "max_risk_pct":      f"{self.max_risk_pct:.1%}",
            "hard_stop_pct":     f"{self.hard_stop_pct:.1%}",
            "max_drawdown_pct":  f"{self.max_drawdown_pct:.1%}",
            "min_rr_ratio":      self.min_rr_ratio,
        }
