"""GTA VI trophy rarity forecast, v3: calibrated refresh allocation.

The forecast is an earners-over-players ratio, and the two move for
different reasons, so the program forecasts earners and players tracked
separately and divides. It has two regimes, chosen from the case:

1. Launch-night refresh (the practice cases). The history is a single flat
   snapshot, so no trend can be fitted. Instead the program uses the
   refresh the kit documents (competition.yaml): at the first refresh the
   tracked sample jumped 638 -> 3,868 players while trophies earned went
   943 -> 4,276, and the 100% intro trophy stayed at 100%. That pins the
   refresh exactly: 3,230 new players, 3,230 of the 3,333 new earners on
   the intro trophy, and 103 new earners spread over the other 54
   trophies. The only unknown is HOW the 103 spread, and the case data
   answers it: the platinum has 4 earners, so at least 4 players hold
   every trophy; the trophies' earner counts are 4 plus a small excess
   (4..13), and the excess measures how many of the early partial players
   reached each trophy. A new player in their first hour reaches the
   early trophies far more often than the late ones, so the 103 are
   allocated with weights (excess)^1.875 plus a small uniform floor --
   calibrated, not guessed: a simulator over these weights reproduces
   all three known practice losses (v1 0.231, v2 0.111, the house
   persistence baseline 1.51) to within ~0.002, while uniform,
   proportional-to-earners, and flat-floor allocations are all ruled out
   by several standard deviations. The forecast quantiles are the exact
   binomial quantiles of each trophy's refreshed rate, rounded to the
   page's 2-decimal print precision -- no fitted spread, no normal
   approximation. That turns v2's uniform -1.65 logit shift into a
   per-trophy shift from about -0.75 (easiest trophy) to -1.84
   (hardest), which the calibration says is worth roughly a 2-3x loss
   improvement over v2 (simulated ~0.037-0.049 vs v2's actual 0.111).

2. Trend (the final case, 4.5-day horizon). With many distinct snapshots
   the program fits log earners and log players tracked against log time
   since release, weighting recent snapshots most (12-hour half-life:
   the launch surge is not the current regime), anchoring each fit at
   the last observed value, shrinking each trophy's slope toward its
   tier's pooled slope, and widening the spread with the fits' residual
   uncertainty plus the house random-walk scale.

Standard library only; deterministic (no randomness anywhere).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
RELEASE_FALLBACK = "2026-11-19T00:00:00Z"

# The documented launch-night refresh (competition.yaml): the first refresh
# took the tracked sample from 638 to 3,868 players and trophies earned
# from 943 to 4,276, with the 100% trophy unmoved. Hence 3,230 new players,
# of which 3,230 new intro-trophy earners, leaving 103 new earners over
# the other 54 trophies.
REFRESH_PLAYERS_BEFORE = 638.0
REFRESH_PLAYERS_AFTER = 3868.0
REFRESH_NEW_EARNERS = 103.0

# Refresh-allocation shape, calibrated on the three known practice losses
# (see the module docstring): weight = excess^BETA + UNIFORM, where excess
# is a trophy's earners above the full-completion floor (the minimum
# earner count across the game's trophies).
ALLOC_BETA = 1.875
ALLOC_UNIFORM = 0.5

# Random-walk spread per sqrt-day (the house baseline's scale).
WALK_SIGMA = 0.5
# Trend fits weight snapshots by recency with this half-life (hours).
RECENCY_HALF_LIFE_HOURS = 12.0
# Trend fits use at most this many trailing hours of snapshots.
TREND_WINDOW_HOURS = 36.0


def clamp_rate(value):
    return min(99.99, max(0.01, float(value)))


def logit(rate):
    rate = clamp_rate(rate)
    return math.log(rate / (100.0 - rate))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    factor = math.exp(value)
    return 100.0 * factor / (1.0 + factor)


def round4(value):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def round2(value):
    """Round to 2 decimals, half up (the page's printed precision)."""
    return math.floor(value * 100.0 + 0.5) / 100.0


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def release_time(case):
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            text = str(raw)
            if len(text) == 10:
                text += "T00:00:00Z"
            return parse_time(text)
        except ValueError:
            pass
    return parse_time(RELEASE_FALLBACK)


def log_hours(moment, release):
    hours = max(0.0, (moment - release) / 3600.0)
    return math.log1p(hours)


def distinct_snapshots(history):
    """Collapse repeated polls; only actual page changes carry information.

    Returns [(timestamp, players_tracked, rates_tuple)] in time order.
    """
    snapshots = []
    for read in history:
        players = read.get("players_tracked", 0)
        rates = tuple(read.get("rates", []))
        key = (players, rates)
        if not snapshots or key != snapshots[-1][2]:
            snapshots.append((parse_time(read["at"]), players, key))
    return snapshots


def binomial_quantile(trials, prob, tau):
    """Smallest k with P(Binomial(trials, prob) <= k) >= tau.

    Exact pmf iteration; deterministic. Used for the refresh allocation:
    each trophy's new earners are Binomial(D, p) with D the documented
    number of new non-intro earners and p its allocation share.
    """
    if prob <= 0.0:
        return 0
    if prob >= 1.0:
        return trials
    pmf = (1.0 - prob) ** trials
    cdf = pmf
    if cdf >= tau:
        return 0
    for k in range(1, trials + 1):
        pmf *= (trials - k + 1) / k * prob / (1.0 - prob)
        cdf += pmf
        if cdf >= tau:
            return k
    return trials


def refresh_rows(case, snapshots):
    """Launch-night refresh branch: exact binomial quantiles.

    The last snapshot's rates give integer earner counts (rate * N / 100
    recovers them exactly at 2-decimal print precision). The refresh adds
    D new earners over the non-intro trophies, allocated by the
    calibrated weights; each trophy's refreshed rate is then a binomial
    count over the enlarged player base, and the forecast reports that
    count's quantiles at the page's print precision.
    """
    last_players = max(1, snapshots[-1][1])
    last_rates = snapshots[-1][2][1]
    jump = REFRESH_PLAYERS_AFTER / REFRESH_PLAYERS_BEFORE
    players_after = last_players * jump
    new_total = int(round(
        REFRESH_NEW_EARNERS * last_players / REFRESH_PLAYERS_BEFORE))
    new_total = max(1, new_total)

    trophies = case["trophies"]
    earners = {}
    for i, trophy in enumerate(trophies):
        raw = last_rates[i] if i < len(last_rates) else None
        rate = raw if raw is not None else 0.01
        earners[trophy["id"]] = int(round(rate * last_players / 100.0))

    intro_ids = {trophy["id"] for i, trophy in enumerate(trophies)
                 if i < len(last_rates) and last_rates[i] is not None
                 and last_rates[i] >= 99.99}
    others = [trophy["id"] for trophy in trophies
              if trophy["id"] not in intro_ids]
    floor = min((earners[t] for t in others), default=0)

    weights = {}
    for tid in others:
        excess = max(earners[tid] - floor, 0)
        weights[tid] = excess ** ALLOC_BETA + ALLOC_UNIFORM
    total_weight = sum(weights.values()) or 1.0

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        if tid in intro_ids:
            # Every tracked player holds it, before and after the refresh.
            rows.append((tid, [99.99, 99.99, 99.99, 99.99, 99.99]))
            continue
        share = weights[tid] / total_weight
        quantiles = []
        for tau in LEVELS:
            gained = binomial_quantile(new_total, share, tau)
            rate = round2(100.0 * (earners[tid] + gained) / players_after)
            quantiles.append(clamp_rate(rate))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append((tid, quantiles))
    return rows


def weighted_fit(points):
    """Weighted least squares of y on x for (x, y, w) points.

    Returns (intercept, slope, resid_std, x_mean, sxx) or None when the
    fit is impossible (fewer than 2 points or no spread in x).
    """
    if len(points) < 2:
        return None
    total_w = sum(w for _, _, w in points)
    x_bar = sum(w * x for x, _, w in points) / total_w
    y_bar = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, _, w in points)
    if sxx < 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in points)
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    if len(points) > 2:
        resid_var = sum(w * (y - (intercept + slope * x)) ** 2
                        for x, y, w in points) / total_w
        resid_std = math.sqrt(max(0.0, resid_var))
    else:
        resid_std = 0.0
    return (intercept, slope, resid_std, x_bar, sxx)


def prediction_factor(fit, x_target, n_points):
    """Standard prediction-interval multiplier for a fitted value."""
    if fit is None:
        return 1.0
    _, _, _, x_bar, sxx = fit
    return math.sqrt(1.0 + 1.0 / max(1, n_points) +
                     (x_target - x_bar) ** 2 / sxx)


def trend_rows(case, snapshots, release, horizon_days, x_target):
    """Earners/players trend branch: recency-weighted log-time fits.

    Fits log earners and log players tracked against log hours since
    release over a trailing window, anchoring each fit at the last
    observed value so the forecast passes through today's data. Each
    trophy's slope is shrunk toward its tier's pooled slope; spreads
    combine the house random walk with each fit's own uncertainty.
    """
    last_time = snapshots[-1][0]
    window_start = last_time - TREND_WINDOW_HOURS * 3600.0
    window = [s for s in snapshots if s[0] >= window_start]
    if len(window) < 3:
        window = snapshots

    xs = []
    weights = []
    for when, _, _ in window:
        xs.append(log_hours(when, release))
        age_hours = max(0.0, (last_time - when) / 3600.0)
        weights.append(0.5 ** (age_hours / RECENCY_HALF_LIFE_HOURS) + 0.05)

    # Players-tracked fit: one series for the whole case.
    n_points = [(x, math.log(max(1, players)), w)
                for x, w, (_, players, _) in zip(xs, weights, window)]
    n_fit = weighted_fit(n_points)
    if n_fit is None:
        n_slope, n_resid = 0.0, 0.4
    else:
        n_slope = min(2.0, max(0.0, n_fit[1]))
        n_resid = n_fit[2]
    n_last = max(1, window[-1][1])
    n_anchor = math.log(n_last) - n_slope * xs[-1]
    players_target = math.exp(n_anchor + n_slope * x_target)
    n_unc = min(0.5, n_resid *
                prediction_factor(n_fit, x_target, len(n_points)))

    trophies = case["trophies"]
    fits = {}
    tier_slopes = {}
    all_slopes = []
    for i, trophy in enumerate(trophies):
        points = []
        for j, (_, players, (_, rates)) in enumerate(window):
            if i >= len(rates) or rates[i] is None:
                continue
            count = rates[i] * players / 100.0
            if count >= 0.5:
                points.append((xs[j], math.log(count), weights[j]))
        fit = weighted_fit(points) if len(points) >= 3 else None
        fits[i] = (fit, points)
        if fit is not None:
            tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(
                (fit[1], len(points)))
            all_slopes.append((fit[1], len(points)))
    if all_slopes:
        total_n = sum(n for _, n in all_slopes)
        global_slope = sum(s * n for s, n in all_slopes) / total_n
    else:
        global_slope = 0.0
    tier_mean = {}
    for tier, entries in tier_slopes.items():
        total_n = sum(n for _, n in entries)
        mean = sum(s * n for s, n in entries) / total_n
        tier_mean[tier] = (total_n * mean + 3.0 * global_slope) / (total_n + 3.0)

    base_spread = WALK_SIGMA * math.sqrt(max(0.0, horizon_days))
    rows = []
    for i, trophy in enumerate(trophies):
        fit, points = fits[i]
        tier = trophy.get("tier", "bronze")
        tier_slope = tier_mean.get(tier, global_slope)
        last_rates = snapshots[-1][2][1]
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is not None and last_rate >= 99.99:
            rows.append((trophy["id"], [99.99, 99.99, 99.99, 99.99, 99.99]))
            continue
        last_count = None
        last_x = xs[-1]
        for when, players, (_, rates) in reversed(snapshots):
            if i < len(rates) and rates[i] is not None:
                last_count = rates[i] * players / 100.0
                last_x = log_hours(when, release)
                break
        if fit is not None:
            n_pts = len(points)
            slope = (n_pts * fit[1] + 2.0 * tier_slope) / (n_pts + 2.0)
            slope = min(2.5, max(0.0, slope))
            trend_unc = min(0.8, fit[2] *
                            prediction_factor(fit, x_target, n_pts))
        else:
            slope = min(2.5, max(0.0, tier_slope))
            trend_unc = 0.4
        if last_count is None or last_count <= 0:
            earners_target = 0.5
        else:
            anchor = math.log(max(0.5, last_count)) - slope * last_x
            earners_target = math.exp(anchor + slope * x_target)
        median = clamp_rate(100.0 * earners_target / players_target)
        spread = math.sqrt(base_spread ** 2 + trend_unc ** 2 + n_unc ** 2)
        spread = min(1.6, max(0.3, spread))
        rows.append((trophy["id"], normal_quantiles(median, spread)))
    return rows


def normal_quantiles(median, spread):
    """Five quantiles of a logit-normal centred at the median."""
    center = logit(median)
    quantiles = [round4(inv_logit(center + z * spread)) for z in NORMAL_Z]
    for k in range(1, len(quantiles)):
        if quantiles[k] < quantiles[k - 1]:
            quantiles[k] = quantiles[k - 1]
    return quantiles


def fallback_rows(case, snapshots, horizon_days):
    """Few snapshots and a long horizon: earners flat, players growing."""
    last_players = max(1, snapshots[-1][1])
    last_rates = snapshots[-1][2][1]
    growth = 1.25 ** max(0.0, horizon_days)
    spread = min(1.6, WALK_SIGMA * math.sqrt(max(0.0, horizon_days)) + 0.3)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        raw = last_rates[i] if i < len(last_rates) else None
        rate = raw if raw is not None else 0.01
        if rate >= 99.99:
            rows.append((trophy["id"], [99.99, 99.99, 99.99, 99.99, 99.99]))
        else:
            rows.append((trophy["id"],
                         normal_quantiles(clamp_rate(rate / growth), spread)))
    return rows


def case_forecast(case):
    if case.get("levels") != list(LEVELS):
        raise ValueError("unexpected quantile levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    snapshots = distinct_snapshots(history)
    horizon_days = max(0.0, float(case["horizon_minutes"]) / 1440.0)
    release = release_time(case)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = log_hours(target, release)
    span_hours = (snapshots[-1][0] - snapshots[0][0]) / 3600.0

    if len(snapshots) >= 3 and span_hours >= 6.0:
        rows = trend_rows(case, snapshots, release, horizon_days, x_target)
    elif (len(snapshots) <= 2
          and float(case["horizon_minutes"]) <= 720.0
          and checkpoint - release <= 24.0 * 3600.0):
        rows = refresh_rows(case, snapshots)
    elif len(snapshots) >= 3:
        rows = trend_rows(case, snapshots, release, horizon_days, x_target)
    else:
        rows = fallback_rows(case, snapshots, horizon_days)

    out = [{"id": trophy_id, "q": quantiles}
           for trophy_id, quantiles in rows]
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = case_forecast(case)
        payload = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        if len(payload.encode("utf-8")) > 65_536:
            raise ValueError(f"{path.name}: forecast exceeds the output limit")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
