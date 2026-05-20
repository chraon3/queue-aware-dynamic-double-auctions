from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from econpinn_double_auction import (
    Config,
    HardConstrainedDoubleAuction,
    sample_market,
    utilities,
)


def load_config(path: Path) -> Config:
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = Config.__dataclass_fields__.keys()
    return Config(**{key: value for key, value in data.items() if key in valid})


def load_model(run_dir: Path) -> tuple[Config, HardConstrainedDoubleAuction]:
    cfg = load_config(run_dir / "config.json")
    device = torch.device(cfg.device)
    model = HardConstrainedDoubleAuction(cfg.hidden, cfg.depth, cfg.feature_mode, cfg.clearance_cost).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return cfg, model


def side_certificate(
    model: HardConstrainedDoubleAuction,
    buyer_values: torch.Tensor,
    seller_costs: torch.Tensor,
    grid: torch.Tensor,
    side: str,
    agent_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = buyer_values.shape[0]
    grid_size = grid.numel()
    b_rep = buyer_values.repeat_interleave(grid_size, dim=0)
    s_rep = seller_costs.repeat_interleave(grid_size, dim=0)
    if side == "buyer":
        b_rep[:, agent_idx] = grid.repeat(batch)
    elif side == "seller":
        s_rep[:, agent_idx] = grid.repeat(batch)
    else:
        raise ValueError(side)

    out = model(b_rep, s_rep)
    u_b, u_s = utilities(
        out,
        buyer_values.repeat_interleave(grid_size, dim=0),
        seller_costs.repeat_interleave(grid_size, dim=0),
    )
    utility = (u_b[:, agent_idx] if side == "buyer" else u_s[:, agent_idx]).view(batch, grid_size)
    truthful_out = model(buyer_values, seller_costs)
    truthful_b, truthful_s = utilities(truthful_out, buyer_values, seller_costs)
    truthful = truthful_b[:, agent_idx] if side == "buyer" else truthful_s[:, agent_idx]
    grid_regret = torch.relu(utility.max(dim=1).values - truthful)

    mesh = float((grid[1] - grid[0]).item()) if grid_size > 1 else 1.0
    local_slopes = (utility[:, 1:] - utility[:, :-1]).abs() / max(mesh, 1.0e-12)
    lipschitz = local_slopes.max(dim=1).values
    continuous_bound = grid_regret + 0.5 * mesh * lipschitz
    return grid_regret, lipschitz, continuous_bound


def summarize(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    flat = values.detach().flatten()
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_p95": float(torch.quantile(flat, 0.95).item()),
        f"{prefix}_max": float(flat.max().item()),
    }


def certify_run(run_dir: Path, samples: int, grid_size: int, seed_offset: int, distribution: str | None) -> Dict[str, float | str]:
    cfg, model = load_model(run_dir)
    if distribution:
        cfg = replace(cfg, distribution=distribution)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed + seed_offset)
    buyers, sellers = sample_market(samples, cfg.n_buyers, cfg.n_sellers, device, cfg.distribution)
    grid = torch.linspace(0.0, 1.0, grid_size, device=device)

    buyer_regrets = []
    seller_regrets = []
    buyer_lipschitz = []
    seller_lipschitz = []
    buyer_bounds = []
    seller_bounds = []
    with torch.no_grad():
        for i in range(cfg.n_buyers):
            regret, lip, bound = side_certificate(model, buyers, sellers, grid, "buyer", i)
            buyer_regrets.append(regret)
            buyer_lipschitz.append(lip)
            buyer_bounds.append(bound)
        for j in range(cfg.n_sellers):
            regret, lip, bound = side_certificate(model, buyers, sellers, grid, "seller", j)
            seller_regrets.append(regret)
            seller_lipschitz.append(lip)
            seller_bounds.append(bound)

    b_reg = torch.stack(buyer_regrets, dim=1)
    s_reg = torch.stack(seller_regrets, dim=1)
    b_lip = torch.stack(buyer_lipschitz, dim=1)
    s_lip = torch.stack(seller_lipschitz, dim=1)
    b_bound = torch.stack(buyer_bounds, dim=1)
    s_bound = torch.stack(seller_bounds, dim=1)
    all_reg = torch.cat([b_reg.flatten(), s_reg.flatten()])
    all_lip = torch.cat([b_lip.flatten(), s_lip.flatten()])
    all_bound = torch.cat([b_bound.flatten(), s_bound.flatten()])

    row: Dict[str, float | str] = {
        "run": run_dir.name,
        "distribution": cfg.distribution,
        "samples": float(samples),
        "grid_size": float(grid_size),
        "mesh": float(1.0 / max(grid_size - 1, 1)),
    }
    row.update(summarize(all_reg, "grid_regret"))
    row.update(summarize(all_lip, "lipschitz"))
    row.update(summarize(all_bound, "certified_regret"))
    row.update(summarize(b_bound, "buyer_certified_regret"))
    row.update(summarize(s_bound, "seller_certified_regret"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["experiments/ranked_dual_uniform_3x3_seed51"])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--distribution", default="")
    parser.add_argument("--out-dir", default="experiments/static_ic_certificate")
    args = parser.parse_args()

    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(Path().glob(pattern)))
    run_dirs = [path for path in run_dirs if (path / "model.pt").exists()]
    if not run_dirs:
        raise SystemExit("No model runs found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, run_dir in enumerate(run_dirs):
        print(f"Certifying {run_dir.name}", flush=True)
        row = certify_run(
            run_dir,
            samples=args.samples,
            grid_size=args.grid_size,
            seed_offset=70_000 + idx * 997,
            distribution=args.distribution or None,
        )
        rows.append(row)
        (out_dir / f"{run_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "static_ic_certificate_by_run.csv", index=False)
    summary = raw.drop(columns=["run", "distribution"]).agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(out_dir / "static_ic_certificate_summary.csv", index=False)
    print(raw.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
