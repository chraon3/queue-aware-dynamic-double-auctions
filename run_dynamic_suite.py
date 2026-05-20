from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from dynamic_double_auction import DynamicConfig, train_dynamic


def suite(full: bool = False) -> list[DynamicConfig]:
    base = DynamicConfig(
        max_buyers=4,
        max_sellers=4,
        horizon=6,
        batch_size=80,
        train_steps=130,
        eval_episodes=450,
        regret_grid=5,
        regret_weight=2.0,
        regret_target=0.025,
        augmented_rho=4.0,
        wait_cost=0.02,
        arrival_prob_buyer=0.75,
        arrival_prob_seller=0.75,
        max_patience=5,
        abandon_base=0.01,
        abandon_slope=0.08,
        value_loss_weight=0.2,
        feature_mode="ranked",
        log_every=50,
    )
    seeds = [81, 83, 89, 91]
    if full:
        seeds.extend([93, 97, 101, 103, 107, 109])

    configs = []
    for idx, seed in enumerate(seeds):
        cfg = replace(
            base,
            seed=seed,
            out_dir=f"experiments/dynamic_patience_value_seed{seed}",
            value_loss_weight=0.3 if seed in {91, 103} else base.value_loss_weight,
        )
        if idx == 0:
            cfg = replace(cfg, train_steps=180, batch_size=96, eval_episodes=600)
        configs.append(cfg)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the 10-seed paper suite instead of the 4-seed quick suite.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose metrics.json already exists.")
    args = parser.parse_args()

    Path("experiments").mkdir(exist_ok=True)
    manifest = []
    for cfg in suite(full=args.full):
        if args.skip_existing and (Path(cfg.out_dir) / "metrics.json").exists():
            print(f"\n=== Skipping existing {cfg.out_dir} ===", flush=True)
            continue
        print(f"\n=== Running {cfg.out_dir} ===", flush=True)
        _, _, metrics = train_dynamic(cfg)
        manifest.append({"config": asdict(cfg), "metrics": metrics})
    suffix = "full" if args.full else "quick"
    Path(f"experiments/dynamic_manifest_{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
