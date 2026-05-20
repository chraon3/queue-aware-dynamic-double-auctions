from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from econpinn_double_auction import Config, train, write_report


def suite() -> list[Config]:
    base = Config(
        batch_size=320,
        train_steps=650,
        eval_samples=3000,
        regret_eval_samples=900,
        regret_grid=9,
        eval_grid=31,
        hidden=64,
        depth=3,
        lr=2.0e-3,
        log_every=50,
    )
    return [
        replace(
            base,
            out_dir="experiments/dual_uniform_3x3_seed7",
            train_method="dual",
            regret_method="grid",
            eval_regret_method="grid",
            regret_weight=2.0,
            regret_target=0.006,
            dual_lr=1.0,
            distribution="uniform",
            seed=7,
        ),
        replace(
            base,
            out_dir="experiments/dual_uniform_3x3_adv_eval_seed7",
            train_method="dual",
            regret_method="grid",
            eval_regret_method="adv",
            adv_steps=8,
            adv_restarts=3,
            regret_weight=2.0,
            regret_target=0.006,
            dual_lr=1.0,
            distribution="uniform",
            seed=7,
            eval_samples=2200,
            regret_eval_samples=450,
        ),
        replace(
            base,
            out_dir="experiments/dual_beta_easy_3x3_seed11",
            train_method="dual",
            regret_method="grid",
            eval_regret_method="grid",
            regret_weight=2.0,
            regret_target=0.006,
            dual_lr=1.0,
            distribution="beta_easy",
            seed=11,
        ),
        replace(
            base,
            out_dir="experiments/dual_correlated_3x3_seed13",
            train_method="dual",
            regret_method="grid",
            eval_regret_method="grid",
            regret_weight=2.0,
            regret_target=0.006,
            dual_lr=1.0,
            distribution="correlated",
            seed=13,
        ),
        replace(
            base,
            out_dir="experiments/dual_uniform_5x5_seed17",
            n_buyers=5,
            n_sellers=5,
            batch_size=220,
            train_steps=520,
            eval_samples=1800,
            regret_eval_samples=500,
            train_method="dual",
            regret_method="grid",
            eval_regret_method="grid",
            regret_weight=2.0,
            regret_target=0.006,
            dual_lr=1.0,
            distribution="uniform",
            seed=17,
        ),
    ]


def main() -> None:
    Path("experiments").mkdir(exist_ok=True)
    manifest = []
    for cfg in suite():
        print(f"\n=== Running {cfg.out_dir} ===", flush=True)
        _, _, metrics = train(cfg)
        write_report(cfg, metrics)
        manifest.append({"config": asdict(cfg), "metrics": metrics})
    Path("experiments/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
