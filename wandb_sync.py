"""Upload finished results to Weights & Biases after the fact, so W&B can never stall or break a training run.

For every results/ae_gate_<name>.json: one run (group "ae_gate") with the training curve as history, the gate
metrics per dev set as summary, and the config. Also one "data" run with the data and state-cache stats.
Runs already uploaded are recorded in results/wandb_synced.json and skipped.

Needs a prior `wandb login` by the user. Run: python wandb_sync.py [--project soft-spec-decoding]
"""
import argparse
import glob
import json
import os

import wandb

from common import RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="soft-spec-decoding")
    args = ap.parse_args()
    if not wandb.login(anonymous="never", verify=True):
        raise SystemExit("not logged in to W&B: run `wandb login` first")
    done_p = os.path.join(RES, "wandb_synced.json")
    done = json.load(open(done_p)) if os.path.exists(done_p) else []
    for p in sorted(glob.glob(os.path.join(RES, "ae_gate_*.json"))):
        name = os.path.basename(p)[:-5]
        if name in done or name.endswith("_smoke"):
            continue
        r = json.load(open(p))
        run = wandb.init(project=args.project, group="ae_gate", name=name, config=r["args"], reinit="finish_previous")
        for row in r["train_log"]:
            run.log({k: v for k, v in row.items() if k != "step"}, step=row["step"])
        summ = {"gate_pass": r["pass"]}
        for s, m in r["sets"].items():
            summ.update({f"{s}/agree": m["agree"], f"{s}/tv": m["tv"], f"{s}/kl": m["kl"], f"{s}/gate_pass": m["gate"]["pass"]})
            summ.update({f"{s}/agree_noise_{k}": v for k, v in m["agree_noise"].items()})
            summ.update({f"{s}/agree_margin_{b}": v[0] for b, v in m["agree_by_margin"].items() if v[0] is not None})
        run.summary.update(summ)
        run.finish()
        done.append(name)
        print("synced", name, flush=True)
    if "data" not in done:
        run = wandb.init(project=args.project, group="data", name="data_and_state_stats", reinit="finish_previous")
        for f in ("data_stats.json", "state_stats.json"):
            p = os.path.join(RES, f)
            if os.path.exists(p):
                art = wandb.Artifact(f[:-5], type="stats")
                art.add_file(p)
                run.log_artifact(art)
        st = json.load(open(os.path.join(RES, "state_stats.json")))
        tab = wandb.Table(columns=["split", "positions", "top1_eq_next", "frac_margin_below_0.5", "median_rel_flip_radius"])
        for k, v in st["splits"].items():
            tab.add_data(k, v["positions"], v["top1_eq_next"], v["frac_margin_below_0.5"], v["rel_flip_radius_quantiles"]["0.5"])
        run.log({"state_cache": tab})
        run.finish()
        done.append("data")
        print("synced data", flush=True)
    json.dump(done, open(done_p, "w"), indent=1)


if __name__ == "__main__":
    main()
