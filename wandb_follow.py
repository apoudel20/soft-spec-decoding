"""Stream a training log to Weights & Biases while the run is going, without touching the training process.

Follows logs/<run>.log from the start, so a run already in progress is backfilled first. Logs every JSON line that
has a "step" key as history, and dev result lines ("dev_math {...}", "EVAL {...}") as summary. Stops when the log
prints its "=== ... done" line or stays unchanged for --idle_min minutes. CPU only.

Needs a prior `wandb login` by the user.
Run:    python wandb_follow.py --log logs/stage2_v2.log --name gister_v2 [--group gister]
"""
import argparse
import json
import os
import time

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="gister")
    ap.add_argument("--project", default="soft-spec-decoding")
    ap.add_argument("--notes", default="")
    ap.add_argument("--idle_min", type=float, default=30)
    args = ap.parse_args()
    while not os.path.exists(args.log):
        time.sleep(10)
    run = wandb.init(project=args.project, name=args.name, group=args.group, notes=args.notes,
                     config={"log": args.log}, reinit=True)
    f = open(args.log, encoding="utf-8", errors="replace")
    last = time.time()
    while True:
        line = f.readline()
        if not line:
            if time.time() - last > args.idle_min * 60:
                break
            time.sleep(5)
            continue
        last = time.time()
        s = line.strip()
        try:
            if s.startswith("{") and '"step"' in s:
                r = json.loads(s)
                run.log({k: v for k, v in r.items() if isinstance(v, (int, float)) and k != "step"}, step=int(r["step"]))
            elif s.startswith("EVAL "):
                r = json.loads(s[5:])
                run.log({f"dev/{k}": v for k, v in r.items() if isinstance(v, (int, float)) and k != "step"},
                        step=int(r["step"]))
            elif s.startswith("dev_") and "{" in s:
                name, js = s.split(" ", 1)
                r = json.loads(js)
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        run.summary[f"{name}/{k}"] = v
            if s.startswith("=== ") and "done" in s:
                break
        except (ValueError, KeyError):
            pass
    run.finish()
    print("follow done", args.name, flush=True)


if __name__ == "__main__":
    main()
