# -*- coding: utf-8 -*-
"""
06grid_search_oip_tcr.py
- 不训练：只扫 12~20 组 OIP/TCR 超参
- 自动：追踪 -> 评估 -> 汇总每序列增益 + overall 最优组合
- 贴合当前结构：
  YOLOX_outputs/<tracker_name>/tracks/BF_XX.txt
  YOLOX_outputs/eval_m2la_final/<tracker_name>_summary.csv
  YOLOX_outputs/eval_m2la_final/<tracker_name>_hota_per_seq.csv

用法示例（在 BoT-SORT-main 根目录运行）：
  python tools/BFront_MOT/06grid_search_oip_tcr.py --det tempX --cmc busfront --runs 12
 python tools/BFront_MOT/06grid_search_oip_tcr.py --det tempX --cmc busfront --runs 12 --baseline_tracker m2la_botsort_tempX_busfront_v2

可选：
  --baseline_tracker m2la_botsort_tempX_busfront_v2   # 用于计算增益(每序列+overall)
"""





import itertools

from pathlib import Path

import os
import sys
PYTHON_EXE = sys.executable
import time
import shutil
import argparse
import subprocess
from datetime import datetime

import pandas as pd


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_rmtree(p: str):
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)


def move_dir(src: str, dst: str):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source folder not found: {src}")
    if os.path.exists(dst):
        safe_rmtree(dst)
    shutil.move(src, dst)


def run_cmd(cmd, env=None):
    print("\n[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd} (ret={p.returncode})")


def load_overall_from_summary(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    # 你的 summary.csv 最后一行是 OVERALL
    row = df.iloc[-1].to_dict()
    return row


def load_hota_per_seq(csv_path: str) -> pd.DataFrame:
    # eval 脚本输出的 per-seq HOTA 表（包含 BF_01... + COMBINED/OVERALL）
    df = pd.read_csv(csv_path)
    return df


def pick_score_key(metrics: dict, prefer_keys):
    for k in prefer_keys:
        if k in metrics:
            return k
    return None


# def build_param_grid(max_runs: int):
#     """
#     12~20 组默认网格（你可按当前最优点附近继续加密）
#     注意：这些参数要能通过 env 注入到 matching/bot_sort（见下方“必须确保读取 env”）。
#     """
#     # Soft OIP 强度（加在 cost 上的惩罚/奖励系数）
#     oip_alpha_list = [0.2, 0.25,0.3]
#     # Soft TCR 强度（连续性奖励）
#     tcr_beta_list = [0.2]
#     # 触发门槛：当 IoU/运动一致性较差时才启用 Soft OIP/TCR（避免“过拟合式干预”）
#     trigger_iou_list = [0.35]
#     # 小目标阈值（高度/面积的归一化阈值，用于“遮挡感知”）
#     small_norm_list = [0.01]
#
#     grid = []
#     for a in oip_alpha_list:
#         for b in tcr_beta_list:
#             for tiou in trigger_iou_list:
#                 for sn in small_norm_list:
#                     grid.append({
#                         "OIP_ALPHA": a,
#                         "TCR_BETA": b,
#                         "TRIGGER_IOU": tiou,
#                         "SMALL_NORM": sn,
#                     })
#
#     # 截断到 max_runs（默认 12~20）
#     return grid[:max_runs]

def build_param_grid(max_runs: int):
    grid = [
        {"OIP_ALPHA":0.25, "TCR_BETA":0.20, "TRIGGER_IOU":0.35, "SMALL_NORM":0.010},
        {"OIP_ALPHA":0.22, "TCR_BETA":0.20, "TRIGGER_IOU":0.35, "SMALL_NORM":0.010},
        {"OIP_ALPHA":0.28, "TCR_BETA":0.20, "TRIGGER_IOU":0.35, "SMALL_NORM":0.010},
        {"OIP_ALPHA":0.25, "TCR_BETA":0.16, "TRIGGER_IOU":0.35, "SMALL_NORM":0.010},
        {"OIP_ALPHA":0.25, "TCR_BETA":0.24, "TRIGGER_IOU":0.35, "SMALL_NORM":0.010},
        {"OIP_ALPHA":0.25, "TCR_BETA":0.20, "TRIGGER_IOU":0.30, "SMALL_NORM":0.010},

        {"OIP_ALPHA":0.15, "TCR_BETA":0.10, "TRIGGER_IOU":0.25, "SMALL_NORM":0.012},
        {"OIP_ALPHA":0.15, "TCR_BETA":0.12, "TRIGGER_IOU":0.25, "SMALL_NORM":0.012},
        {"OIP_ALPHA":0.18, "TCR_BETA":0.12, "TRIGGER_IOU":0.30, "SMALL_NORM":0.012},
        {"OIP_ALPHA":0.20, "TCR_BETA":0.10, "TRIGGER_IOU":0.20, "SMALL_NORM":0.012},

        {"OIP_ALPHA":0.25, "TCR_BETA":0.20, "TRIGGER_IOU":0.35, "SMALL_NORM":0.008},
        {"OIP_ALPHA":0.25, "TCR_BETA":0.20, "TRIGGER_IOU":0.35, "SMALL_NORM":0.015},
    ]
    return grid[:max_runs]




def infer_default_out_name(det: str, cmc: str) -> str:
    # 04track_external_dets_factorial.py 当前默认输出目录名基本是：
    # YOLOX_outputs/m2la_botsort_<det>_<cmc>
    # 你日志里是 m2la_botsort_tempX_busfront
    det = det.strip()
    cmc = cmc.strip()
    return f"m2la_botsort_{det}_{cmc}"


def dynamic_import_eval_module(eval_py_path: str):
    """
    动态 import 你的 05eval_all_trackers_final.py，
    让它只评估我们指定的 tracker_name（避免你手工改 TRACKERS 列表）。
    """
    import importlib.util
    eval_py_path = abspath(eval_py_path)
    spec = importlib.util.spec_from_file_location("eval_m2la_dynamic", eval_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def run_eval_one_tracker(eval_mod, tracker_name: str):
    """
    复用 05eval_all_trackers_final.py 的逻辑：
    - 强行让 TRACKERS = [tracker_name]
    - 调用 main()
    """
    if hasattr(eval_mod, "TRACKERS"):
        eval_mod.TRACKERS = [tracker_name]
    else:
        # 你的脚本如果不是 TRACKERS，就用 tracker_names
        if hasattr(eval_mod, "tracker_names"):
            eval_mod.tracker_names = [tracker_name]
        else:
            raise AttributeError("eval script has neither TRACKERS nor tracker_names")

    # 某些版本 eval 脚本用 OUT_DIR 写结果（eval_m2la_final）
    # 这里不改 OUT_DIR，沿用你脚本里写死的即可
    eval_mod.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", type=str, default="tempX", choices=["raw", "tempX", "temp1", "temp"], help="det variant")
    ap.add_argument("--cmc", type=str, default="busfront", help="cmc method, e.g., busfront/none")
    ap.add_argument("--runs", type=int, default=12, help="grid size (12~20 recommended)")
    ap.add_argument("--baseline_tracker", type=str, default="", help="baseline tracker name for delta computation")
    ap.add_argument("--track_py", type=str, default="E:/track/BoT-SORT-main/tools/BFront_MOT/04track_external_dets_factorial.py")
    ap.add_argument("--eval_py", type=str, default="E:/track/BoT-SORT-main/tools/BFront_MOT/05eval_all_trackers_final.py")
    ap.add_argument("--yolox_outputs", type=str, default="YOLOX_outputs/grid_search_oip_tcr")
    ap.add_argument("--work_dir", type=str, default="", help="BoT-SORT-main root. empty = current cwd")
    ap.add_argument("--python", type=str, default=PYTHON_EXE, help="python executable")
    args = ap.parse_args()

    work_dir = abspath(args.work_dir) if args.work_dir else os.getcwd()
    yolox_outputs = abspath(os.path.join(work_dir, args.yolox_outputs))



    # ==========================================================
    # 06grid_search_oip_tcr.py
    # - No training. Runs 12~20 hyperparam combos:
    #   tracking (04...) -> evaluation (05...) -> summary
    # - Fits your current directory layout under:
    #   E:\track\BoT-SORT-main\YOLOX_outputs
    # ==========================================================

    PY = sys.executable

    ROOT = Path(r"E:\track\BoT-SORT-main")
    TOOLS = ROOT / "tools" / "BFront_MOT"

    TRACK_SCRIPT = TOOLS / "04track_external_dets_factorial.py"
    EVAL_SCRIPT = TOOLS / "05eval_all_trackers_final.py"

    # baseline tracker folder name (without tag)
    BASE_EXP = "m2la_botsort_tempX_busfront"

    OUT_DIR = ROOT / "YOLOX_outputs" /"grid_search_oip_tcr"/ "eval_grid"

    def run_one(tag: str, hp: dict):
        # 1) tracking
        cmd = [
            PY, str(TRACK_SCRIPT),
            "--det", "tempX",
            "--cmc", "busfront",
            "--tag", tag,
            "--oip_alpha", str(hp["oip_alpha"]),
            "--oip_tau", str(hp["oip_tau"]),
            "--oip_min_iou", str(hp["oip_min_iou"]),
            "--tcr_beta", str(hp["tcr_beta"]),
            "--tcr_max_dt", str(hp["tcr_max_dt"]),
            "--trigger_occ", str(hp["trigger_occ"]),
            "--trigger_small", str(hp["trigger_small"]),
        ]
        print("\n[RUN] Tracking:", " ".join(cmd))
        t0 = time.time()
        subprocess.check_call(cmd, cwd=str(ROOT))
        t1 = time.time()

        # 2) evaluation for this tracker only
        tracker_name = f"{BASE_EXP}_{tag}"
        os.makedirs(OUT_DIR, exist_ok=True)
        cmd2 = [PY, str(EVAL_SCRIPT), "--trackers", tracker_name, "--out_dir", str(OUT_DIR)]
        print("[RUN] Eval:", " ".join(cmd2))
        subprocess.check_call(cmd2, cwd=str(ROOT))
        t2 = time.time()

        # 3) read overall metrics from produced summary CSV
        summary_csv = OUT_DIR / f"{tracker_name}_summary.csv"
        hota_csv = OUT_DIR / f"{tracker_name}_hota_per_seq.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(summary_csv)

        import pandas as pd
        s = pd.read_csv(summary_csv, index_col=0)
        # index uses BF_01.. + OVERALL
        if "OVERALL" in s.index:
            row = s.loc["OVERALL"]
        else:
            # fallback: last row
            row = s.iloc[-1]

        out = {
            "tracker": tracker_name,
            "tag": tag,
            "sec_track": round(t1 - t0, 2),
            "sec_eval": round(t2 - t1, 2),
            "mota": float(row.get("mota", row.get("MOTA", float("nan")))),
            "idf1": float(row.get("idf1", row.get("IDF1", float("nan")))),
            "ids": float(row.get("num_switches", row.get("IDs", float("nan")))),
            "fp": float(row.get("num_false_positives", row.get("FP", float("nan")))),
            "fn": float(row.get("num_misses", row.get("FN", float("nan")))),
        }

        # HOTA from hota_per_seq if exists
        if hota_csv.exists():
            h = pd.read_csv(hota_csv, index_col=0)
            if "OVERALL" in h.index:
                out["hota"] = float(h.loc["OVERALL"].get("HOTA", float("nan")))
                out["deta"] = float(h.loc["OVERALL"].get("DetA", float("nan")))
                out["assa"] = float(h.loc["OVERALL"].get("AssA", float("nan")))
            else:
                out["hota"] = float("nan")
                out["deta"] = float("nan")
                out["assa"] = float("nan")
        else:
            out["hota"] = float("nan")
            out["deta"] = float("nan")
            out["assa"] = float("nan")

        out.update({f"hp_{k}": v for k, v in hp.items()})
        return out

    def main():
        # 12~20 combos: pick a compact grid focused on IDs/HOTA
        grid = {
            "oip_alpha": [0.05, 0.10, 0.15],
            "oip_tau": [0.25, 0.35],
            "oip_min_iou": [0.03],
            "tcr_beta": [0.05, 0.10],
            "tcr_max_dt": [3, 5],
            "trigger_occ": [0.30],
            "trigger_small": [0.25],
        }

        keys = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))

        # cap to 20
        combos = combos[:20]
        print(f"[INFO] Total combos = {len(combos)}  (cap=20)")

        results = []
        for idx, vals in enumerate(combos, 1):
            hp = dict(zip(keys, vals))
            tag = f"GS{idx:02d}"
            try:
                res = run_one(tag, hp)
                results.append(res)
                print("[OK]", tag, "mota=", res["mota"], "idf1=", res["idf1"], "ids=", res["ids"], "hota=",
                      res.get("hota"))
            except Exception as e:
                print("[ERR]", tag, e)

        # save table
        import pandas as pd
        df = pd.DataFrame(results)
        out_csv = OUT_DIR / "grid_results_overall.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("[OK] Saved:", out_csv)

        # choose best by (min IDs, max HOTA) lexicographically
        if len(df):
            df2 = df.copy()
            df2["ids"] = pd.to_numeric(df2["ids"], errors="coerce")
            df2["hota"] = pd.to_numeric(df2["hota"], errors="coerce")
            best = df2.sort_values(["ids", "hota"], ascending=[True, False]).head(1)
            print("\n[BEST]\n", best.to_string(index=False))

    if __name__ == "__main__":
        main()

    track_py = abspath(os.path.join(work_dir, args.track_py))
    eval_py = abspath(os.path.join(work_dir, args.eval_py))

    if not os.path.isfile(track_py):
        raise FileNotFoundError(f"track script not found: {track_py}")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval script not found: {eval_py}")

    ensure_dir(yolox_outputs)

    # 04track 默认输出目录（会被我们“运行后重命名”）
    default_out_name = infer_default_out_name(args.det, args.cmc)
    default_out_dir = os.path.join(yolox_outputs, default_out_name)

    # 评估输出目录（你 eval 脚本内部写死的 OUT_DIR；这里用它默认值）
    # 我们只需要知道生成的 summary / hota_per_seq 在哪里
    # 从你日志看：YOLOX_outputs/eval_m2la_final/<tracker>_summary.csv
    eval_out_dir = os.path.join(yolox_outputs, "eval_m2la_tempX_busfront")

    # 组网格
    runs = max(1, min(int(args.runs), 30))
    grid = build_param_grid(runs)
    if len(grid) == 0:
        raise RuntimeError("Empty grid.")

    # baseline（用于增益）
    baseline_overall = None
    baseline_hota_per_seq = None
    if args.baseline_tracker:
        base_sum = os.path.join(eval_out_dir, f"{args.baseline_tracker}_summary.csv")
        base_hota = os.path.join(eval_out_dir, f"{args.baseline_tracker}_hota_per_seq.csv")
        if os.path.isfile(base_sum):
            baseline_overall = load_overall_from_summary(base_sum)
        if os.path.isfile(base_hota):
            baseline_hota_per_seq = load_hota_per_seq(base_hota)

    # 动态加载 eval 模块
    eval_mod = dynamic_import_eval_module(eval_py)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gs_root = os.path.join(yolox_outputs, f"gridsearch_{args.det}_{args.cmc}_{stamp}")
    ensure_dir(gs_root)

    all_rows = []

    print("\n========== GRID SEARCH START ==========")
    print("work_dir      :", work_dir)
    print("track_py      :", track_py)
    print("eval_py       :", eval_py)
    print("default_out   :", default_out_dir)
    print("eval_out_dir  :", eval_out_dir)
    print("gs_root       :", gs_root)
    print("runs          :", len(grid))
    if args.baseline_tracker:
        print("baseline      :", args.baseline_tracker)

    for idx, hp in enumerate(grid, 1):
        run_tag = f"gs{idx:02d}_a{hp['OIP_ALPHA']}_b{hp['TCR_BETA']}_tiou{hp['TRIGGER_IOU']}_sn{hp['SMALL_NORM']}"
        tracker_name = f"{default_out_name}_{run_tag}"
        print("\n--------------------------------------")
        print(f"[{idx}/{len(grid)}] Running: {tracker_name}")
        print("HP:", hp)

        # 1) 清理默认输出目录（避免叠加旧 tracks）
        if os.path.isdir(default_out_dir):
            safe_rmtree(default_out_dir)

        # 2) 跑追踪（只跑 det tempX + cmc busfront）
        # 通过环境变量注入超参（matching/bot_sort 必须读取这些 env）
        env = os.environ.copy()
        env["BFRONT_SOFT_OIP_ALPHA"] = str(hp["OIP_ALPHA"])
        env["BFRONT_SOFT_TCR_BETA"] = str(hp["TCR_BETA"])
        env["BFRONT_SOFT_TRIGGER_IOU"] = str(hp["TRIGGER_IOU"])
        env["BFRONT_SMALL_NORM"] = str(hp["SMALL_NORM"])
        env["BFRONT_SOFT_ENABLE"] = "1"

        cmd_track = [
            args.python,
            track_py,
            "--det", args.det,
            "--cmc", args.cmc
        ]
        t0 = time.time()
        run_cmd(cmd_track, env=env)
        t1 = time.time()

        # 3) 把默认输出目录重命名成独立 tracker_name（用于评估）
        if not os.path.isdir(default_out_dir):
            raise RuntimeError(f"Tracking finished but output folder missing: {default_out_dir}")

        run_out_dir = os.path.join(yolox_outputs, tracker_name)
        move_dir(default_out_dir, run_out_dir)

        # 4) 跑评估（只评估当前 tracker_name）
        ensure_dir(eval_out_dir)
        run_eval_one_tracker(eval_mod, tracker_name)

        # 5) 读取评估结果
        sum_csv = os.path.join(eval_out_dir, f"{tracker_name}_summary.csv")
        hota_csv = os.path.join(eval_out_dir, f"{tracker_name}_hota_per_seq.csv")

        if not os.path.isfile(sum_csv):
            raise FileNotFoundError(f"Missing summary: {sum_csv}")

        overall = load_overall_from_summary(sum_csv)

        # 尽量兼容不同列名
        hota_key = pick_score_key(overall, ["hota", "HOTA", "HOTA_mean", "HOTA(0)"])
        idf1_key = pick_score_key(overall, ["idf1", "IDF1"])
        ids_key = pick_score_key(overall, ["num_switches", "IDs", "ids", "num_ids"])

        hota_val = float(overall[hota_key]) if hota_key else float("nan")
        idf1_val = float(overall[idf1_key]) if idf1_key else float("nan")
        ids_val = float(overall[ids_key]) if ids_key else float("nan")

        row = {
            "tracker": tracker_name,
            "run_tag": run_tag,
            "OIP_ALPHA": hp["OIP_ALPHA"],
            "TCR_BETA": hp["TCR_BETA"],
            "TRIGGER_IOU": hp["TRIGGER_IOU"],
            "SMALL_NORM": hp["SMALL_NORM"],
            "time_sec": round(t1 - t0, 2),
            "HOTA": hota_val,
            "IDF1": idf1_val,
            "IDs": ids_val,
        }

        # 增益（overall）
        if baseline_overall is not None:
            base_hota = float(baseline_overall.get(hota_key, float("nan"))) if hota_key else float("nan")
            base_idf1 = float(baseline_overall.get(idf1_key, float("nan"))) if idf1_key else float("nan")
            base_ids = float(baseline_overall.get(ids_key, float("nan"))) if ids_key else float("nan")
            row["dHOTA"] = row["HOTA"] - base_hota
            row["dIDF1"] = row["IDF1"] - base_idf1
            row["dIDs"] = base_ids - row["IDs"]  # IDs 越小越好：用“减少量”表示增益

        all_rows.append(row)

        # 每序列增益（只做 HOTA；IDF1 你也可以类似做）
        if os.path.isfile(hota_csv) and baseline_hota_per_seq is not None:
            try:
                cur = load_hota_per_seq(hota_csv)
                # 找 HOTA 列
                hk = None
                for c in cur.columns:
                    if c.lower() == "hota" or c.lower().startswith("hota"):
                        hk = c
                        break
                if hk:
                    # 以 seq 作为键（列名可能是 "seq" 或 "Sequence"）
                    seq_col = None
                    for c in cur.columns:
                        if c.lower() in ["seq", "sequence", "name"]:
                            seq_col = c
                            break
                    if seq_col is None:
                        seq_col = cur.columns[0]
                    cur_map = dict(zip(cur[seq_col].astype(str), cur[hk].astype(float)))

                    bseq_col = None
                    for c in baseline_hota_per_seq.columns:
                        if c.lower() in ["seq", "sequence", "name"]:
                            bseq_col = c
                            break
                    if bseq_col is None:
                        bseq_col = baseline_hota_per_seq.columns[0]
                    bhk = hk if hk in baseline_hota_per_seq.columns else None
                    if bhk:
                        base_map = dict(zip(baseline_hota_per_seq[bseq_col].astype(str),
                                            baseline_hota_per_seq[bhk].astype(float)))
                        # 输出一个 per-seq delta 文件（可画曲线）
                        per_seq_out = os.path.join(gs_root, f"{tracker_name}_dHOTA_per_seq.csv")
                        rows = []
                        for seq, v in cur_map.items():
                            if seq in base_map:
                                rows.append({"seq": seq, "dHOTA": v - base_map[seq]})
                        pd.DataFrame(rows).to_csv(per_seq_out, index=False)
            except Exception as e:
                print("[WARN] per-seq delta failed:", e)

        # 保存进度
        pd.DataFrame(all_rows).to_csv(os.path.join(gs_root, "grid_results_running.csv"), index=False)

    # 选最优：优先 HOTA，其次 IDs（更少更好），再次 IDF1
    df = pd.DataFrame(all_rows)
    df_sorted = df.sort_values(by=["HOTA", "IDs", "IDF1"], ascending=[False, True, False])
    best = df_sorted.iloc[0].to_dict()

    out_all = os.path.join(gs_root, "grid_results_all.csv")
    out_best = os.path.join(gs_root, "grid_best.csv")
    df_sorted.to_csv(out_all, index=False)
    pd.DataFrame([best]).to_csv(out_best, index=False)

    print("\n========== GRID SEARCH DONE ==========")
    print("Saved:", out_all)
    print("Best :", out_best)
    print("BEST CONFIG:", best)


if __name__ == "__main__":
    main()
