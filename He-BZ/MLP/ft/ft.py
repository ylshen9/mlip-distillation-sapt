import os, json, subprocess, shutil, pathlib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "dataset"
TYPE_MAP_FILE = os.path.join(DATA_ROOT, "type_map.raw")
TRAIN_JSON = "train.json"
RUN_DP = True

RESTART_BASENAME = "model.ckpt-800000"
OUTPUT_PB = "model_final.pb"

START_LR = 1e-5
DECAY_STEPS = 80000
STOP_LR = 1e-6
STOP_BATCH = 1600000
NUMB_TEST = 200

BASE_SEL_PER_TYPE = 32
DESCRIPTOR = {
    "type": "se_e2_a",
    "rcut": 8.0,
    "rcut_smth": 0.5,
    "neuron": [32, 64, 128],
    "axis_neuron": 24,
    "resnet_dt": False
}
FITTING_NET = {
    "neuron": [240, 240, 240],
    "resnet_dt": False,
    "activation_function": "tanh"
}

with open(TYPE_MAP_FILE, "r") as f:
    TYPE_LIST = [ln.strip() for ln in f if ln.strip()]
if not TYPE_LIST:
    raise RuntimeError(f"[ERROR] {TYPE_MAP_FILE} is empty")

sel_list = [BASE_SEL_PER_TYPE] * len(TYPE_LIST)

cfg = {
    "model": {
        "type_map": TYPE_LIST,
        "descriptor": dict(DESCRIPTOR, sel=sel_list),
        "fitting_net": FITTING_NET,
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 1e-4,
        "decay_steps": 80000,
        "stop_lr": 5e-5
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 1,
        "limit_pref_e": 2.0,
        "start_pref_f": 0,
        "limit_pref_f": 0.0,
        "start_pref_v": 0.0,
        "limit_pref_v": 0.0
    },
    "training": {
        "training_data": {
            "systems": ["dp_train_sets"],
            "batch_size": 2
        },
        "validation_data": {
            "systems": ["dp_val_sets"],
            "batch_size": 2
        },
        "numb_steps": STOP_BATCH,
        "disp_file": "lcurve.out",
        "disp_freq": 1000,
        "save_ckpt": "model.ckpt",
        "save_freq": 5000,
        "seed": 12345
    }
}

with open(TRAIN_JSON, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(f"[OK] {TRAIN_JSON} written")
print(f"[INFO] elements={TYPE_LIST} -> sel={sel_list}")

dp_bin = shutil.which("dp")
if dp_bin is None:
    raise SystemExit("[ERROR] dp command not found. Please activate deepmd-kit environment.")
print(f"[INFO] Using executable: {dp_bin}")

if RUN_DP:
    restart_flag = []
    if (pathlib.Path(RESTART_BASENAME + ".index").exists()
            and any(pathlib.Path(".").glob(RESTART_BASENAME + ".data*"))):
        restart_flag = ["--restart", RESTART_BASENAME]
        print(f"[INFO] Using --restart {RESTART_BASENAME}")
    else:
        print("[WARN] No checkpoint detected. Training from scratch.")

    with open("log.out", "w") as log_out, open("log.err", "w") as log_err:
        cmd = [dp_bin, "train", TRAIN_JSON] + restart_flag
        print("[RUN]", " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=log_out, stderr=log_err)
        p.wait()
        print(f"[INFO] Training finished with return code {p.returncode}")

    if any(pathlib.Path(".").glob("model.ckpt*")):
        print(f"[INFO] Freezing model to {OUTPUT_PB}")
        subprocess.run([dp_bin, "freeze", "-o", OUTPUT_PB], check=False)

lc = pathlib.Path("lcurve.out")
if lc.exists():
    steps, losses = [], []
    for ln in lc.read_text().splitlines():
        tok = ln.strip().split()
        if len(tok) >= 2 and tok[0].isdigit():
            try:
                steps.append(int(tok[0]))
                losses.append(float(tok[1]))
            except:
                pass
    if steps:
        plt.figure()
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("DeepMD Learning Curve")
        plt.tight_layout()
        plt.savefig("learning_curve.png", dpi=200)
        print("[OK] Learning curve saved to learning_curve.png")
