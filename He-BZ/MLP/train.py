import os, json, subprocess, shutil, pathlib, matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
OUT_ROOT = "dataset"
CONFIG_FILE = "input.json"
RUN_DP = True

type_map_path = os.path.join(OUT_ROOT, "type_map.raw")
if os.path.isfile(type_map_path):
    with open(type_map_path, "r") as f:
        TYPE_MAP = [ln.strip() for ln in f if ln.strip()]
    if not TYPE_MAP:
        raise RuntimeError(f"{type_map_path} is empty.")
else:
    print(f"{type_map_path} not found. Using fallback type map: ['X']")
    TYPE_MAP = ["X"]

WR_TRAIN = "dataset_train"
WR_VAL = "dataset_val"
os.makedirs(WR_TRAIN, exist_ok=True)
os.makedirs(WR_VAL, exist_ok=True)

def safe_symlink(src_dir, wrapper_dir, set_name):
    src = os.path.join(src_dir, set_name)
    dst = os.path.join(wrapper_dir, set_name)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"{src} not found.")
    if os.path.islink(dst) or os.path.isdir(dst):
        return
    os.symlink(os.path.relpath(src, start=wrapper_dir), dst)

safe_symlink(OUT_ROOT, WR_TRAIN, "set.000")
safe_symlink(OUT_ROOT, WR_VAL, "set.001")

base_sel_per_type = 32
SEL_LIST = [base_sel_per_type] * len(TYPE_MAP)

config = {
    "model": {
        "type_map": TYPE_MAP,
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 8.0,
            "rcut_smth": 0.5,
            "sel": SEL_LIST,
            "neuron": [32, 64, 128],
            "axis_neuron": 24,
            "resnet_dt": False
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": False,
            "activation_function": "tanh"
        }
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 1e-3,
        "decay_steps": 200000,
        "stop_lr": 1e-5
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1.0,
        "start_pref_f": 1000.0,
        "limit_pref_f": 10.0,
        "start_pref_v": 0.0,
        "limit_pref_v": 0.0
    },
    "training": {
        "seed": 12345,
        "numb_steps": 800000,
        "disp_file": "lcurve.out",
        "save_ckpt": "model.ckpt",
        "disp_freq": 1000,
        "save_freq": 5000,
        "training_data": {
            "systems": [f"./{WR_TRAIN}"],
            "batch_size": 2
        },
        "validation_data": {
            "systems": [f"./{WR_VAL}"],
            "batch_size": 2
        },
    }
}

with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"Wrote {CONFIG_FILE}")
print(f"TYPE_MAP = {TYPE_MAP}")
print(f"systems.train = {config['training']['training_data']['systems']}")
print(f"systems.val   = {config['training']['validation_data']['systems']}")

dp_bin = shutil.which("dp")
if dp_bin is None:
    print("dp command not found.")
    exit(1)

if RUN_DP:
    print(f"Using executable: {dp_bin}")
    log_out = open("log.out", "w")
    log_err = open("log.err", "w")
    p = subprocess.Popen([dp_bin, "train", CONFIG_FILE], stdout=log_out, stderr=log_err)
    p.wait()
    log_out.close()
    log_err.close()
    print(f"Training finished, return code = {p.returncode}")

    if any(pathlib.Path(".").glob("model.ckpt*")):
        print("Exporting frozen_model.pb")
        subprocess.run([dp_bin, "freeze", "-o", "frozen_model.pb"], check=False)
else:
    print("Run manually: dp train input.json 2> log.err | tee log.out")

lc = pathlib.Path("lcurve.out")
if lc.exists():
    steps, losses = [], []
    for ln in lc.read_text().splitlines():
        tok = ln.strip().split()
        if len(tok) >= 2 and tok[0].isdigit():
            steps.append(int(tok[0]))
            losses.append(float(tok[1]))
    if steps:
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("DeepMD Learning Curve")
        plt.tight_layout()
        plt.savefig("learning_curve.png", dpi=200)
        print("Saved learning_curve.png")