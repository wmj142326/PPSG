##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################
# 1.1 (session-safe, GPU-aware, error-handled)
# Run:
#   streamlit run demo_app.py -- --model_path <model_dir_path> --n_generate <n>

import streamlit as st
import argparse
import torch
import numpy as np
import json
import os
from datetime import datetime
from smplx import SMPL
import text2pose.config as config
from text2pose.vocab import Vocabulary  # required by load_model deps
from text2pose.generative.evaluate_generative import load_model
import text2pose.utils_visu as utils_visu

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model directory.')
parser.add_argument('--n_generate', type=int, default=12, help='Number of poses to generate for each description.')
parser.add_argument('--smpl_model_path', type=str, default='/home/meijun/Projects/PPSG/data/smpl_body_models', help='Path to SMPL model folder.')
args = parser.parse_args()

# -----------------------------
# Global config
# -----------------------------
model_path = args.model_path
smpl_model_path = args.smpl_model_path
n_generate = max(1, int(args.n_generate))
nb_cols = 4
margin_img = 80

# Prefer GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SEED = 42

# -----------------------------
# Cached setup (model + body models)
# -----------------------------
@st.cache_resource(show_spinner=True)
def setup(model_path: str, smpl_model_path: str, device: str):
    """Load text2pose model, optional BodyModel for rendering, and SMPL for OBJ export."""
    # Load text2pose generator
    model, _ = load_model(model_path, device)

    # Try body_model for image rendering (optional)
    body_model = None
    try:
        import human_body_prior.body_model.body_model as hbp
        body_model = hbp.BodyModel(
            bm_fname=config.SMPL_NEUTRAL_BM,
            num_betas=config.SMPL_n_betas,
            model_type='smpl',
        )
        body_model.eval()
        body_model = body_model.to(device)
    except Exception as e:
        # Rendering will be disabled; we inform the user in the UI later
        body_model = None

    # SMPL from smplx is used on CPU for reliable OBJ export
    smpl_model = SMPL(model_path=smpl_model_path, gender='NEUTRAL', batch_size=1)

    return model, body_model, smpl_model

model, body_model, smpl_model = setup(model_path, smpl_model_path, device)

def fresh_seed() -> int:
    return int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF

# -----------------------------
# Helpers
# -----------------------------

def generate_and_render(description: str, seed: int = DEFAULT_SEED):
    """Generate n_generate poses from description and (optionally) render images.
    Returns (imgs, pose_params), where
      - imgs: list[np.ndarray] or list[None]
      - pose_params: torch.Tensor of shape [n_generate, 72]
    """
    # Seed to control stochasticity
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    with torch.no_grad():
        # Sample poses (expects a dict with key 'pose_body')
        out = model.sample_str_nposes(description, n=n_generate)
        pose_body = out['pose_body']  # expected shape [1, n_generate, 72] or [1, n_generate, 69+3]
        pose_params = pose_body[0, ...].reshape(n_generate, -1)  # [n_generate, 72]

    # Render if body_model is available
    if body_model is not None:
        try:
            imgs = utils_visu.image_from_pose_data(
                pose_params, body_model, viewpoints=[(-90, (1, 0, 0))]
            )
        except Exception:
            imgs = [None] * n_generate
    else:
        imgs = [None] * n_generate

    return imgs, pose_params


def save_single_pose(description: str, pose_parameters: torch.Tensor, index: int, save_dir: str = 'output'):
    """Save pose[index] to JSON (72D params) and OBJ (mesh vertices from SMPL)."""
    os.makedirs(save_dir, exist_ok=True)

    # Timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # --- Save JSON
    pose_vec = pose_parameters[index].detach().cpu().numpy().tolist()
    save_data = {
        'timestamp': timestamp,
        'description': description,
        'pose_parameters': pose_vec,  # 72D: [global_orient(3) + body_pose(69)]
    }
    json_path = os.path.join(save_dir, f'{timestamp}_pose_{index}.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    # --- Save OBJ via SMPL forward (CPU)
    pose = pose_parameters[index].detach().cpu().unsqueeze(0)  # [1, 72]
    betas = torch.zeros(1, 10)  # mean shape; customize if needed
    global_orient = pose[:, :3]
    body_pose = pose[:, 3:]

    with torch.no_grad():
        smpl_out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient)
        vertices = smpl_out.vertices[0].detach().cpu().numpy()
        faces = smpl_model.faces

    obj_path = os.path.join(save_dir, f'{timestamp}_pose_{index}.obj')
    with open(obj_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            # OBJ is 1-indexed
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    return json_path, obj_path


# -----------------------------
# UI
# -----------------------------

st.markdown(
    """
### 💡 Instructions
- Enter a pose description (e.g., a pre-collision pose template).
- Click **Preview (deterministic)** to get a reproducible batch (seed=42), or **Refresh (random)** for diversity.
- Click **Save Pose i** under any card to export that pose as **JSON + OBJ**.
---
"""
)

st.write(f"🛠️ Using model: `{model_path}` | Device: `{device}`")

# Session state to persist the current batch
if 'batch' not in st.session_state:
    st.session_state.batch = None  # {'desc', 'seed', 'imgs', 'poses'}

# Inputs
description = st.text_area(
    "📝 Pose description:",
    placeholder=(
        "The person is standing with feet shoulder-width apart, "
        "left knee slightly bent, and the right arm raised above the head."
    ),
)

save_dir = st.text_input("💾 Save directory", value="output")

# Actions
col1, col2 = st.columns(2)
preview_btn = col1.button("Preview (deterministic)")
refresh_btn = col2.button("Refresh (random)")

# Generate actions
try:
    if preview_btn:
        if not description or description.strip() == "":
            st.warning("Please enter a description before generating.")
        else:
            imgs, poses = generate_and_render(description.strip(), seed=DEFAULT_SEED)
            st.session_state.batch = {
                'desc': description.strip(),
                'seed': DEFAULT_SEED,
                'imgs': imgs,
                'poses': poses,
            }

    if refresh_btn:
        if not description or description.strip() == "":
            st.warning("Please enter a description before generating.")
        else:
            new_seed = fresh_seed()
            last = st.session_state.get("last_seed")
            if last is not None and new_seed == last:
                new_seed = fresh_seed()
            st.session_state["last_seed"] = new_seed

            imgs, poses = generate_and_render(description.strip(), seed=new_seed)
            st.session_state.batch = {
                'desc': description.strip(),
                'seed': new_seed,
                'imgs': imgs,
                'poses': poses,
            }
except Exception as e:
    st.error(f"❌ Generation failed: {e}")

# Hints when description changed but no regen yet
if st.session_state.batch is not None and description and description.strip() != st.session_state.batch['desc']:
    st.info("Description changed. Click **Preview** or **Refresh** to regenerate.")

# Render current batch
batch = st.session_state.batch
if batch is None:
    st.warning("No batch generated yet. Enter a description and click **Preview** or **Refresh**.")
else:
    st.write(f"🖼️ **Generated poses** — seed: `{batch['seed']}` | count: `{n_generate}`")

    # Prepare columns grid
    cols = st.columns(nb_cols)
    msg_box = st.container()

    for i in range(n_generate):
        with cols[i % nb_cols]:
            with st.expander(f"Pose {i}", expanded=True):
                img = batch['imgs'][i] if batch['imgs'] is not None else None
                if img is not None:
                    try:
                        h, w = img.shape[:2]
                        m = min(margin_img, h // 4, w // 4)  # safer crop
                        st.image(img[m: h - m, m: w - m])
                    except Exception:
                        st.image(img)
                else:
                    st.caption("(Rendering disabled — body_model not available)")

                if st.button(f"Save Pose {i}", key=f"save_{i}"):
                    try:
                        json_path, obj_path = save_single_pose(batch['desc'], batch['poses'], i, save_dir=save_dir)
                        with msg_box:
                            st.success(
                                f"✅ **Pose {i} saved!**\n\n- JSON: `{json_path}`\n- OBJ: `{obj_path}`"
                            )
                    except Exception as e:
                        with msg_box:
                            st.error(f"❌ Save failed for Pose {i}: {e}")

# Footer notice if rendering unavailable
if body_model is None:
    st.info(
        "ℹ️ Rendering preview is disabled because `human_body_prior` BodyModel couldn't be initialized. "
        "You can still generate poses and export JSON/OBJ."
    )
