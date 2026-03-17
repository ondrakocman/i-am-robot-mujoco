#!/usr/bin/env python3
"""
offline_replay.py — Deterministic Desktop Replay of Recorded Episodes

Loads a JSON episode file exported from the browser EpisodeRecorder,
replays the actions through native MuJoCo (not WASM), and validates
physics determinism between browser and desktop.

Usage:
    pip install mujoco numpy
    python scripts/offline_replay.py episode.json [--render] [--export-hdf5 out.h5]

Output:
    - Console: per-frame state comparison (browser vs replay)
    - Optional: MuJoCo viewer rendering
    - Optional: HDF5 dataset for IL training
"""

import argparse
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "public", "mujoco", "g1_dex3_tabletop.xml")

# Joint names in actuator order (must match mujocoWorker.js / PhysicsBridge.js)
ACTIVE_JOINT_NAMES = [
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # Left hand (7)
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    # Right hand (7)
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

OBJECT_BODY_NAMES = ["red_cube", "blue_cube", "green_cube"]


def load_episode(path):
    """Load a JSON episode file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Support both single episode and array of episodes
    if isinstance(data, list):
        if len(data) == 0:
            print("Error: empty episode array")
            sys.exit(1)
        print(f"Found {len(data)} episodes, using first one")
        data = data[0]

    info = data.get("info", {})
    frames = data.get("data", [])
    print(f"Episode: {data.get('id', 'unknown')}")
    print(f"  Duration:  {info.get('duration_s', 0):.2f}s")
    print(f"  Frames:    {info.get('num_frames', len(frames))}")
    print(f"  Frequency: {info.get('frequency_hz', 30)} Hz")
    print(f"  Source:    {info.get('source', 'unknown')}")
    return data, frames


def build_action_array(frame):
    """Extract the 28-element action vector from a frame."""
    actions = frame.get("actions", {})
    left_arm = actions.get("left_arm", {}).get("qpos", [0]*7)
    right_arm = actions.get("right_arm", {}).get("qpos", [0]*7)
    left_ee = actions.get("left_ee", {}).get("qpos", [0]*7)
    right_ee = actions.get("right_ee", {}).get("qpos", [0]*7)
    return np.array(left_arm + right_arm + left_ee + right_ee, dtype=np.float64)


def build_state_array(frame):
    """Extract the 28-element state qpos vector from a frame."""
    states = frame.get("states", {})
    left_arm = states.get("left_arm", {}).get("qpos", [0]*7)
    right_arm = states.get("right_arm", {}).get("qpos", [0]*7)
    left_ee = states.get("left_ee", {}).get("qpos", [0]*7)
    right_ee = states.get("right_ee", {}).get("qpos", [0]*7)
    return np.array(left_arm + right_arm + left_ee + right_ee, dtype=np.float64)


def build_object_poses(frame):
    """Extract object poses from a frame."""
    env_state = frame.get("env_state", {})
    objects = env_state.get("objects", [])
    poses = []
    for obj in objects:
        pos = np.array(obj.get("pos", [0, 0, 0]), dtype=np.float64)
        quat = np.array(obj.get("quat", [1, 0, 0, 0]), dtype=np.float64)
        poses.append((pos, quat))
    return poses


def replay(model_path, frames, render=False, verbose=True):
    """
    Replay an episode through native MuJoCo.

    For each frame:
    1. Set ctrl to the recorded actions
    2. Step the simulation
    3. Compare the resulting qpos with the recorded browser state
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Build joint index maps
    joint_qpos_idx = []
    joint_qvel_idx = []
    for name in ACTIVE_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"  [WARN] Joint '{name}' not found in model")
            joint_qpos_idx.append(-1)
            joint_qvel_idx.append(-1)
        else:
            joint_qpos_idx.append(model.jnt_qposadr[jid])
            joint_qvel_idx.append(model.jnt_dofadr[jid])

    object_body_ids = []
    for name in OBJECT_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        object_body_ids.append(bid)

    # Reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Replay metrics
    qpos_errors = []
    object_pos_errors = []
    replay_states = []
    replay_actions = []

    # How many physics substeps per recorded frame
    # Browser runs at ~60Hz physics, recording at 30Hz → 2 substeps per frame
    record_hz = 30
    physics_hz = 60
    substeps = max(1, physics_hz // record_hz)

    # Optional viewer
    viewer = None
    if render:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"  [WARN] Could not launch viewer: {e}")
            viewer = None

    print(f"\nReplaying {len(frames)} frames ({substeps} substeps each)...")

    for i, frame in enumerate(frames):
        action = build_action_array(frame)
        browser_state = build_state_array(frame)
        browser_objects = build_object_poses(frame)

        # Set actuator controls
        for j in range(28):
            data.ctrl[j] = action[j]

        # Step physics
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        # Read solved state
        replay_qpos = np.zeros(28)
        for j in range(28):
            idx = joint_qpos_idx[j]
            if idx >= 0:
                replay_qpos[j] = data.qpos[idx]

        # Compare with browser state
        qpos_err = np.abs(replay_qpos - browser_state)
        qpos_errors.append(qpos_err)

        # Compare object positions
        for o, bid in enumerate(object_body_ids):
            if bid < 0 or o >= len(browser_objects):
                continue
            replay_pos = data.xpos[bid].copy()
            browser_pos = browser_objects[o][0]
            pos_err = np.linalg.norm(replay_pos - browser_pos)
            object_pos_errors.append(pos_err)

        # Store for export
        replay_states.append(replay_qpos.copy())
        replay_actions.append(action.copy())

        if verbose and (i % 30 == 0 or i == len(frames) - 1):
            max_err = qpos_err.max()
            mean_err = qpos_err.mean()
            print(f"  Frame {i:4d}/{len(frames)}: "
                  f"max_err={max_err:.6f} rad, mean_err={mean_err:.6f} rad")

        if viewer is not None:
            viewer.sync()

    # Summary
    all_errors = np.array(qpos_errors)
    print(f"\n=== Replay Summary ===")
    print(f"  Frames replayed: {len(frames)}")
    print(f"  Joint qpos error (max):  {all_errors.max():.6f} rad")
    print(f"  Joint qpos error (mean): {all_errors.mean():.6f} rad")
    print(f"  Joint qpos error (p95):  {np.percentile(all_errors, 95):.6f} rad")
    if object_pos_errors:
        obj_errs = np.array(object_pos_errors)
        print(f"  Object pos error (max):  {obj_errs.max():.6f} m")
        print(f"  Object pos error (mean): {obj_errs.mean():.6f} m")

    if viewer is not None:
        viewer.close()

    return {
        "states": np.array(replay_states),
        "actions": np.array(replay_actions),
        "qpos_errors": all_errors,
    }


def export_hdf5(output_path, episode_data, replay_result, frames):
    """Export replay data as HDF5 for IL training pipelines."""
    try:
        import h5py
    except ImportError:
        print("  [ERROR] h5py not installed. Run: pip install h5py")
        return

    info = episode_data.get("info", {})

    with h5py.File(output_path, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["episode_id"] = episode_data.get("id", "unknown")
        meta.attrs["duration_s"] = info.get("duration_s", 0)
        meta.attrs["num_frames"] = len(frames)
        meta.attrs["frequency_hz"] = info.get("frequency_hz", 30)
        meta.attrs["source"] = info.get("source", "unknown")

        # Actions and states
        f.create_dataset("actions", data=replay_result["actions"], compression="gzip")
        f.create_dataset("states", data=replay_result["states"], compression="gzip")

        # Timestamps
        timestamps = np.array([frame.get("timestamp", 0) for frame in frames])
        f.create_dataset("timestamps", data=timestamps)

        # Object poses
        obj_positions = []
        obj_quats = []
        for frame in frames:
            poses = build_object_poses(frame)
            if poses:
                obj_positions.append(np.array([p[0] for p in poses]))
                obj_quats.append(np.array([p[1] for p in poses]))

        if obj_positions:
            f.create_dataset("object_positions", data=np.array(obj_positions), compression="gzip")
            f.create_dataset("object_quaternions", data=np.array(obj_quats), compression="gzip")

        # Determinism error
        f.create_dataset("replay_qpos_errors", data=replay_result["qpos_errors"], compression="gzip")

    print(f"  Exported HDF5: {output_path}")
    print(f"  Actions shape: {replay_result['actions'].shape}")
    print(f"  States shape:  {replay_result['states'].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline deterministic replay of MuJoCo WASM episodes"
    )
    parser.add_argument("episode", help="Path to episode JSON file")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to MJCF model XML (default: public/mujoco/g1_dex3_tabletop.xml)")
    parser.add_argument("--render", action="store_true",
                        help="Launch MuJoCo viewer during replay")
    parser.add_argument("--export-hdf5", metavar="PATH",
                        help="Export replay data as HDF5 file")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    args = parser.parse_args()

    if not os.path.exists(args.episode):
        print(f"Error: episode file not found: {args.episode}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}")
        print("  Run 'python scripts/prepare_model.py' first.")
        sys.exit(1)

    episode_data, frames = load_episode(args.episode)

    if len(frames) == 0:
        print("Error: no frames in episode")
        sys.exit(1)

    result = replay(args.model, frames, render=args.render, verbose=not args.quiet)

    if args.export_hdf5:
        export_hdf5(args.export_hdf5, episode_data, result, frames)

    print("\nDone!")


if __name__ == "__main__":
    main()
