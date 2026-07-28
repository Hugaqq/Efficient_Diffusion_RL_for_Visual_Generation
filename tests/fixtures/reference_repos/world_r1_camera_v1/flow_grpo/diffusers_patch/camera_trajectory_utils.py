"""Test double for the World-R1 camera trajectory helper.

Same public prompt-to-trajectory surface as the upstream
``flow_grpo/diffusers_patch/camera_trajectory_utils.py`` (fixture
``world_r1_camera_v1``), but NumPy-only: the upstream file imports ``rp`` and
``torch`` at top level for its optical-flow/noise-wrapping functions, and this
double deliberately drops those functions so contract tests can build typed
camera trajectories without the ``rp`` dependency.  Trajectory math, matrix
string formatting and movement detection are byte-compatible with upstream.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Camera movement definitions
layout_info = {
    "push_in": {"scenenum": 1, "prompt": "push in", "layout_type": "intra"},
    "pull_out": {"scenenum": 1, "prompt": "pull out", "layout_type": "inter"},
    "move_left": {"scenenum": 1, "prompt": "move left", "layout_type": "inter"},
    "move_right": {"scenenum": 1, "prompt": "move right", "layout_type": "inter"},
    "orbit_left": {"scenenum": 1, "prompt": "orbit left", "layout_type": "intra"},
    "orbit_right": {"scenenum": 1, "prompt": "orbit right", "layout_type": "intra"},
    "pan_left": {"scenenum": 1, "prompt": "pan left", "layout_type": "inter"},
    "pan_right": {"scenenum": 1, "prompt": "pan right", "layout_type": "inter"},
    "pull_left": {"scenenum": 1, "prompt": "move left, pull out, then pan left", "layout_type": "inter"},
    "pull_right": {"scenenum": 1, "prompt": "move right, pull out, then pan right", "layout_type": "inter"},
    "fixed": {"scenenum": 1, "prompt": "fixed", "layout_type": "camera fix"}
}

primitive_camera_movements = [
    "push_in",
    "pull_out",
    "move_left",
    "move_right",
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
    "fixed",
]


class TrajectoryGenerator:
    """Generate camera trajectories for different camera movements."""

    def __init__(
        self,
        start_pos: List[float],
        num_frames: int = 81,
        motion_profile: Optional[Dict[str, float]] = None,
    ):
        self.num_frames = num_frames
        self.start_pos = np.array(start_pos, dtype=float)
        self.default_rot = np.eye(3)
        self.motion_profile = motion_profile or {}

    def get_identity_4x4(self) -> np.ndarray:
        return np.eye(4)

    def format_matrix(self, mat: np.ndarray) -> str:
        """
        Format 4x4 matrix to string format.
        Example: "[r00 r10 r20 0] [r01 r11 r21 0] [r02 r12 r22 0] [tx ty tz 1]"
        """
        cols = []
        for i in range(4):
            col = mat[:, i]
            col_str = f"[{col[0]:.6g} {col[1]:.6g} {col[2]:.6g} {col[3]:.6g}]"
            cols.append(col_str)
        return " ".join(cols) + " "

    def get_translation_matrix(self, offset: np.ndarray) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, 3] = offset
        return mat

    def get_rotation_y_matrix(self, angle_deg: float) -> np.ndarray:
        """Rotate around Y axis (Pan)"""
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
        return rot

    def get_rotation_x_matrix(self, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ])
        return rot

    def get_rotation_z_matrix(self, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return rot

    def get_compound_rotation_matrix(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        return (
            self.get_rotation_y_matrix(yaw_deg)
            @ self.get_rotation_x_matrix(pitch_deg)
            @ self.get_rotation_z_matrix(roll_deg)
        )

    def generate(self, layout_name: str) -> Dict[str, str]:
        """Generate camera trajectory for given layout."""
        trajectories = {}

        step_count = max(self.num_frames - 1, 1)
        move_total = float(self.motion_profile.get("move_total", 1800.0))
        zoom_total = float(self.motion_profile.get("zoom_total", 2200.0))
        yaw_total = float(self.motion_profile.get("yaw_total_deg", 200.0))
        pitch_total = float(self.motion_profile.get("pitch_total_deg", 20.0))
        roll_total = float(self.motion_profile.get("roll_total_deg", 12.0))
        orbit_radius = float(self.motion_profile.get("orbit_radius", 600.0))
        orbit_arc = float(self.motion_profile.get("orbit_arc_deg", max(180.0, abs(yaw_total))))

        speed_move = abs(move_total) / step_count
        speed_zoom = abs(zoom_total) / step_count
        speed_rot = abs(yaw_total) / step_count

        for i in range(self.num_frames):
            frame_key = f"frame{i}"
            progress = i / step_count
            pitch_i = pitch_total * progress
            roll_i = roll_total * progress

            # Initialize current frame matrix
            current_mat = np.eye(4)
            current_mat[:3, 3] = self.start_pos

            # Apply transformation based on layout_name
            if layout_name == "fixed":
                pass  # No movement

            elif layout_name == "push_in":
                offset = np.array([0, 0, speed_zoom * i])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "pull_out":
                offset = np.array([0, 0, -speed_zoom * i])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "move_left":
                offset = np.array([-speed_move * i, 0, 0])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "move_right":
                offset = np.array([speed_move * i, 0, 0])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "pan_left":
                rot_mat = self.get_compound_rotation_matrix(-speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            elif layout_name == "pan_right":
                rot_mat = self.get_compound_rotation_matrix(speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            elif layout_name.startswith("orbit"):
                center = self.start_pos + np.array([0, 0, -orbit_radius])
                angle = orbit_arc * progress if "left" in layout_name else -orbit_arc * progress
                rad = np.radians(angle)

                new_x = center[0] + orbit_radius * np.sin(rad)
                new_z = center[2] + orbit_radius * np.cos(rad)
                current_mat[:3, 3] = [new_x, self.start_pos[1], new_z]

                rot_mat = self.get_compound_rotation_matrix(-angle, pitch_i, roll_i)
                current_mat[:3, :3] = rot_mat[:3, :3]

            elif layout_name == "pull_left":
                offset = np.array([-speed_move * i * 0.5, 0, -speed_zoom * i * 0.5])
                current_mat = current_mat @ self.get_translation_matrix(offset)

                rot_mat = self.get_compound_rotation_matrix(-speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            elif layout_name == "pull_right":
                offset = np.array([speed_move * i * 0.5, 0, -speed_zoom * i * 0.5])
                current_mat = current_mat @ self.get_translation_matrix(offset)

                rot_mat = self.get_compound_rotation_matrix(speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            if layout_name in {"push_in", "pull_out", "move_left", "move_right"}:
                rot_sign = 1.0
                if layout_name in {"pull_out", "move_left"}:
                    rot_sign = -1.0
                current_mat[:3, :3] = self.get_compound_rotation_matrix(
                    rot_sign * speed_rot * i,
                    pitch_i,
                    roll_i,
                )[:3, :3]

            trajectories[frame_key] = self.format_matrix(current_mat)

        return trajectories


def sample_motion_profiles(
    trajectory_names: List[str],
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, float]]:
    del rng
    profiles = []
    for name in trajectory_names:
        profile = {
            "move_total": 0.0,
            "zoom_total": 0.0,
            "yaw_total_deg": 0.0,
            "pitch_total_deg": 0.0,
            "roll_total_deg": 0.0,
            "orbit_radius": 520.0,
            "orbit_arc_deg": 0.0,
        }

        if name == "fixed":
            pass
        elif name in {"orbit_left", "orbit_right"}:
            profile["orbit_arc_deg"] = 24.0
            profile["orbit_radius"] = 600.0
        elif name in {"pan_left", "pan_right"}:
            profile["yaw_total_deg"] = 14.0 if name == "pan_left" else -14.0
        elif name == "push_in":
            profile["zoom_total"] = 600.0
        elif name == "pull_out":
            profile["zoom_total"] = 600.0
        elif name == "move_left":
            profile["move_total"] = 220.0
        elif name == "move_right":
            profile["move_total"] = 220.0
        elif name == "pull_left":
            profile["move_total"] = 160.0
            profile["zoom_total"] = 300.0
            profile["yaw_total_deg"] = 8.0
        elif name == "pull_right":
            profile["move_total"] = 160.0
            profile["zoom_total"] = 300.0
            profile["yaw_total_deg"] = -8.0
        else:
            raise ValueError(f"Unsupported trajectory name: {name}")

        profiles.append(profile)

    return profiles


def detect_camera_movements(prompts: Union[str, List[str]]) -> List[str]:
    """
    Detect camera movements in prompts based on layout_info.

    Args:
        prompts: Single prompt string or list of prompts

    Returns:
        List of detected camera movement names (e.g., ['push_in', 'pan_left'])
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    detected_movements = []

    for prompt in prompts:
        prompt_lower = prompt.lower()
        prompt_matches = []

        # Only detect primitive motions automatically.
        # Composite prompts such as "move left, pull out, then pan left"
        # should be expanded into the ordered primitive sequence instead of
        # being matched both as components and as the synthetic "pull_left".
        for movement_name in primitive_camera_movements:
            movement_prompt = layout_info[movement_name]["prompt"].lower()
            pattern = rf"(?<![a-z]){re.escape(movement_prompt)}(?![a-z])"
            for match in re.finditer(pattern, prompt_lower):
                prompt_matches.append(
                    (match.start(), -(match.end() - match.start()), movement_name)
                )

        prompt_matches.sort()

        last_start = None
        for start, _, movement_name in prompt_matches:
            if last_start == start:
                continue
            detected_movements.append(movement_name)
            last_start = start

    return detected_movements


def detect_camera_movements_for_batch(
    prompts: Union[str, List[str]],
    force_camera_movement: Optional[str] = None,
) -> List[str]:
    """
    Detect camera movements for a batch using the same logic as noise wrapping.
    """
    if force_camera_movement is not None:
        # If force_camera_movement is specified, use it directly.
        print(f"Force camera movement: {force_camera_movement}")
        return [force_camera_movement] if force_camera_movement != "fixed" else []

    if isinstance(prompts, list):
        per_prompt_movements = [detect_camera_movements(p) for p in prompts]
        non_empty = [m for m in per_prompt_movements if len(m) > 0]
        detected_movements = non_empty[0] if len(non_empty) > 0 else []

        # Warn if batch contains different camera movements.
        unique_non_empty = {tuple(m) for m in non_empty}
        if len(unique_non_empty) > 1:
            print(
                f"Warning: batch prompts contain multiple camera movement sequences {sorted(unique_non_empty)}; "
                f"using the first detected sequence: {detected_movements}"
            )
        return detected_movements

    return detect_camera_movements(prompts)


def expand_prompts_for_batch(
    prompts: Union[str, List[str]],
    batch_size: Optional[int] = None,
) -> List[str]:
    """
    Expand prompts so there is one prompt string for each batch item.

    This keeps prompt-to-latent / prompt-to-trajectory mapping stable when a
    batch contains multiple different prompts or repeated samples per prompt.
    """
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = list(prompts)

    if batch_size is None or batch_size <= len(prompt_list):
        return prompt_list[:batch_size] if batch_size is not None else prompt_list

    if len(prompt_list) == 1:
        return prompt_list * batch_size

    repeats = (batch_size + len(prompt_list) - 1) // len(prompt_list)
    return (prompt_list * repeats)[:batch_size]


def remove_camera_keywords_from_prompts(prompts: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove all camera movement keywords from prompts.

    This is useful for testing whether noise wrapping is responsible for video motion.
    By removing camera keywords while keeping noise wrapping active, you can isolate
    the effect of noise wrapping alone.

    Args:
        prompts: Single prompt string or list of prompts

    Returns:
        Cleaned prompts with camera keywords removed
    """
    is_single = isinstance(prompts, str)
    if is_single:
        prompts = [prompts]

    cleaned_prompts = []
    for prompt in prompts:
        cleaned = prompt
        # Remove all camera movement keyword phrases
        for info in layout_info.values():
            movement_prompt = info['prompt']
            # Case-insensitive replacement
            cleaned = re.sub(re.escape(movement_prompt), '', cleaned, flags=re.IGNORECASE)

        # Clean up extra spaces
        cleaned = ' '.join(cleaned.split())
        cleaned_prompts.append(cleaned)

    return cleaned_prompts[0] if is_single else cleaned_prompts


def parse_camera_matrix(matrix_str: str) -> np.ndarray:
    """
    Parse camera matrix string to 4x4 numpy array.

    Args:
        matrix_str: Camera matrix in string format

    Returns:
        4x4 numpy array
    """
    cols_str = matrix_str.strip().split('] [')
    cols = []
    for col_str in cols_str:
        col_str = col_str.replace('[', '').replace(']', '').strip()
        values = [float(x) for x in col_str.split()]
        cols.append(values)
    matrix = np.array(cols).T
    return matrix


def concatenate_camera_trajectories(
    trajectory_names: List[str],
    frames_per_trajectory: int = 81,
    start_position: List[float] = [3390, 1380, 240],
    motion_profiles: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, str]:
    """
    Generate and concatenate multiple camera trajectories into one.

    Args:
        trajectory_names: List of trajectory names (e.g., ['push_in', 'pan_left'])
        frames_per_trajectory: Number of frames for each trajectory segment
        start_position: Starting camera position [x, y, z]

    Returns:
        Dictionary mapping frame keys to camera matrix strings
    """
    if not trajectory_names:
        # No camera movements detected, return identity trajectory
        generator = TrajectoryGenerator(start_position, num_frames=frames_per_trajectory)
        return generator.generate("fixed")

    if motion_profiles is None:
        motion_profiles = sample_motion_profiles(trajectory_names)
    if len(motion_profiles) != len(trajectory_names):
        raise ValueError(
            f"motion_profiles length {len(motion_profiles)} must match trajectory_names length {len(trajectory_names)}"
        )

    concatenated_trajectory = {}
    current_start_pose = np.eye(4)
    current_start_pose[:3, 3] = np.array(start_position, dtype=float)

    for traj_idx, (traj_name, motion_profile) in enumerate(zip(trajectory_names, motion_profiles)):
        # Generate the segment in the local camera frame, then compose it onto the
        # full 4x4 pose from the previous segment. This preserves orientation and
        # avoids catastrophic flow spikes at segment boundaries.
        generator = TrajectoryGenerator(
            [0.0, 0.0, 0.0],
            num_frames=frames_per_trajectory,
            motion_profile=motion_profile,
        )
        segment_trajectory = generator.generate(traj_name)
        segment_frame_keys = sorted(segment_trajectory.keys(), key=lambda x: int(x.replace('frame', '')))

        frame_offset = traj_idx * frames_per_trajectory
        for frame_key in segment_frame_keys:
            matrix_str = segment_trajectory[frame_key]
            frame_num = int(frame_key.replace('frame', ''))
            new_frame_key = f"frame{frame_offset + frame_num}"
            local_pose = parse_camera_matrix(matrix_str)
            global_pose = current_start_pose @ local_pose
            concatenated_trajectory[new_frame_key] = TrajectoryGenerator([0.0, 0.0, 0.0]).format_matrix(global_pose)

        last_local_pose = parse_camera_matrix(segment_trajectory[segment_frame_keys[-1]])
        current_start_pose = current_start_pose @ last_local_pose

    return concatenated_trajectory


def get_camera_trajectory_for_prompts(
    prompts: Union[str, List[str]],
    frames_per_trajectory: int = 81,
    force_camera_movement: Optional[str] = None,
    motion_profiles: Optional[List[Dict[str, float]]] = None,
) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Return the camera trajectory and detected movements for prompts.
    """
    detected_movements = detect_camera_movements_for_batch(
        prompts, force_camera_movement=force_camera_movement
    )
    if not detected_movements:
        return None, detected_movements
    segment_frames = max(2, int(math.ceil(frames_per_trajectory / len(detected_movements))))
    trajectory = concatenate_camera_trajectories(
        detected_movements,
        frames_per_trajectory=segment_frames,
        motion_profiles=motion_profiles,
    )
    return trajectory, detected_movements


def get_camera_trajectories_for_batch(
    prompts: Union[str, List[str]],
    batch_size: Optional[int] = None,
    frames_per_trajectory: int = 81,
    force_camera_movement: Optional[str] = None,
) -> Tuple[List[Optional[Dict[str, str]]], List[List[str]], List[str], List[Optional[List[Dict[str, float]]]]]:
    """
    Return one camera trajectory per batch item.

    Unlike ``get_camera_trajectory_for_prompts`` this keeps trajectories aligned
    with individual prompts instead of collapsing a whole batch to the first
    detected camera motion sequence.
    """
    expanded_prompts = expand_prompts_for_batch(prompts, batch_size=batch_size)
    trajectories = []
    detected_movements_batch = []
    motion_profiles_batch = []
    cached_prompt_data: Dict[str, Tuple[Optional[Dict[str, str]], List[str], Optional[List[Dict[str, float]]]]] = {}

    for prompt in expanded_prompts:
        if prompt in cached_prompt_data:
            trajectory, detected_movements, motion_profiles = cached_prompt_data[prompt]
        else:
            detected_movements = detect_camera_movements_for_batch(
                prompt,
                force_camera_movement=force_camera_movement,
            )
            motion_profiles = (
                sample_motion_profiles(detected_movements)
                if detected_movements
                else None
            )
            trajectory, detected_movements = get_camera_trajectory_for_prompts(
                prompt,
                frames_per_trajectory=frames_per_trajectory,
                force_camera_movement=force_camera_movement,
                motion_profiles=motion_profiles,
            )
            cached_prompt_data[prompt] = (trajectory, detected_movements, motion_profiles)
        trajectories.append(trajectory)
        detected_movements_batch.append(detected_movements)
        motion_profiles_batch.append(motion_profiles)

    return trajectories, detected_movements_batch, expanded_prompts, motion_profiles_batch
