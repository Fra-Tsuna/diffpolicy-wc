"""Draw the denoising of each Diffusion Policy action chunk on top of rollout frames (PushT and ALOHA)."""

import cv2
import mujoco
import numpy as np
import torch
from dm_control.mujoco.engine import Camera
from matplotlib import colormaps


class DiffusionOverlay:
    """Record every denoising process of a DiffusionPolicy and draw it on rendered frames.

    The policy's noise scheduler step() is wrapped to capture the sample after each denoising step,
    so the policy code itself is untouched. Use it as:

        action = policy.select_action(batch)
        frames.extend(overlay.denoising_frames(frame))  # empty unless a new chunk was just sampled
        ... env.step(action) ...
        frames.append(overlay.draw_plan(new_frame))
    """

    def __init__(self, policy, to_pixels, denoising_frames=15):
        """
        Args:
            policy: LeRobot DiffusionPolicy used for the rollout.
            to_pixels: Maps normalized action samples (S, horizon, action_dim) and a frame shape to pixel
                coordinates (S, horizon, K, 2) of K trajectories, e.g. PushtPixels or AlohaGripperPixels.
            denoising_frames: Number of video frames used to show each denoising process.
        """
        config = policy.config
        self.to_pixels = to_pixels
        self.n_denoising_frames = denoising_frames
        # Horizon indices of the actions that are executed (see DiffusionPolicy.select_action).
        start = config.n_obs_steps - 1
        self.executed = range(start, start + config.n_action_steps)
        colors = colormaps["plasma"](np.linspace(0.0, 0.85, config.horizon))[:, :3] * 255
        self.colors = [tuple(int(c) for c in color) for color in colors]
        self.faded_colors = [tuple(int(0.35 * c + 0.65 * 255) for c in color) for color in colors]

        self.plan = None
        self.steps_executed = 0
        self._samples = []
        self._new_chunk = False
        self._record(policy.diffusion.noise_scheduler)

    def _record(self, scheduler):
        step = scheduler.step

        def recording_step(model_output, timestep, sample, *args, **kwargs):
            output = step(model_output, timestep, sample, *args, **kwargs)
            # The first timestep starts a new chunk; keep the initial noise too.
            if timestep == scheduler.timesteps[0]:
                self._samples = [sample[0].clone()]
                self._new_chunk = True
            self._samples.append(output.prev_sample[0].clone())
            return output

        scheduler.step = recording_step

    def denoising_frames(self, frame):
        """Animate the denoising of a newly sampled chunk on `frame`; empty if no chunk was sampled."""
        if not self._new_chunk:
            return []
        self._new_chunk = False
        trajectories = self.to_pixels(torch.stack(self._samples), frame.shape)
        self.plan = trajectories[-1]
        self.steps_executed = 0

        n_steps = len(trajectories) - 1
        # Most structure appears in the last denoising steps, so sample frames more densely there.
        progress = 1 - (1 - np.linspace(0, 1, self.n_denoising_frames)) ** 2
        indices = np.unique(np.round(progress * n_steps).astype(int))
        # The connecting line fades in as the trajectory takes shape.
        return [
            self._draw(
                frame, trajectories[i], f"denoising {i}/{n_steps}" if i else "noise",
                line_alpha=(i / n_steps) ** 2,
            )
            for i in indices
        ]

    def draw_plan(self, frame):
        """Draw the latest plan, marking the action the agent has just executed."""
        if self.plan is None:
            return frame
        self.steps_executed += 1
        current = self.executed[min(self.steps_executed, len(self.executed)) - 1]
        label = f"executing {self.steps_executed}/{len(self.executed)}"
        return self._draw(frame, self.plan, label, current=current)

    def _draw(self, frame, points, label, current=None, line_alpha=1.0):
        canvas = frame.copy()
        height, width = frame.shape[:2]
        scale = min(height, width) / 480
        # Pure noise reaches well outside the workspace; pin those points to the frame border.
        points = np.clip(np.round(points), 0, [width - 1, height - 1]).astype(np.int32)
        # One trajectory per tracked point (PushT: the agent; ALOHA: each gripper), shape (K, horizon, 2).
        trajectories = [[tuple(int(v) for v in point) for point in trajectory] for trajectory in points.swapaxes(0, 1)]

        # Faded lines and dots for the whole horizon, then bold dots for the executed actions.
        if line_alpha > 0:
            lines = canvas.copy()
            cv2.polylines(
                lines, [np.array(trajectory, dtype=np.int32) for trajectory in trajectories], False,
                (150, 150, 150), max(1, round(2 * scale)), cv2.LINE_AA,
            )
            canvas = cv2.addWeighted(lines, line_alpha, canvas, 1 - line_alpha, 0)
        for trajectory in trajectories:
            for index, point in enumerate(trajectory):
                if index not in self.executed:
                    cv2.circle(canvas, point, round(4 * scale), self.faded_colors[index], -1, cv2.LINE_AA)
        for trajectory in trajectories:
            for index in self.executed:
                cv2.circle(canvas, trajectory[index], round(7 * scale), self.colors[index], -1, cv2.LINE_AA)
            if current is not None:
                cv2.circle(canvas, trajectory[current], round(12 * scale), (30, 30, 30), max(1, round(2 * scale)),
                           cv2.LINE_AA)

        cv2.putText(
            canvas, label, (round(14 * scale), round(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * scale, (40, 40, 40), max(1, round(2 * scale)), cv2.LINE_AA,
        )
        return canvas


class PushtPixels:
    """PushT actions are agent target positions in a square workspace; scale them to the frame (K = 1)."""

    def __init__(self, to_env_action, workspace_size=512):
        """
        Args:
            to_env_action: Maps normalized actions of any leading shape to env actions.
            workspace_size: Side of the env workspace in action units (PushT: 512).
        """
        self.to_env_action = to_env_action
        self.workspace_size = workspace_size

    def __call__(self, samples, frame_shape):
        actions = self.to_env_action(samples).float().cpu().numpy()
        height, width = frame_shape[:2]
        return (actions * np.array([width, height]) / self.workspace_size)[:, :, None, :]


class AlohaGripperPixels:
    """ALOHA actions are joint targets; draw where they put each gripper's fingertips (K = 2: left, right).

    Each action's arm joint targets are applied to a copy of the simulator state, forward kinematics gives the
    fingertip positions (the camera_focus body between the fingers), and the video camera projects them.
    """

    # gym-aloha's 14-D action: left arm joints, left gripper, right arm joints, right gripper. The arm joints are
    # the first six position actuators of each arm, and the gripper opening barely moves the fingertip point.
    ARM_ACTIONS = (slice(0, 6), slice(7, 13))
    ARMS = ("vx300s_left", "vx300s_right")

    def __init__(self, to_env_action, physics, camera):
        """
        Args:
            to_env_action: Maps normalized actions of any leading shape to env actions.
            physics: dm_control Physics of the ALOHA scene, read for the current state and camera pose.
            camera: Name of the scene camera the video is rendered from.
        """
        self.to_env_action = to_env_action
        self.physics = physics
        self.camera = camera
        model = physics.model.ptr
        self.data = mujoco.MjData(model)
        self.qpos_indices = [
            [model.jnt_qposadr[model.actuator_trnid[actuator, 0]] for actuator in range(start, start + 6)]
            for start in (0, 8)
        ]
        self.bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}/camera_focus") for arm in self.ARMS]
        self.cameras = {}

    def __call__(self, samples, frame_shape):
        actions = self.to_env_action(samples).float().cpu().numpy()
        height, width = frame_shape[:2]
        model = self.physics.model.ptr
        self.data.qpos[:] = self.physics.data.qpos
        points = np.empty((*actions.shape[:2], len(self.ARMS), 3))
        for index in np.ndindex(actions.shape[:2]):
            for arm, (columns, qpos) in enumerate(zip(self.ARM_ACTIONS, self.qpos_indices)):
                self.data.qpos[qpos] = actions[index][columns]
            mujoco.mj_kinematics(model, self.data)
            points[index] = self.data.xpos[self.bodies]

        # Homogeneous pinhole projection with the video camera's current pose (matrix reads it on every access).
        if (height, width) not in self.cameras:
            self.cameras[height, width] = Camera(self.physics, height=height, width=width, camera_id=self.camera)
        matrix = self.cameras[height, width].matrix
        projected = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1) @ matrix.T
        # dm_control's camera looks along its -z axis, so points in front of it have a negative third coordinate.
        # Points behind it (possible for pure noise) are pushed off-frame and pinned to the border.
        scale = projected[..., 2:]
        in_front = scale < -1e-6
        return np.where(in_front, projected[..., :2] / np.where(in_front, scale, -1.0), -1e6)
