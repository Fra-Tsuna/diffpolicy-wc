"""Draw the denoising of each Diffusion Policy action chunk on top of PushT rollout frames."""

import cv2
import numpy as np
import torch
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

    def __init__(self, policy, to_env_action, denoising_frames=15, workspace_size=512):
        """
        Args:
            policy: LeRobot DiffusionPolicy used for the rollout.
            to_env_action: Maps normalized actions of any leading shape to env actions.
            denoising_frames: Number of video frames used to show each denoising process.
            workspace_size: Side of the env workspace in action units (PushT: 512).
        """
        config = policy.config
        self.to_env_action = to_env_action
        self.n_denoising_frames = denoising_frames
        self.workspace_size = workspace_size
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
        trajectories = self._to_pixels(torch.stack(self._samples), frame)
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

    def _to_pixels(self, samples, frame):
        actions = self.to_env_action(samples).float().cpu().numpy()
        height, width = frame.shape[:2]
        return actions * np.array([width, height]) / self.workspace_size

    def _draw(self, frame, points, label, current=None, line_alpha=1.0):
        canvas = frame.copy()
        height, width = frame.shape[:2]
        scale = min(height, width) / 480
        # Pure noise reaches well outside the workspace; pin those points to the frame border.
        points = np.clip(np.round(points), 0, [width - 1, height - 1]).astype(np.int32)
        points = [tuple(int(v) for v in point) for point in points]

        # Faded line and dots for the whole horizon, then bold dots for the executed actions.
        if line_alpha > 0:
            lines = canvas.copy()
            cv2.polylines(
                lines, [np.array(points, dtype=np.int32)], False, (150, 150, 150),
                max(1, round(2 * scale)), cv2.LINE_AA,
            )
            canvas = cv2.addWeighted(lines, line_alpha, canvas, 1 - line_alpha, 0)
        for index, point in enumerate(points):
            if index not in self.executed:
                cv2.circle(canvas, point, round(4 * scale), self.faded_colors[index], -1, cv2.LINE_AA)
        for index in self.executed:
            cv2.circle(canvas, points[index], round(7 * scale), self.colors[index], -1, cv2.LINE_AA)
        if current is not None:
            cv2.circle(canvas, points[current], round(12 * scale), (30, 30, 30), max(1, round(2 * scale)), cv2.LINE_AA)

        cv2.putText(
            canvas, label, (round(14 * scale), round(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
            0.9 * scale, (40, 40, 40), max(1, round(2 * scale)), cv2.LINE_AA,
        )
        return canvas
