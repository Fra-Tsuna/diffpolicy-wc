from functools import partial
from pathlib import Path


def save_fig(fig, fig_name, fig_dir, fig_fmt="png", fig_size=(6.4, 4), dpi=300, transparent_png=False):
    """Save an existing figure without introducing task-specific drawing code."""
    destination = Path(fig_dir) / fig_fmt
    destination.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(fig_size, forward=False)
    path = destination / f"{fig_name}.{fig_fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=dpi, transparent=transparent_png if fig_fmt == "png" else False)
    return path


def setup_savefig(res_path, fig_fmt="png", dpi=300, transparent_png=False):
    return partial(save_fig, fig_dir=res_path, fig_fmt=fig_fmt, dpi=dpi, transparent_png=transparent_png)


def mujoco_physics(env):
    """The dm_control physics of a single-env MuJoCo simulator such as gym-aloha, or None (e.g. PushT)."""
    return getattr(getattr(env.envs[0].unwrapped, "_env", None), "physics", None)


def make_video_renderer(env, camera):
    """Return a function rendering the current video frame: the env's own render, or a named MuJoCo camera."""
    if camera is None:
        return lambda: env.call("render")[0]
    # gym-aloha renders only "top"; its dm_control physics renders any camera of the scene.
    sim = env.envs[0].unwrapped
    physics = mujoco_physics(env)
    if physics is None:
        raise ValueError(f"video_camera needs a MuJoCo simulator such as ALOHA, got {type(sim).__name__}.")
    cameras = [physics.model.id2name(index, "camera") for index in range(physics.model.ncam)]
    if camera not in cameras:
        raise ValueError(f"Unknown video_camera {camera!r}; the scene has {cameras}.")
    return lambda: physics.render(height=sim.visualization_height, width=sim.visualization_width, camera_id=camera)
