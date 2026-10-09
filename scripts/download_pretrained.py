"""Download a pre-0.4 LeRobot policy from the Hub and convert it to the current processor format.

Hub checkpoints such as lerobot/diffusion_pusht still embed normalization layers, which
lerobot 0.6 no longer loads. This wraps lerobot's migration script and fixes a bug in it:
JSON lists (e.g. crop_shape) must become tuples, or saving the config fails.
"""

import argparse
import sys

from lerobot.processor import migrate_policy_normalization as migration

make_policy_config = migration.make_policy_config


def make_policy_config_with_tuples(policy_type, **kwargs):
    kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in kwargs.items()}
    return make_policy_config(policy_type, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default="lerobot/diffusion_pusht")
    parser.add_argument("--output-dir", default="ckpt/diffusion_pusht")
    args = parser.parse_args()

    migration.make_policy_config = make_policy_config_with_tuples
    sys.argv = [sys.argv[0], "--pretrained-path", args.repo_id, "--output-dir", args.output_dir]
    migration.main()


if __name__ == "__main__":
    main()
