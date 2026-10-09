import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_two_process_ddp_checkpointing(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo", "PYTHONPATH": str(ROOT)}
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--master_addr=127.0.0.1", f"--master_port={port}",
        str(ROOT / "tests/distributed_checkpoint_worker.py"), str(tmp_path / "checkpoints"),
    ], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=40)
    output = result.stdout + result.stderr
    if "DDP_UNAVAILABLE:" in output:
        pytest.skip("This runtime blocks Gloo socket operations; run the DDP test on an unrestricted host.")
    assert result.returncode == 0, output
    assert "DDP_CHECKPOINT_PASS" in output
