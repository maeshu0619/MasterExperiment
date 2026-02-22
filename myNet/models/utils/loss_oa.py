# loss_oa.py

import subprocess
import os
import sys
import pickle
import atexit
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.append(ROOT_DIR)

from compress.OctAttention.compression import *
from compress.OctAttention.encoderTool import *
from compress.OctAttention.networkTool import *
from compress.OctAttention.octAttention import *
from models.utils.utils_loss import *


class OALossHelper:
    def __init__(self, model, qs, writer, file_date):
        self.model = model
        self.qs = qs
        self.writer = writer
        self.file_date = file_date
        self._oa_p = None

        self._start_oa_worker()

    # ===============================
    # Direct encoder call
    # ===============================
    def run_octattention_encoder(self, args, pts):
        return oa_main(
            args,
            pts=pts,
            model=self.model,
            qs=self.qs,
            writer=self.writer,
            file_date=self.file_date
        )

    # ===============================
    # Subprocess (one-shot)
    # ===============================
    def run_octattention_subprocess(self, args, pts):
        env = os.environ.copy()
        env["OA_WORKER"] = "1"

        p = subprocess.Popen(
            [sys.executable, "compress/OctAttention/oa_worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env
        )

        pickle.dump(pts.detach().cpu(), p.stdin)
        pickle.dump(args, p.stdin)
        p.stdin.close()

        com = pickle.load(p.stdout)
        p.stdout.close()
        p.wait()

        return com

    # ===============================
    # Persistent worker
    # ===============================
    def _start_oa_worker(self):
        if self._oa_p is not None and self._oa_p.poll() is None:
            return

        env = os.environ.copy()

        self._oa_p = subprocess.Popen(
            [sys.executable, "compress/OctAttention/oa_worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
            env=env
        )

        atexit.register(self._stop_oa_worker)

    def _stop_oa_worker(self):
        p = self._oa_p
        self._oa_p = None
        if p is None:
            return

        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass

        try:
            p.terminate()
        except Exception:
            pass

    def run_oa_worker(self, args, pts):
        if self._oa_p is None or self._oa_p.poll() is not None:
            self._start_oa_worker()

        p = self._oa_p

        try:
            pickle.dump(pts.detach().cpu(), p.stdin)
            pickle.dump(args, p.stdin)
            p.stdin.flush()

            com = pickle.load(p.stdout)
            return com

        except (BrokenPipeError, EOFError):
            # retry once
            self._stop_oa_worker()
            self._start_oa_worker()
            p = self._oa_p

            pickle.dump(pts.detach().cpu(), p.stdin)
            pickle.dump(args, p.stdin)
            p.stdin.flush()

            com = pickle.load(p.stdout)
            return com