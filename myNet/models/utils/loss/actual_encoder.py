import os
import shutil
import sys
import tempfile

import numpy as np
import torch


class _OctAttentionActualEncoder:
    """
    Run the same OctAttention path as compress/octree/OctAttention/encoder.py.

    This path intentionally leaves autograd: generated points are written as a
    temporary PLY, converted to the OctAttention mat/octree format, and encoded
    by the pretrained entropy model. The returned bit count is therefore a real
    measurement, not a differentiable quantity.
    """
    def __init__(self, args, writer=None):
        self.args = args
        self.writer = writer
        self.qs = float(getattr(args, "qs", 2.0))
        self.actualcode = bool(getattr(args, "octattention_actualcode", False))
        self.tmp_root = getattr(args, "octattention_tmp_dir", "")
        self._loaded = False
        self._model = None
        self._data_prepare = None
        self._write_ply_data = None
        self._oa_bptt = int(getattr(args, "bptt", 1024))

    def _lazy_init(self):
        if self._loaded:
            return

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        project_root = os.path.join(repo_root, "myNet")
        if project_root not in sys.path:
            sys.path.append(project_root)
        oa_dir = os.path.join(repo_root, "compress", "octree", "OctAttention")
        if oa_dir not in sys.path:
            sys.path.append(oa_dir)

        from Preparedata.data import dataPrepare
        from networkTool import levelNumK
        from pt import write_ply_data
        from models.utils.compression.proxy_octree import _OctAttentionTeacherModel

        ckpt_path = getattr(self.args, "octattention_ckpt", "")
        if not ckpt_path:
            ckpt_path = os.path.join(oa_dir, "modelsave", "obj", "encoder_epoch_00800093.pth")
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(os.path.join(repo_root, ckpt_path))

        save_dic = torch.load(ckpt_path, map_location="cpu")
        state_dict = save_dic["encoder"] if "encoder" in save_dic else save_dic

        oa_model = _OctAttentionTeacherModel(max_octree_level=12)
        oa_model.load_state_dict(state_dict)
        oa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oa_model.to(oa_device)
        oa_model.eval()
        for param in oa_model.parameters():
            param.requires_grad_(False)

        self._model = oa_model
        self._data_prepare = dataPrepare
        self._level_num_k = levelNumK
        self._write_ply_data = write_ply_data
        self._oa_device = oa_device
        self._loaded = True

        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(f"OctAttention actual encoder loaded: {ckpt_path}")

    def _make_tmp_dir(self):
        root = self.tmp_root
        if not root:
            root = "/dev/shm/mynet_octattention_actual" if os.path.isdir("/dev/shm") else None
        if root:
            os.makedirs(root, exist_ok=True)
            return tempfile.mkdtemp(prefix="oa_actual_", dir=root)
        return tempfile.mkdtemp(prefix="oa_actual_")

    def encode_bits(self, pts_3n):
        self._lazy_init()
        pts_np = (
            pts_3n.detach()
            .transpose(0, 1)
            .contiguous()
            .to("cpu")
            .numpy()
            .astype(np.float32, copy=False)
        )

        tmp_dir = self._make_tmp_dir()
        try:
            ply_path = os.path.join(tmp_dir, "input.ply")
            mat_dir = os.path.join(tmp_dir, "mat")
            bin_path = os.path.join(tmp_dir, "encoded.bin")
            self._write_ply_data(ply_path, pts_np)
            mat_file, _dq_pt, _ref_pt = self._data_prepare(
                ply_path,
                saveMatDir=mat_dir,
                qs=self.qs,
                ptNamePrefix="",
                rotation=False,
            )

            import h5py as _h5py

            mat = None
            try:
                mat = _h5py.File(mat_file, "r")
                cell = mat["patchFile"]
                ref = cell[0, 0]
                data_arr = np.array(mat[ref])
                oct_data_seq = np.transpose(data_arr).astype(np.int32)[:, -self._level_num_k:, 0:6]
            finally:
                if mat is not None:
                    mat.close()

            binsz, oct_len = self._compress_oct_seq(oct_data_seq, bin_path)
            single_count = self._single_child_count(oct_data_seq)
            return {
                "bit": float(binsz),
                "bpp": float(binsz) / max(float(pts_3n.shape[-1]), 1.0),
                "bpn": float(binsz) / max(float(oct_len), 1.0),
                "single": float(single_count),
                "node": float(oct_len),
                "point_count": int(pts_3n.shape[-1]),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _single_child_count(oct_data_seq):
        oct_code = oct_data_seq[:, -1, 0].astype(np.int32)
        pop = (
            (oct_code & 1) + ((oct_code >> 1) & 1) +
            ((oct_code >> 2) & 1) + ((oct_code >> 3) & 1) +
            ((oct_code >> 4) & 1) + ((oct_code >> 5) & 1) +
            ((oct_code >> 6) & 1) + ((oct_code >> 7) & 1)
        )
        return int((pop == 1).sum())

    def _generate_square_subsequent_mask(self, size):
        return torch.triu(
            torch.full((size, size), float("-inf"), device=self._oa_device),
            diagonal=1,
        )

    @staticmethod
    def _batchify(oct_seq, bptt, oct_len):
        oct_seq = oct_seq.copy()
        oct_seq[:-1, 0:-1, :] = oct_seq[1:, 0:-1, :]
        oct_seq[:-1, -1, 1:3] = oct_seq[1:, -1, 1:3]
        oct_seq[:, :, 0] = oct_seq[:, :, 0] - 1
        pad_len = bptt
        padded = np.zeros((bptt + oct_len + pad_len, *oct_seq.shape[1:]), dtype=oct_seq.dtype)
        padded[bptt:bptt + oct_len] = oct_seq
        oct_seq_t = torch.from_numpy(padded).long()

        data_id = torch.full((bptt + oct_len + pad_len,), -1, dtype=torch.long)
        data_id[bptt:bptt + oct_len] = torch.arange(oct_len, dtype=torch.long)
        return data_id.unsqueeze(1), oct_seq_t.unsqueeze(1)

    @staticmethod
    def _estimate_bits(pro_bit, oct_seq, level_id):
        oct_values = oct_seq.astype(np.int64).reshape(-1) - 1
        level_values = np.asarray(level_id, dtype=np.int64).reshape(-1)

        prob_hit = np.take_along_axis(pro_bit[:oct_values.shape[0]], oct_values[:, None], axis=1).squeeze(1)
        bit_each = -np.log2(prob_hit + 1e-7)
        bit = float(bit_each.sum())

        level_change = np.empty(level_values.shape[0], dtype=bool)
        level_change[0] = level_values[0] != 1
        level_change[1:] = level_values[1:] != level_values[:-1]
        level_change_idx = np.flatnonzero(level_change)

        binsz_list = np.concatenate([np.cumsum(bit_each)[level_change_idx], np.array([bit])])
        oct_num_list = np.concatenate([level_change_idx + 1, np.array([oct_values.shape[0]])])
        return bit, binsz_list, oct_num_list

    def _compress_oct_seq(self, oct_data_seq, output_file):
        from networkTool import MAX_OCTREE_LEVEL

        level_id = oct_data_seq[:, -1, 1].copy()
        oct_data_seq = oct_data_seq.copy()
        if level_id.max() > MAX_OCTREE_LEVEL:
            level_id = np.minimum(level_id, MAX_OCTREE_LEVEL)

        oct_seq = oct_data_seq[:, -1:, 0].astype(int)
        oct_len = len(oct_seq)
        bptt_eff = min(self._oa_bptt, oct_len - 1)
        if bptt_eff < 32:
            raise ValueError(f"oct_len too small for OctAttention: oct_len={oct_len}")

        data_id, padded_data = self._batchify(oct_data_seq, bptt_eff, oct_len)
        pading_length = padded_data.shape[0]
        src_mask = self._generate_square_subsequent_mask(bptt_eff)
        pro_bit_chunks = [] if self.actualcode else None
        oct_values = torch.from_numpy(oct_seq.astype(np.int64).reshape(-1) - 1)
        processed = 0
        total_bits = 0.0

        with torch.inference_mode():
            for i in range(0, pading_length - bptt_eff, bptt_eff):
                inp = padded_data[i:i + bptt_eff].to(device=self._oa_device, non_blocking=True)
                node_id = data_id[i + 1:i + bptt_eff + 1].reshape(-1)
                valid_mask = node_id >= 0
                if not valid_mask.any():
                    continue
                output = self._model(inp, src_mask, [])
                output = output.reshape(-1, 255)
                prob = torch.softmax(output, dim=1)
                valid_prob = prob[valid_mask]
                valid_count = min(int(valid_prob.shape[0]), oct_len - processed)
                if valid_count <= 0:
                    break
                valid_prob = valid_prob[:valid_count]
                target = oct_values[processed:processed + valid_count].to(device=self._oa_device, non_blocking=True)
                prob_hit = valid_prob.gather(1, target.view(-1, 1)).squeeze(1)
                total_bits += float((-torch.log2(prob_hit + 1e-7)).sum().detach().cpu())
                if pro_bit_chunks is not None:
                    pro_bit_chunks.append(valid_prob.detach().cpu().numpy())
                processed += valid_count

        if processed <= 0:
            raise ValueError(f"OctAttention produced no valid probability rows: oct_len={oct_len}")
        if processed < oct_len and self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                f"OctAttention warning: probability rows shorter than octree length "
                f"({processed}/{oct_len}); using available rows for bit estimate."
            )
        binsz = total_bits

        if self.actualcode:
            import numpyAc
            if not pro_bit_chunks:
                raise ValueError(f"OctAttention produced no probability chunks for arithmetic coding: oct_len={oct_len}")
            pro_bit = np.vstack(pro_bit_chunks)[:oct_len]
            binsz, _binsz_list, _oct_num_list = self._estimate_bits(pro_bit, oct_seq, level_id)
            codec = numpyAc.arithmeticCoding()
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            _, binsz = codec.encode(pro_bit[:oct_len, :], oct_seq.astype(np.int16).squeeze(-1) - 1, output_file)
            del pro_bit
        del data_id, padded_data, src_mask
        if self._oa_device.type == "cuda":
            torch.cuda.empty_cache()
        return float(binsz), int(oct_len)
