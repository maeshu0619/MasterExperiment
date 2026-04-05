import torch

class OctreeProxyCompressor:
    """
    ネットワーク出力点群に対して、
    final_w を考慮した soft な Octree プロキシ代理圧縮を計算するクラス

    このクラスが行うこと
    - 点群を [B, N, 3] に正規化する
    - final_w を [B, N] に正規化する
    - 三線形補間で soft voxelization を行う
    - 各レベルで soft occupancy から
        - entropy
        - node 数期待値
        - single child 数期待値
      を計算する
    - それらを合成して圧縮後ビットストリーム proxy を返す

    注意
    - final_w は座標を動かさない
    - final_w は各点の voxel への寄与重みとして使う
    - これは実 codec そのものではなく、微分可能な proxy である
    """

    def __init__(
        self,
        base_voxel_size: float,
        levels: int = 3,
        com_bit: float = 1.0,
        com_sin: float = 1.0,
        com_node: float = 1.0,
        eps: float = 1e-6,
    ):
        self.base_voxel_size = float(base_voxel_size)
        self.levels = int(levels)

        # bitstream 合成時の重み
        self.com_bit = float(com_bit)
        self.com_sin = float(com_sin)
        self.com_node = float(com_node)

        self.eps = float(eps)

    # =========================================================
    # 入力正規化
    # =========================================================
    def _normalize_points_to_bn3(self, points: torch.Tensor) -> torch.Tensor:
        """
        入力点群を [B, N, 3] に正規化する

        許可する入力
        - [N, 3]
        - [B, N, 3]
        - [B, 3, N]
        """
        if points.dim() == 2:
            if points.shape[1] != 3:
                raise ValueError("points が2次元の場合は [N, 3] を想定している")
            return points.unsqueeze(0)

        if points.dim() == 3:
            if points.shape[-1] == 3:
                return points
            if points.shape[1] == 3:
                return points.transpose(1, 2).contiguous()

        raise ValueError("points は [N,3], [B,N,3], [B,3,N] のいずれかである必要がある")

    def _normalize_weights_to_bn(
        self,
        points_bn3: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        重みを [B, N] に正規化する

        許可する入力
        - None
        - [N]
        - [B, N]
        - [B, 1, N]
        """
        if weights is None:
            return None

        B, N, _ = points_bn3.shape

        if weights.dim() == 1:
            if B != 1 or weights.shape[0] != N:
                raise ValueError("weights が1次元の場合は points が1バッチで長さNである必要がある")
            return weights.unsqueeze(0)

        if weights.dim() == 2:
            if weights.shape[0] != B or weights.shape[1] != N:
                raise ValueError("weights が2次元の場合は [B, N] を想定している")
            return weights

        if weights.dim() == 3:
            if weights.shape[0] == B and weights.shape[1] == 1 and weights.shape[2] == N:
                return weights.squeeze(1)

        raise ValueError("weights は None, [N], [B,N], [B,1,N] のいずれかである必要がある")

    # =========================================================
    # soft voxelization
    # =========================================================
    def soft_voxelize_sparse(
        self,
        points: torch.Tensor,
        voxel_size: float,
        weights: torch.Tensor | None = None,
    ):
        """
        三線形補間で各点を周囲8 voxel に分配し、
        soft occupancy を作る

        戻り値
        - voxel_indices_list: 各バッチの voxel index [Mi, 3]
        - occ_prob_list     : 各バッチの occupancy probability [Mi]
        """
        points_bn3 = self._normalize_points_to_bn3(points)
        weights_bn = self._normalize_weights_to_bn(points_bn3, weights)

        B, N, _ = points_bn3.shape
        device = points_bn3.device
        dtype = points_bn3.dtype

        corner_offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=device,
            dtype=torch.long,
        )

        voxel_indices_list = []
        occ_prob_list = []

        for b in range(B):
            pts = points_bn3[b]                 # [N, 3]
            pts_scaled = pts / float(voxel_size)

            base = torch.floor(pts_scaled)      # [N, 3]
            frac = pts_scaled - base            # [N, 3]
            base_long = base.long()             # [N, 3]

            wx0 = 1.0 - frac[:, 0]
            wy0 = 1.0 - frac[:, 1]
            wz0 = 1.0 - frac[:, 2]
            wx1 = frac[:, 0]
            wy1 = frac[:, 1]
            wz1 = frac[:, 2]

            if weights_bn is None:
                point_w = torch.ones(N, device=device, dtype=dtype)
            else:
                point_w = weights_bn[b].to(dtype)

            all_voxel_idx = []
            all_contrib = []

            for k in range(8):
                ox, oy, oz = corner_offsets[k]

                cx = wx1 if int(ox.item()) == 1 else wx0
                cy = wy1 if int(oy.item()) == 1 else wy0
                cz = wz1 if int(oz.item()) == 1 else wz0

                # 各点の三線形補間係数
                coeff = cx * cy * cz

                # final_w を occupancy 寄与重みとして掛ける
                coeff = coeff * point_w

                voxel_idx = base_long + corner_offsets[k]  # [N, 3]

                valid = coeff > 1e-12
                if valid.any():
                    all_voxel_idx.append(voxel_idx[valid])
                    all_contrib.append(coeff[valid])

            if len(all_voxel_idx) == 0:
                voxel_indices_list.append(torch.empty((0, 3), device=device, dtype=torch.long))
                occ_prob_list.append(torch.empty((0,), device=device, dtype=dtype))
                continue

            all_voxel_idx = torch.cat(all_voxel_idx, dim=0)   # [M, 3]
            all_contrib = torch.cat(all_contrib, dim=0)       # [M]

            uniq_voxel_idx, inverse = torch.unique(
                all_voxel_idx, dim=0, return_inverse=True
            )

            voxel_mass = torch.zeros(
                uniq_voxel_idx.shape[0], device=device, dtype=dtype
            )
            voxel_mass.index_add_(0, inverse, all_contrib)

            # occupancy proxy
            occ_prob = 1.0 - torch.exp(-voxel_mass.clamp_min(0.0))

            voxel_indices_list.append(uniq_voxel_idx)
            occ_prob_list.append(occ_prob)

        return voxel_indices_list, occ_prob_list

    # =========================================================
    # 情報量 / 木構造統計
    # =========================================================
    def occupancy_bits(
        self,
        occ_prob: torch.Tensor,
        pred_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        occupancy probability から bit proxy を計算する
        pred_prob を省略した場合は pred_prob = occ_prob とする
        """
        if pred_prob is None:
            pred_prob = occ_prob

        pred_prob = pred_prob.clamp(self.eps, 1.0 - self.eps)

        bits = -(
            occ_prob * torch.log2(pred_prob) +
            (1.0 - occ_prob) * torch.log2(1.0 - pred_prob)
        ).sum()

        return bits

    def entropy_bits(self, occ_prob: torch.Tensor) -> torch.Tensor:
        """
        occupancy の自己エントロピー
        """
        p = occ_prob.clamp(self.eps, 1.0 - self.eps)
        ent = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).sum()
        return ent

    def node_count_expectation(self, occ_prob: torch.Tensor) -> torch.Tensor:
        """
        soft occupancy をノード存在確率とみなし、
        ノード数期待値を返す
        """
        return occ_prob.sum()

    def single_child_expectation(
        self,
        voxel_idx: torch.Tensor,
        occ_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        各親ノードについて
        「ちょうど1個の子だけが存在する確率」
        を計算して合計する

        parent = floor(child_idx / 2)
        """
        if voxel_idx.numel() == 0 or occ_prob.numel() == 0:
            return occ_prob.new_tensor(0.0)

        parent_idx = torch.div(voxel_idx, 2, rounding_mode='floor')

        uniq_parent, inverse = torch.unique(parent_idx, dim=0, return_inverse=True)

        total_single = occ_prob.new_tensor(0.0)

        for pid in range(uniq_parent.shape[0]):
            mask = (inverse == pid)
            child_p = occ_prob[mask].clamp(self.eps, 1.0 - self.eps)

            if child_p.numel() == 0:
                continue

            # ちょうど1子のみ存在する確率
            # sum_i p_i * prod_{j!=i}(1-p_j)
            one_prob = child_p.new_tensor(0.0)

            for i in range(child_p.numel()):
                p_i = child_p[i]
                others = torch.cat([child_p[:i], child_p[i+1:]], dim=0)
                if others.numel() == 0:
                    term = p_i
                else:
                    term = p_i * torch.prod(1.0 - others)
                one_prob = one_prob + term

            total_single = total_single + one_prob

        return total_single

    # =========================================================
    # Octree proxy 本体
    # =========================================================
    def octree_proxy_compress(
        self,
        points: torch.Tensor,
        final_w: torch.Tensor | None = None,
    ):
        """
        final_w を考慮した出力点群に対して
        soft な木構造を用いた Octree proxy 圧縮を行う

        戻り値
        - total_bits
        - stats
        """
        points_bn3 = self._normalize_points_to_bn3(points)
        weights_bn = self._normalize_weights_to_bn(points_bn3, final_w)

        total_bits = points_bn3.new_tensor(0.0)
        total_single = points_bn3.new_tensor(0.0)
        total_nodes = points_bn3.new_tensor(0.0)
        total_entropy = points_bn3.new_tensor(0.0)

        current_voxel = float(self.base_voxel_size)

        for _ in range(self.levels):
            voxel_indices_list, occ_prob_list = self.soft_voxelize_sparse(
                points_bn3,
                voxel_size=current_voxel,
                weights=weights_bn,
            )

            if len(occ_prob_list) == 0:
                current_voxel *= 0.5
                continue

            level_entropy = points_bn3.new_tensor(0.0)
            level_nodes = points_bn3.new_tensor(0.0)
            level_single = points_bn3.new_tensor(0.0)

            for voxel_idx_b, occ_prob_b in zip(voxel_indices_list, occ_prob_list):
                if occ_prob_b.numel() == 0:
                    continue

                level_entropy = level_entropy + self.entropy_bits(occ_prob_b)
                level_nodes = level_nodes + self.node_count_expectation(occ_prob_b)
                level_single = level_single + self.single_child_expectation(voxel_idx_b, occ_prob_b)

            total_entropy = total_entropy + level_entropy
            total_nodes = total_nodes + level_nodes
            total_single = total_single + level_single

            current_voxel *= 0.5

        # 圧縮後 bitstream proxy
        total_bits = (
            self.com_bit * total_entropy +
            self.com_sin * total_single +
            self.com_node * total_nodes
        )

        num_points = max(int(points_bn3.shape[1]), 1)
        bpp = total_bits / num_points
        bpn = total_bits / (total_nodes + self.eps)

        stats = {
            "bit": total_bits.detach(),
            "bpp": bpp.detach(),
            "bpn": bpn.detach(),
            "single": total_single.detach(),
            "node": total_nodes.detach(),
            "entropy": total_entropy.detach(),
        }

        return total_bits, stats