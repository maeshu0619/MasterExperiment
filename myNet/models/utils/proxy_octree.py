import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProxyOctreeConfig: # Proxy計算で使うハイパーパラメータをまとめる設定クラスの定義
    max_depth: int = 12 # 何段のOctree Levelまでを見るか
    bbox_pad: float = 1e-6 # bboxの大きさが0になることを防ぐための最小値
    eps: float = 1e-12 # 0除算等を防ぐため
    occ_gain: float = 4.0 # occupancy -> probability 変換の傾き
    occ_threshold: float = 0.5 # soft occupancy の活性化閾値 # Occupancyを「存在している」とみなす際の閾値
    level_decay: float = 0.85 # 各 level の重み # 深いレベルほど重みを少し下げる
    lambda_node_count: float = 0.2 # ノード数 penalty # ノード数ペナルティの重み
    lambda_single_child: float = 0.5 # single-child penalty # 単一子ノードペナルティの重み
    lambda_entropy: float = 1.0 # occupancy entropy 側の重み # エントロピーの重み

class SoftOctreeRateProxy(nn.Module): # 
    def __init__(self, cfg: ProxyOctreeConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        gen_xyz: torch.Tensor,
        final_w: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if gen_xyz.ndim != 3 or gen_xyz.shape[1] != 3: # 引数の点群の形の確認
            raise ValueError("gen_xyz must have shape [B, 3, N]")

        B, _, N = gen_xyz.shape # バッチサイズと点数の取得
        device = gen_xyz.device # 入力が載っているデバイスの取得（CPU/GPU）
        dtype = gen_xyz.dtype # 入力のデータ型の取得

        if final_w is None: # 重みが無い時
            point_w = torch.ones((B, 1, N), device=device, dtype=dtype)
        else: # 重みがある時
            if final_w.ndim == 2:
                point_w = final_w.unsqueeze(1)
            elif final_w.ndim == 3:
                point_w = final_w
            else:
                raise ValueError("final_w must be None, [B,N], or [B,1,N]")
            if point_w.shape != (B, 1, N):
                raise ValueError("final_w must broadcast to [B,1,N]")
            point_w = point_w.clamp(min=0.0, max=1.0)

        pts = gen_xyz.transpose(1, 2).contiguous() # 点群のデータの並び替え
        bbox_min, bbox_max = self._compute_bbox(pts) # 各バッチの点群を囲う立方体bboxの最小・最大座標の計算
        norm_pts = self._normalize_points(pts, bbox_min, bbox_max) # 点群をbbox内で0~1に正規化

        total_entropy = torch.zeros((), device=device, dtype=dtype) # エントロピー初期化
        total_node_count = torch.zeros((), device=device, dtype=dtype) # ノード数初期化
        total_single_child = torch.zeros((), device=device, dtype=dtype) # 単一子ノード初期化
        level_weights = [] # depthにかける重みを入れるリスト
        for depth in range(1, self.cfg.max_depth + 1):
            level_weights.append(self.cfg.level_decay ** (depth - 1)) # 浅いDepthは重く、深いDepthは軽くする
        level_weights = torch.tensor(level_weights, device=device, dtype=dtype) # テンソルで計算するためにTensor変換

        for depth in range(1, self.cfg.max_depth + 1):
            occ = self._soft_occupancy_per_level(
                norm_pts=norm_pts,
                point_w=point_w.squeeze(1),
                depth=depth,
            ) # [B, M] where M = 8^depth # 正規化点群と重みから、そのDepthにおける各Voxel/NodeのSoft Occupancyの計算

            p_occ = torch.sigmoid(
                self.cfg.occ_gain * (occ - self.cfg.occ_threshold)
            ).clamp(min=self.cfg.eps, max=1.0 - self.cfg.eps) # Occupancyの大きさから「そのNodeが存在している確率らしさ」をSigmoidで計算

            y_soft = occ.clamp(0.0, 1.0) # Soft Occupancyを0～1に制限し、存在ラベルのように扱う
            
            level_entropy = (
                -y_soft * torch.log2(p_occ + self.cfg.eps)
                -(1.0 - y_soft) * torch.log2(1.0 - p_occ + self.cfg.eps)
            ).sum() # 各Nodeに関して、存在/非存在のBCEをBit単位で計算し、それをLevel全体で合計

            level_node_count = y_soft.sum() # このLevelにどれだけNodeがあるかをSoftに合計
            level_single_child = self._soft_single_child_count(
                occ=occ,
                depth=depth,
            ) # このLevelにどれだけ単一子ノードらしい構造があるかをSoftに計算

            w_level = level_weights[depth - 1] # このDepthに対応する重みの取得
            total_entropy = total_entropy + w_level * level_entropy # 重み付きでEntropyを全Level合計に足す
            total_node_count = total_node_count + w_level * level_node_count # 重み付きでSoftノード数を全Level合計に足す
            total_single_child = total_single_child + w_level * level_single_child # 重み付きでSoft単一子ノード数を全Level合計に足す
        
        rate_entropy = self.cfg.lambda_entropy * total_entropy # 重み係数との積
        rate_node_count = self.cfg.lambda_node_count * total_node_count # 重み係数との積
        rate_single_child = self.cfg.lambda_single_child * total_single_child # 重み係数との積
        rate_total = rate_entropy + rate_node_count + rate_single_child # 最終的なビットレート予測
        N_float = float(N) # 点群数
        stats = { # 圧縮後情報のまとめ
            "bit": rate_total.detach(),
            "bpp": (rate_total / N_float).detach(),
            "bpn": (rate_total / (total_node_count + 1e-12)).detach(),
            "single": rate_single_child.detach(),
            "node": total_node_count.detach(),
            "entropy": rate_entropy.detach()
        }
        
        return {
            "rate_total": rate_total,
            "rate_entropy": rate_entropy,
            "rate_node_count": rate_node_count,
            "rate_single_child": rate_single_child,
            "soft_node_count": total_node_count,
            "soft_single_child_count": total_single_child
        }, rate_total, stats

    def _compute_bbox( # 点群からbboxを作る補助関数
        self, pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bbox_min = pts.min(dim=1).values # 各バッチで最小座標を求める
        bbox_max = pts.max(dim=1).values # 各バッチで最大座標を求める
        center = 0.5 * (bbox_min + bbox_max) # bboxの中心座標の計算
        extent = (bbox_max - bbox_min).max(dim=1, keepdim=True).values # xyzで最も長い変の長さを取り、立方体の基準にする
        extent = extent.clamp(min=self.cfg.bbox_pad) # bboxが極端に小さい際に、0除算になることを防ぐ
        half = 0.5 * extent # 立方体bboxの半辺長の算出
        cube_min = center - half # 立方体bboxの最小座標
        cube_max = center + half # 立方体bboxの最大座標
        return cube_min, cube_max

    def _normalize_points( # 点群をbbox内で正規化
        self,
        pts: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
    ) -> torch.Tensor:
        denom = (bbox_max - bbox_min).clamp(min=self.cfg.bbox_pad) # 各軸の幅の計算
        norm_pts = (pts - bbox_min.unsqueeze(1)) / denom.unsqueeze(1) # スケーリング
        return norm_pts.clamp(0.0, 1.0 - 1e-7)

    def _soft_occupancy_per_level( # あるDepthにおけるSoft Occupancyの算出関数
        self,
        norm_pts: torch.Tensor,
        point_w: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        B, N, _ = norm_pts.shape # バッチサイズと点数の取得
        G = 2 ** depth # DeothのGrid解像度の算出
        device = norm_pts.device # デバイスの取得
        dtype = norm_pts.dtype # 型の取得
        pos = norm_pts * G # [B, N, 3] # 正規化座標をGrid座標に変換
        base = torch.floor(pos).clamp(min=0, max=G - 1) # 各点が属する基準Voxelの整数Indexを求める
        frac = (pos - base).clamp(0.0, 1.0) # Voxel内での小数部分を取り出し、補完重みに利用
        base = base.long() # Voxel Indexを整数型に変換
        offsets = torch.tensor(
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
        ) # [8,3] # 各点が寄与する8個の隣接するVoxelの相対位置を定義

        occ = torch.zeros((B, G, G, G), device=device, dtype=dtype) # Occupancyを蓄積する3D Gridを初期化
        fx, fy, fz = frac[..., 0], frac[..., 1], frac[..., 2] # xyz各軸の小数部分を分けて取り出す

        for k in range(8):
            ox, oy, oz = offsets[k] # 今回の近傍Voxelのオフセットを取り出す
            nx = (base[..., 0] + ox).clamp(0, G - 1) # x方向の近傍Voxel Indexを計算
            ny = (base[..., 1] + oy).clamp(0, G - 1) # y方向の近傍Voxel Indexを計算
            nz = (base[..., 2] + oz).clamp(0, G - 1) # z方向の近傍Voxel Indexを計算
            wx = fx if ox == 1 else (1.0 - fx) # x方向の補間重みの計算
            wy = fy if oy == 1 else (1.0 - fy) # y方向の補間重みの計算
            wz = fz if oz == 1 else (1.0 - fz) # z方向の補間重みの計算
            wk = wx * wy * wz # [B,N] # 重みの積でTrilinear補間重みにする
            contrib = wk * point_w # [B,N] # 各点のOccupancy寄与にPoint重みを掛ける
            for b in range(B):
                occ[b].index_put_(
                    (nx[b], ny[b], nz[b]),
                    contrib[b],
                    accumulate=True,
                ) # 各点の寄与を対応するVoxelに加算
        occ = 1.0 - torch.exp(-occ) # 単純な加算値を存在強度のような連続Occupancyに変換
        occ = occ.reshape(B, -1) # 3D Gridを1次元に平坦化
        return occ

    def _soft_single_child_count( # Soft Occupancyから単一子ノードらしさの合計を計算
        self,
        occ: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        if depth == 1: # Level1では親子の関係がないので別計算
            return occ.new_zeros(())

        B = occ.shape[0] # バッチサイズの取得
        child_occ = occ.view(B, -1, 8) # [B, num_parents, 8] # 現在のLevelのNodeを、親1つに対して子8個に並び替え

        # s = Σ_k c_k Π_{j!=k}(1-c_j)
        one_minus = (1.0 - child_occ).clamp(min=0.0, max=1.0) # 各子が「存在しないらしさ」を計算
        s = torch.zeros_like(child_occ[..., 0]) # 単一子ノードらしさを蓄積する変数を0で初期化
        for k in range(8):
            prod = torch.ones_like(child_occ[..., 0]) # 他の子が存在しない確率を掛けるための積を1で初期化
            for j in range(8):
                if j == k:
                    continue
                prod = prod * one_minus[..., j] # 「他の子が存在しないらしさ」を全部掛ける
            s = s + child_occ[..., k] * prod # k番目の子だけ存在するらしさを加算
        return s.sum()