import argparse
from cfgs.utils import str2bool

pretrained_date = "20260416"
pretrained_time = "172956"
method_loss = "cd"
model_name = "best"

def parse_pugan_args(parser, file_day, file_time, dir_input):
    """基本情報"""
    parser.add_argument('--date', default=f'{file_day}', type=str, help='日付')
    parser.add_argument('--time', default=f'{file_time}', type=str, help='時刻')
    parser.add_argument('--input_dir', default=f'../data/train/{dir_input}', type=str, help='入力点群データのフォルダパス')
    parser.add_argument('--cpu', action='store_true', help='GPUを使わずCPUで学習するかどうか')
    parser.add_argument('--print_rate', default=1, type=int, help='ログ出力頻度（1なら毎ステップ、0なら最初と最後のみ）')

    """ネットワーク条件"""
    # Network
    parser.add_argument('--encoder_0grad', default=True, type=str2bool, help='Encoderを学習対象にするかどうか')
    parser.add_argument('--prune', default=True, type=str2bool, help='Pruning Moduleを使用するか')
    parser.add_argument('--add', default=True, type=str2bool, help='Adding Moduleを使用するか')
    parser.add_argument('--disp', default=True, type=str2bool, help='Displacement Moduleを使用するか')
    # Encoder, FP Module
    parser.add_argument('--encoder_dim', default=64, type=int, help='各dense blockにおける特徴次元数（入力/出力）')
    parser.add_argument('--out_dim', default=64, type=int, help='各dense blockにおける特徴次元数（入力/出力）')
    parser.add_argument('--local_feat_dim', default=192, type=int, help='局所幾何特徴の次元（Analyzer用）')
    parser.add_argument('--fused_feat_dim', default=64, type=int, help='Encoderで統合された特徴の次元')
    parser.add_argument('--fp_mlp_channels', nargs='+', type=int, default=[128, 64], help='Feature PropagationのMLPの隠れ層チャネル数')
    # Pruning Module
    parser.add_argument('--prune_hidden_dim', default=64, type=int, help='Pruning用MLPの隠れ層次元')
    parser.add_argument('--prune_d_high_is_inlier', default=True, type=str2bool, help='dスコアが高いほどインライアとみなすか')
    parser.add_argument('--prune_robust_c', default=2.0, type=float, help='ロバスト重みのパラメータ')
    parser.add_argument('--prune_ratio_min', default=0.85, type=float, help='保持割合の最小値')
    parser.add_argument('--prune_ratio_max', default=0.995, type=float, help='保持割合の最大値')
    parser.add_argument('--prune_use_label_count', default=True, type=str2bool, help='外れ点ラベルの数をPruningの保持点数教師として使うか')
    # Adding Module
    parser.add_argument('--add_hidden_dim', default=64, type=int, help='Add ModuleのMLP隠れ層次元')
    parser.add_argument('--add_fit_ref_max', default=4096, type=int, help='フィッティング損失計算で参照する最大点数')
    parser.add_argument('--add_attr', default=True, type=str2bool, help='追加点に色情報を付与するか')
    parser.add_argument('--add_color', default='Red', type=str, help='追加点の色')
    parser.add_argument('--add_th', default=0.5, type=float, help='追加判定のしきい値')
    parser.add_argument('--target_add_ratio', default=0.01, type=float, help='目標とする追加割合')
    parser.add_argument('--max_add_ratio', default=0.03, type=float, help='追加割合の最大値')
    parser.add_argument('--add_oct_weight', default=1.0, type=float, help='Octreeグリッドへのスナップの重み')
    parser.add_argument('--lambda_sparse', default=1e-3, type=float, help='スパース性の正則化係数')
    # Displacement Module
    parser.add_argument('--disp_hidden_dim', default=64, type=int, help='Displacement用MLPの隠れ層次元')
    parser.add_argument('--max_disp_offset', default=0.002, type=float, help='最大移動距離（メートル）')
    parser.add_argument('--disp_num_blocks',default=4,type=int,help='残差ブロックの数')
    parser.add_argument('--disp_num_steps',default=1,type=int,help='反復更新回数')
    parser.add_argument('--disp_step_size',default=1.0,type=float,help='移動更新のステップサイズ')
    parser.add_argument('--disp_step_decay',default=0.95,type=float,help='ステップサイズの減衰率')
    parser.add_argument('--disp_grad_clip',default=10.0,type=float,help='勾配クリッピング値')
    parser.add_argument('--target_disp_ratio', default=0.25, type=float, help='移動する点の目標割合')
    parser.add_argument('--disp_use_gate',default=True,type=str2bool,help='ゲーティング機構を使うか')
    parser.add_argument('--disp_reg_weight', default=1e-4, type=float, help='移動量の正則化係数')
    parser.add_argument('--disp_ratio_weight', default=1e-4, type=float, help='移動割合の正則化係数')
    parser.add_argument('--disp_snap_strength', default=0.35, type=float, help='Octreeグリッドへのスナップ強度')
    parser.add_argument('--disp_mag_bias', default=-1.0, type=float, help='移動量の初期バイアス')
    parser.add_argument('--disp_gate_bias', default=0.0, type=float, help='ゲートの初期バイアス')
    parser.add_argument('--disp_soft_match_tau', default=0.05, type=float, help='soft top-kの温度パラメータ')
    # Analyzer
    parser.add_argument('--octree_qlevel', type=int, default=12,help='Octree量子化レベル')
    parser.add_argument('--octree_ctx_level', type=int, default=5,help='Octreeコンテキストの深さ')
    parser.add_argument('--octree_ctx_dim', type=int, default=8,help='Octreeコンテキスト特徴次元')
    parser.add_argument('--outlier_label_th_scale', default=4.0, type=float, help='外れ値判定のしきい値スケール（MAD基準）')
    parser.add_argument('--outlier_label_min_ratio', default=0.03, type=float, help='外れ値割合の最小値')
    parser.add_argument('--outlier_label_max_ratio', default=0.15, type=float, help='外れ値割合の最大値')
    # その他設定
    parser.add_argument('--encoder_bn', default=False, type=str2bool, help='EncoderでBatchNormを使うか')
    parser.add_argument('--k', default=16, type=int, help='近傍点数（kNN）')
    parser.add_argument('--global_mlp', default=True, type=str2bool, help='global MLPを使うか')
    parser.add_argument('--encoder_query_chunk', default=65536, type=int, help='Encoderのattention計算の分割サイズ（0で無効）')

    """Train"""
    parser.add_argument('--save_dir', default=f'trained_model', type=str, help='モデル保存ディレクトリ')
    parser.add_argument('--ckpt', default=f'pretrained/{pretrained_date}/{pretrained_time}_{method_loss}/{model_name}.pth', type=str, help='チェックポイントのパス')
    parser.add_argument('--out_path', default=f'pretrained/{file_day}/{file_time}', type=str, help='ログ・チェックポイント保存先')
    parser.add_argument('--optim', default='adam', type=str, help='最適化手法（adamまたはsgd）')
    parser.add_argument('--expansion', action='store_true', help='拡張データを使用するか')
    parser.add_argument('--gamma', default=0.5, type=float, help='学習率減衰の係数')
    parser.add_argument('--lr_decay_step', default=24, type=int, help='学習率を減衰させるステップ間隔')
    parser.add_argument('--max_files', default=1, type=int, help='読み込む最大ファイル数')
    parser.add_argument('--episodes', default=24, type=int, help='学習エピソード数')
    parser.add_argument('--lr', default=1e-3, type=float, help='学習率')
    parser.add_argument('--save_eval', default='loss', type=str, help='評価指標（lossまたはpsnr）')
    parser.add_argument('--deform', default=False, type=str2bool, help='変形モジュールをゆっくり学習するか')
    parser.add_argument('--loss_type', default='cd', type=str, help='幾何損失の種類')

    # 損失関数のパラメータ
    parser.add_argument('--com_bit',    default=10*100, type=float, help='圧縮損失のbit項の重み')
    parser.add_argument('--com_sin',    default=1, type=float, help='単一子ノード損失の重み')
    parser.add_argument('--com_node',   default=4, type=float, help='ノード数損失の重み')
    parser.add_argument('--com_ent',   default=2, type=float, help='エントロピー損失の重み')
    parser.add_argument('--prun_cnt',   default=5, type=float, help='Pruningの個数制御損失')
    parser.add_argument('--prun_out',   default=20*100, type=float, help='Pruningの外れ値損失')
    parser.add_argument('--add_cnt',    default=5, type=float, help='Addの個数制御損失')
    parser.add_argument('--add_fit',    default=4*100, type=float, help='Addのフィッティング損失')
    parser.add_argument('--add_rep',    default=1*100, type=float, help='Addの分散抑制損失')
    parser.add_argument('--disp_cnt',    default=5, type=float, help='Displacementの個数制御損失')
    parser.add_argument('--disp_fit',    default=4*100, type=float, help='Displacementのフィッティング損失')
    parser.add_argument('--w_geom',     default=1, type=float, help='幾何損失の重み')
    parser.add_argument('--w_com',      default=1*10**2, type=float, help='圧縮損失の重み')
    parser.add_argument('--w_prun',     default=3*10**2, type=float, help='Pruning損失の重み')
    parser.add_argument('--w_add',      default=3, type=float, help='Add損失の重み')
    parser.add_argument('--w_dis',      default=1*10^5, type=float, help='Displacement損失の重み')

    parser.add_argument('--lambda_p',   default=10**-5, type=float, help='soft圧縮損失の係数')
    parser.add_argument('--discrete_loss_mode', default='ste_hard', type=str, help='離散学習のモード')
    parser.add_argument('--discrete_surrogate_weight', default=1.0, type=float, help='STE時の代理勾配の重み')
    parser.add_argument('--discrete_policy_weight', default=0.01, type=float, help='ポリシー勾配の重み')
    parser.add_argument('--discrete_policy_reward_clip', default=100.0, type=float, help='報酬のクリップ値（0で無効）')
    parser.add_argument('--discrete_policy_baseline_momentum', default=0.95, type=float, help='ベースラインのEMA係数')

    """Compression"""
    parser.add_argument('--compress', default='OctAttention', type=str, help='使用する圧縮手法')
    parser.add_argument('--octree_voxel', type=float, default=1e-3, help='Octreeボクセルサイズ')
    parser.add_argument('--qs', type=int, default=2, help='量子化ステップサイズ')

    # OctAttention
    parser.add_argument('--max_gpu_mem_it', type=int, default=2**9, help='GPUメモリ制限に応じた反復回数')
    parser.add_argument('--oa_subprocess', default=False, type=str2bool, help='サブプロセスで圧縮を行うか')
    parser.add_argument('--compression_loss_backend', default='octattention_surrogate', type=str, help='圧縮損失の計算方法(proxy/octattention_actual/octattention_actual_ste/octattention_surrogate)')
    parser.add_argument('--compression_grad_probe', default=True, type=str2bool, help='圧縮損失から出力点群へ勾配が流れるか各stepで表示するか')
    parser.add_argument('--compression_grad_probe_every', default=1, type=int, help='圧縮損失の勾配診断を何回に1回表示するか')
    parser.add_argument('--octattention_actualcode', default=False, type=str2bool, help='OctAttention実圧縮で算術符号化後の実bitを使うか（学習中はFalse推奨）')
    parser.add_argument('--octattention_ckpt', default='compress/octree/OctAttention/modelsave/obj/encoder_epoch_00800093.pth', type=str, help='OctAttention encoder checkpoint')
    parser.add_argument('--octattention_tmp_dir', default='', type=str, help='OctAttention実圧縮用の一時ディレクトリ（空なら/dev/shm優先）')
    parser.add_argument('--compression_surrogate_levels', default='4,6,8', type=str, help='Soft octree surrogate特徴に使う階層')
    parser.add_argument('--compression_surrogate_hidden_dim', default=128, type=int, help='圧縮サロゲートMLPの隠れ次元')
    parser.add_argument('--compression_surrogate_lr', default=3e-3, type=float, help='圧縮サロゲートのオンライン学習率')
    parser.add_argument('--compression_surrogate_weight_decay', default=1e-5, type=float, help='圧縮サロゲートのweight decay')
    parser.add_argument('--compression_surrogate_train_steps', default=64, type=int, help='各stepでサロゲートを教師bitに合わせて更新する回数')
    parser.add_argument('--compression_surrogate_grad_clip', default=10.0, type=float, help='圧縮サロゲートの勾配クリップ')
    parser.add_argument('--compression_surrogate_target_scale', default=100.0, type=float, help='サロゲート内部で相対変化率を何倍して学習するか（100なら内部教師は％相当）')
    parser.add_argument('--compression_surrogate_pred_clip', default=100.0, type=float, help='サロゲートが予測するスケール後の相対変化量のtanhクリップ')
    parser.add_argument('--compression_surrogate_occ_gain', default=1.0, type=float, help='Soft occupancy変換のゲイン')
    parser.add_argument('--compression_surrogate_bit_weight', default=4.0, type=float, help='サロゲート教師損失のbit重み')
    parser.add_argument('--compression_surrogate_node_weight', default=1.0, type=float, help='サロゲート教師損失のnode重み')
    parser.add_argument('--compression_surrogate_single_weight', default=1.0, type=float, help='サロゲート教師損失のsingle-child重み')
    parser.add_argument('--compression_surrogate_bpn_weight', default=1.0, type=float, help='サロゲート教師損失のbpn(bits per node)重み')
    parser.add_argument('--compression_surrogate_entropy_weight', default=1.0, type=float, help='旧互換用: 現在はbpn重みとして扱う')
    parser.add_argument('--compression_surrogate_comp_bit_weight', default=1.0, type=float, help='提案手法に戻す圧縮損失のbit予測重み')
    parser.add_argument('--compression_surrogate_comp_node_weight', default=0.25, type=float, help='提案手法に戻す圧縮損失のnode予測重み')
    parser.add_argument('--compression_surrogate_comp_single_weight', default=0.25, type=float, help='提案手法に戻す圧縮損失のsingle-child予測重み')
    parser.add_argument('--compression_surrogate_comp_bpn_weight', default=0.25, type=float, help='提案手法に戻す圧縮損失のbpn(bits per node)予測重み')
    parser.add_argument('--compression_surrogate_comp_entropy_weight', default=0.25, type=float, help='旧互換用: 現在はbpn予測重みとして扱う')
    parser.add_argument('--compression_surrogate_loss_scale', default=100.0, type=float, help='サロゲート圧縮損失全体のスケール（100なら損失値は％相当）')

    # proxyOctreeCompression
    parser.add_argument('--proxy_max_depth',     default=12,    type=int,   help='Octreeの最大深さ')
    parser.add_argument('--proxy_lambda_entropy', default=1,    type=float,   help='エントロピー項の重み')
    parser.add_argument('--proxy_lambda_node_count',   default=1,  type=float,   help='ノード数項の重み')
    parser.add_argument('--proxy_lambda_single_child', default=1,     type=float,   help='単一子ノード項の重み')
    parser.add_argument('--proxy_round_tau', default=0.12, type=float, help='soft丸めの温度パラメータ')
    parser.add_argument('--proxy_mass_to_occ_gain', default=1.0, type=float, help='質量→占有変換のスケール')
    parser.add_argument('--octattention_teacher_device', default='auto', type=str, help='OctAttention teacherの実行先(auto/cuda/cpu/balanced)')
    parser.add_argument('--compression_rate_metric', default='total_bits', type=str, help='圧縮率損失の基準(total_bits/bits_per_point/bits_per_input_point)')

    """Test"""
    parser.add_argument('--input_dir_test', default=f'../data/train/{dir_input}/UVG/CasualSquat', type=str, help='テスト用入力点群のパス')
    parser.add_argument('--max_files_test', default=5, type=int, help='テスト時に読み込む最大ファイル数')
    parser.add_argument('--save_ply_dir', default=f'../data/test', type=str, help='出力点群の保存先')

    """設定"""
    parser.add_argument('--seed', default=21, type=float, help='乱数シード')
    parser.add_argument('--deterministic', default=False, type=str2bool, help='再現性のためCUDAを固定するか')
    parser.add_argument('--num_points', default=12288, type=int, help='1パッチあたりの点数')
    parser.add_argument('--max_input_points', default=0, type=int, help='入力点数の上限（0で無効）')
    parser.add_argument('--input_sampling', default='random', type=str, help='サンプリング方法')
    parser.add_argument('--split2patch', default=False, type=str2bool, help='点群をパッチ分割するか')
    parser.add_argument('--patch_rate', default=1.0, type=float, help='パッチ重なり率')
    parser.add_argument('--batch_size', default=1, type=int, help='バッチサイズ')
    parser.add_argument('--patch_batch_size', default=4, type=int, help='パッチ単位のバッチサイズ')
    parser.add_argument('--patch_cover_retry', default=4, type=int, help='カバーできない点を再試行する回数')
    parser.add_argument('--num_workers', default=4, type=int, help='データローダのワーカー数')
    parser.add_argument('--pin_memory', default=True, type=str2bool, help='CPU→GPU転送高速化のためメモリ固定するか')
    parser.add_argument('--persistent_workers', default=True, type=str2bool, help='ワーカーを維持するか')
    parser.add_argument('--dataset_cache', default=True, type=str2bool, help='データセットをメモリにキャッシュするか')
    parser.add_argument('--mp_start_method', default='auto', type=str, help='マルチプロセス起動方法')
    parser.add_argument('--weight_decay', default=0, type=float, help='重み減衰')
    parser.add_argument('--bptt', default=1024, type=float, help='（内部用パラメータ）')
    parser.add_argument('--parallel', default=False, type=str2bool, help='並列モデルを使うか')
    parser.add_argument('--module_bn_use_running_stats', default=False, type=str2bool, help='Encoder以外のBatchNormでrunning statsを使うか')
    parser.add_argument('--use_tf32', default=True, type=str2bool, help='TF32を使用するか')
    parser.add_argument('--use_amp', default=True, type=str2bool, help='混合精度学習を使うか')
    parser.add_argument('--amp_dtype', default='auto', type=str, help='AMPのデータ型')
    parser.add_argument('--amp_init_scale', default=1.0, type=float, help='GradScaler初期値')
    parser.add_argument('--amp_overflow_patience', default=2, type=int, help='オーバーフロー許容回数')
    parser.add_argument('--cache_frozen_inputs', default=True, type=str2bool, help='Encoder出力をキャッシュするか')
    parser.add_argument('--cache_gt_loss', default=True, type=str2bool, help='GT側損失をキャッシュするか')
    parser.add_argument('--cache_max_entries', default=192, type=int, help='キャッシュ最大数')
    parser.add_argument('--cache_max_memory_mb', default=8192, type=int, help='キャッシュ最大メモリ（MB）')
    parser.add_argument('--auto_disable_partial_frozen_cache', default=True, type=str2bool, help='キャッシュ不足時に自動無効化するか')
    parser.add_argument('--clear_main_ply_cache_for_workers', default=True, type=str2bool, help='メモリ重複を防ぐためキャッシュ削除')
    parser.add_argument('--warmup_frozen_cache', default=False, type=str2bool, help='事前にキャッシュを作るか')
    parser.add_argument('--warmup_gt_cache', default=False, type=str2bool, help='GTキャッシュを事前生成するか')
    parser.add_argument('--warmup_max_files', default=0, type=int, help='ウォームアップ対象ファイル数')
    parser.add_argument('--warmup_max_seconds', default=0, type=float, help='ウォームアップ最大時間')
    parser.add_argument('--warmup_log_rate', default=8, type=int, help='ウォームアップログ間隔')
    parser.add_argument('--log_flush_every', default=32, type=int, help='ログ書き込みフラッシュ間隔')
    parser.add_argument('--log_sync_every', default=0, type=int, help='ログ同期間隔')
    parser.add_argument('--verbose_step_logs', default=True, type=str2bool, help='詳細ログを出すか')
    parser.add_argument('--epoch_plot_rate', default=4, type=int, help='エポックごとのプロット保存間隔')
    parser.add_argument('--episode_plot_rate', default=1, type=int, help='エピソードごとのプロット保存間隔')
    parser.add_argument('--retain_debug_tensors', default=False, type=str2bool, help='中間勾配を保持するか')
    parser.add_argument('--debug_grad_flow', default=False, type=str2bool, help='勾配ノルムをログ出力するか')
    parser.add_argument('--debug_grad_flow_rate', default=1, type=int, help='勾配ログの出力間隔')
    parser.add_argument('--debug_timing', default=False, type=str2bool, help='ステップ内の時間内訳をログ出力するか')
    parser.add_argument('--mail_notify', default=False, type=str2bool, help='学習イベントをメール通知するか')
    parser.add_argument('--mail_to', default='maeshu0619@gmail.com', type=str, help='通知メールの宛先')
    parser.add_argument('--mail_from', default='maeshu0619@gmail.com', type=str, help='通知メールの送信元')
    parser.add_argument('--mail_smtp_host', default='smtp.gmail.com', type=str, help='SMTPホスト。空ならsendmailを試す')
    parser.add_argument('--mail_smtp_port', default=587, type=int, help='SMTPポート')
    parser.add_argument('--mail_smtp_user', default='', type=str, help='SMTPユーザ')
    parser.add_argument('--mail_smtp_password_env', default='MYNET_MAIL_PASSWORD', type=str, help='SMTPパスワードを読む環境変数名')
    parser.add_argument('--mail_use_tls', default=True, type=str2bool, help='SMTP STARTTLSを使うか')
    parser.add_argument('--mail_timeout', default=10.0, type=float, help='メール送信タイムアウト秒')
    parser.add_argument('--mail_sendmail_path', default='/usr/sbin/sendmail', type=str, help='sendmailコマンドのパス')

    args = parser.parse_args()
    mode_alias = {
        "hard_ste": "ste_hard",
        "soft": "weighted_soft",
        "legacy": "weighted_soft",
    }
    discrete_loss_mode = str(args.discrete_loss_mode).strip().lower()
    args.discrete_loss_mode = mode_alias.get(discrete_loss_mode, discrete_loss_mode)
    if args.discrete_loss_mode not in {"ste_hard", "hard", "weighted_soft"}:
        raise ValueError(
            "--discrete_loss_mode must be one of: ste_hard, hard, weighted_soft "
            f"(got {args.discrete_loss_mode})"
        )
    args.compression_rate_metric = str(args.compression_rate_metric).strip().lower()
    if args.compression_rate_metric not in {"total_bits", "bits_per_point", "bits_per_input_point"}:
        raise ValueError(
            "--compression_rate_metric must be one of: total_bits, bits_per_point, "
            f"bits_per_input_point (got {args.compression_rate_metric})"
        )
    args.compression_loss_backend = str(args.compression_loss_backend).strip().lower()
    if args.compression_loss_backend not in {"proxy", "octattention_actual", "octattention_actual_ste", "octattention_surrogate"}:
        raise ValueError(
            "--compression_loss_backend must be one of: proxy, octattention_actual, "
            f"octattention_actual_ste, octattention_surrogate (got {args.compression_loss_backend})"
        )
    
    return args
