import argparse
from cfgs.utils import str2bool

pretrained_date = "20260405"
pretrained_time = "150948"
method_loss = "cd"
model_name = "best"

def parse_pugan_args(parser, file_day, file_time, dir_input):
    """基本情報"""
    parser.add_argument('--date', default=f'{file_day}', type=str, help='date')
    parser.add_argument('--time', default=f'{file_time}', type=str, help='time')
    parser.add_argument('--input_dir', default=f'../data/train/{dir_input}', type=str, help='path to folder of input point clouds')
    parser.add_argument('--cpu', action='store_true', help='whether not using GPU for training')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')

    """ネットワーク条件"""
    # Network
    parser.add_argument('--encoder_0grad', default=True, type=str2bool, help='whether optimizing encoder')
    parser.add_argument('--prune', default=True, type=str2bool, help='whether use Pruning Module')
    parser.add_argument('--add', default=True, type=str2bool, help='whether use Adding Module')
    parser.add_argument('--disp', default=True, type=str2bool, help='whether use Displacement Module')
    # Encoder, FP Module
    parser.add_argument('--encoder_dim', default=64, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--out_dim', default=64, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--local_feat_dim', default=192, type=int, help='local geometric feature dim (Analyzer用)')
    parser.add_argument('--fused_feat_dim', default=64, type=int, help='encoder fused feature dim')
    parser.add_argument('--fp_mlp_channels', nargs='+', type=int, default=[128, 64], help='FP MLP hidden channels')
    # Pruning Module
    parser.add_argument('--prune_hidden_dim', default=64, type=int, help='hidden dim of pruning MLP')
    parser.add_argument('--prune_d_high_is_inlier', default=True, type=str2bool, help='whether high d score means inlier in pruning module')
    parser.add_argument('--prune_robust_c', default=2.0, type=float, help='robust weight parameter in pruning module')
    # Adding Module
    parser.add_argument('--add_hidden_dim', default=64, type=int, help='hidden feature dim of AddModule MLP')
    parser.add_argument('--add_fit_ref_max', default=4096, type=int, help='Calculate fitting loss with this number of points sampled from reference')
    parser.add_argument('--add_attr', default=True, type=str2bool, help='whether add color attribute to added points')
    parser.add_argument('--add_color', default='Red', type=str, help='color name of added points')
    parser.add_argument('--add_th', default=0.5, type=float, help='Add Module importance threshold')
    parser.add_argument('--target_add_ratio', default=0.05, type=float, help='target addition ratio for Add Module')
    parser.add_argument('--max_add_ratio', default=0.2, type=float, help='Add Module maximum addition ratio')
    parser.add_argument('--lambda_sparse', default=1e-3, type=float, help='Add Module sparse regularization weight')
    # Displacement Module
    parser.add_argument('--disp_hidden_dim', default=64, type=int, help='hidden dim of displacement module MLP')
    parser.add_argument('--max_disp_offset', default=0.002, type=float, help='maximum displacement offset in meters')
    parser.add_argument('--disp_num_blocks',default=4,type=int,help='number of residual blocks in displacement module')
    parser.add_argument('--disp_num_steps',default=1,type=int,help='number of iterative refinement steps')
    parser.add_argument('--disp_step_size',default=1.0,type=float,help='step size for displacement update')
    parser.add_argument('--disp_step_decay',default=0.95,type=float,help='decay rate of step size per iteration')
    parser.add_argument('--disp_grad_clip',default=10.0,type=float,help='gradient clipping value for displacement update')
    parser.add_argument('--disp_use_gate',default=True,type=lambda x:x.lower()=='true',help='whether to use gating mechanism (True/False)')
    # Analyzer
    parser.add_argument('--octree_qlevel', type=int, default=12,help='')
    parser.add_argument('--octree_ctx_level', type=int, default=5,help='')
    parser.add_argument('--octree_ctx_dim', type=int, default=5,help='')
    # その他設定
    parser.add_argument('--encoder_bn', default=False, type=str2bool, help='whether use batch normalization in encoder')
    parser.add_argument('--k', default=16, type=int, help='neighbor number in encoder')
    parser.add_argument('--global_mlp', default=True, type=str2bool, help='whether use global_mlp in encoder')

    """Train"""
    parser.add_argument('--save_dir', default=f'trained_model', type=str, help='save trained model')
    parser.add_argument('--ckpt', default=f'pretrained/{pretrained_date}/{pretrained_time}_{method_loss}/{model_name}.pth', type=str, help='path to save output point clouds')
    parser.add_argument('--out_path', default=f'pretrained/{file_day}/{file_time}', type=str, help='the checkpoint and log save path')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--expansion', action='store_true', help='whether using expanded data for training')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    parser.add_argument('--lr_decay_step', default=24, type=int, help='learning rate decay step size')
    parser.add_argument('--max_files', default=20, type=int, help='maximum number of files to load')
    parser.add_argument('--episodes', default=24, type=int, help='training episodes')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--save_eval', default='loss', type=str, help='loss or psnr')
    parser.add_argument('--deform', default=False, type=str2bool, help='whether otimizing deform module slowly')
    parser.add_argument('--loss_type', default='cd', type=str, help='what kind of loss function for geometry')

    # 損失関数のパラメータ
    parser.add_argument('--com_bit',    default=10*100, type=float, help='one of Loss function parameter')
    parser.add_argument('--com_sin',    default=1, type=float, help='one of Loss function parameter')
    parser.add_argument('--com_node',   default=4, type=float, help='one of Loss function parameter')
    parser.add_argument('--com_ent',   default=2, type=float, help='one of Loss function parameter')
    parser.add_argument('--prun_cnt',   default=5, type=float, help='one of Loss function parameter')
    parser.add_argument('--prun_out',   default=20*100, type=float, help='one of Loss function parameter')
    parser.add_argument('--add_cnt',    default=5, type=float, help='one of Loss function parameter')
    parser.add_argument('--add_fit',    default=4*100, type=float, help='one of Loss function parameter')
    parser.add_argument('--add_rep',    default=1*100, type=float, help='one of Loss function parameter')
    parser.add_argument('--disp_cnt',    default=5, type=float, help='one of Loss function parameter')
    parser.add_argument('--disp_fit',    default=4*100, type=float, help='one of Loss function parameter')
    parser.add_argument('--w_geom',     default=2, type=float, help='one of Loss function parameter')
    parser.add_argument('--w_com',      default=1.5*10, type=float, help='one of Loss function parameter')
    parser.add_argument('--w_prun',     default=2*10**2, type=float, help='one of Loss function parameter')
    parser.add_argument('--w_add',      default=1, type=float, help='one of Loss function parameter')
    parser.add_argument('--w_dis',      default=3, type=float, help='one of Loss function parameter')

    parser.add_argument('--lambda_p',   default=10**-5, type=float, help='lambda parameter for soft compression loss')

    """Compression"""
    parser.add_argument('--compress', default='OctAttention', type=str, help='what kind of compression')
    parser.add_argument('--octree_voxel', type=float, default=1e-3, help='voxel size for octree / voxel grid filter')
    parser.add_argument('--qs', type=int, default=2, help='quantization step sizes for compression')

    # OctAttention
    parser.add_argument('--max_gpu_mem_it', type=int, default=2**9, help='maximum GPU memory iteration for OctAttention compression')
    parser.add_argument('--oa_subprocess', default=False, type=str2bool, help='whether use subprocess for OctAttention compression')

    # proxyOctreeCompression
    parser.add_argument('--proxy_max_depth',     default=8,    type=int,   help='')
    parser.add_argument('--proxy_lambda_entropy', default=1,    type=float,   help='')
    parser.add_argument('--proxy_lambda_node_count',   default=1,  type=float,   help='')
    parser.add_argument('--proxy_lambda_single_child', default=1,     type=float,   help='')

    """Test"""
    # parser.add_argument('--input_dir_test', default=f'../data/train/{dir_input}/MVUB/andrew9/frame0000.ply', type=str, help='path to folder of input point clouds')
    # parser.add_argument('--input_dir_test', default=f'../data/train/{dir_input}/8iVSLF/LongDress/longdress_vox10_1052.ply', type=str, help='path to folder of input point clouds')
    # parser.add_argument('--input_dir_test', default=f'../data/train/{dir_input}/UVG/CasualSquat/CasualSquat_UVG_raw_25_0_250_0000.ply', type=str, help='path to folder of input point clouds')
    parser.add_argument('--input_dir_test', default=f'../data/train/{dir_input}/UVG/CasualSquat', type=str, help='path to folder of input point clouds')
    parser.add_argument('--max_files_test', default=5, type=int, help='maximum number of files to load for testing')
    parser.add_argument('--save_ply_dir', default=f'../data/test', type=str, help='path to save output point clouds')
    # parser.add_argument('--save_ply_dir', default=f'../data/test/{file_day}-{file_time}', type=str, help='path to save output point clouds')
    
    """設定"""
    parser.add_argument('--seed', default=21, type=float, help='seed')
    parser.add_argument('--num_points', default=4096, type=int, help='number of points per patch')
    parser.add_argument('--split2patch', default=False, type=str2bool, help='whether split point cloud to patches')
    parser.add_argument('--patch_rate', default=1, type=int, help='patch sampling rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--patch_batch_size', default=32, type=int, help='patch batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers number')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--bptt', default=1024, type=float, help='gamma for scheduler_steplr')
    parser.add_argument('--parallel', default=False, type=str2bool, help='whether use model with parallel')

    args = parser.parse_args()
    
    return args
