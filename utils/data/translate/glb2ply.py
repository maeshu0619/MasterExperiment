import trimesh

# GLBファイルを読み込む
mesh = trimesh.load("../Dataset/ori.glb")

# 複数メッシュを含む場合は結合
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh.dump())

# PLY（メッシュ）として保存
mesh.export("ori_mesh.ply")

print("ori_mesh.ply を出力しました")
