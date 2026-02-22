# compress/OctAttention/oa_worker.py
import os
import sys
import pickle
import torch
# import sys
# sys.stdout = sys.stderr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from compress.OctAttention.encoderTool import oa_main
from compress.OctAttention.octAttention import build_model  # * をやめる

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_encoder_model():
    m = build_model(device)
    ckpt_path = "../compress/octree/OctAttention/modelsave/obj/encoder_epoch_00800093.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt["encoder"])
    m.eval()
    return m

def main():
    model = load_encoder_model()  # 起動時に1回だけGPUへ

    # 以後、stdinから何度でも (pts, args) を受け取って返す
    while True:
        try:
            pts = pickle.load(sys.stdin.buffer)
            args = pickle.load(sys.stdin.buffer)
        except EOFError:
            break  # 親が閉じた

        with torch.no_grad():
            com = oa_main(
                args=args,
                pts=pts.to(device),
                model=model,
                qs=2,
                writer=None,
                file_date="tmp"
            )

        pickle.dump(com, sys.stdout.buffer)
        sys.stdout.buffer.flush()

        # 念のため（通常不要だが、断片化対策として入れてよい）
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()