import os

def get_bitstream_size(file_path: str):
    """
    圧縮後のbinファイルのビットストリームサイズを計測する関数

    引数:
        file_path: 計測したいファイルのパス

    戻り値:
        (byte数, bit数) のタプル
    """
    # ファイルサイズ（バイト）を取得
    byte_size = os.path.getsize(file_path)
    # ビット数に変換（1 byte = 8 bit）
    bit_size = byte_size * 8
    return byte_size, bit_size


if __name__ == "__main__":
    # 例: RENOやDracoの出力binファイルパスを指定
    bin_path = "Dataset/encoded/KITTI/Draco/gt.bin"  # 実際のパスに書き換える

    bytes_, bits_ = get_bitstream_size(bin_path)

    print(f"ファイル: {bin_path}")
    print(f"ビットストリームサイズ: {bytes_} bytes")
    print(f"ビットストリームサイズ: {bits_} bits")
