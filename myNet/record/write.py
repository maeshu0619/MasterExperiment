import datetime
import os

class Writing:
    # 
    def __init__(self, file_day, file_time, filename, dataname):
        # 現在のファイル（write.py）基準でルートを決定
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )

        # ディレクトリパス
        record_dir = os.path.join(
            base_dir, "log", file_day, filename
        )

        # ディレクトリ作成
        os.makedirs(record_dir, exist_ok=True)

        # ファイルパス
        self.file_path = os.path.join(
            record_dir, f"{file_time}.txt"
        )

        # ファイルを開く（なければ作成）
        self.file = open(self.file_path, "a", encoding="utf-8")
        print(f"[INFO] ログファイルを作成しました: {self.file_path}")

    def write(self, data):
        self.file.write(data + "\n")
        self.file.flush()  # 即時書き込み（実験ログ向け）

    def close(self):
        self.file.close()
