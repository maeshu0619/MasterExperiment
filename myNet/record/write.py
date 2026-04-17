import datetime
import os

class Writing:
    # 
    def __init__(self, file_day, file_time, filename, dataname, flush_every=32, sync_every=0):
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
        # 行単位で即時反映されるように line buffering を使う。
        # これで学習中にログファイルを開いても内容を確認しやすい。
        self.file = open(self.file_path, "a", encoding="utf-8", buffering=1)
        self.flush_every = max(int(flush_every), 1)
        self.sync_every = max(int(sync_every), 0)
        self._pending_writes = 0
        self._pending_sync_writes = 0
        print(f"[INFO] ログファイルを作成しました: {self.file_path}")

    def write(self, data):
        self.file.write(data + "\n")
        self._pending_writes += 1
        self._pending_sync_writes += 1
        if self._pending_writes >= self.flush_every:
            self.flush()

    def flush(self, force_sync=False):
        self.file.flush()
        should_sync = force_sync or (self.sync_every > 0 and self._pending_sync_writes >= self.sync_every)
        if should_sync:
            try:
                os.fsync(self.file.fileno())
            except OSError:
                pass
            self._pending_sync_writes = 0
        self._pending_writes = 0

    def close(self):
        self.flush(force_sync=True)
        self.file.close()
