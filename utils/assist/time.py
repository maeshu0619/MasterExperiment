import time

class TimeTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """計測開始"""
        self.start_time = time.time()
        print("計測開始")

    def stop(self):
        """計測終了"""
        self.end_time = time.time()
        print("計測終了")

    def elapsed(self):
        if self.start_time is None:
            return None
        if self.end_time is None:
            return time.time() - self.start_time  # 計測中の場合
        return self.end_time - self.start_time

    def print_elapsed(self, str):
        if self.start_time is None:
            print("計測が開始されていません。")
            return
        elapsed_time = self.elapsed()
        print(f"{str}: {elapsed_time:.2f} 秒")