import os
import datetime

for fname in sorted(os.listdir("./wav")):
    if fname.endswith(".wav"):
        path = os.path.join("./wav", fname)
        stat = os.stat(path)
        # macOS 13 이상에서만 st_birthtime_ns 지원, 그 외는 st_birthtime * 1e9
        ns = getattr(stat, "st_birthtime_ns", int(stat.st_birthtime * 1e9))
        dt = datetime.datetime.fromtimestamp(ns / 1e9)
        print(f"{fname}: {dt.strftime('%Y-%m-%d %H:%M:%S')}.{ns % 1_000_000_000:09d}")