#!/usr/bin/env python3
"""nsys2perfetto.py —— 把 nsys 的 SQLite 导出转成 Chrome Trace Event Format。

为什么要转：`nsys export --type=json` 产出的是 nsys 自己的 schema（顶层是 {"data":[...]}），
Perfetto / ui.perfetto.dev 读不了。Perfetto 认的是 Chrome Trace Event Format：

    [ {"name":.., "cat":.., "ph":"X", "ts":<微秒>, "dur":<微秒>, "pid":.., "tid":.., "args":{..}}, ... ]

注意 CUPTI 的时间戳是【纳秒】，要除以 1000。

用法：
    nsys profile -o t ./app
    nsys export --type=sqlite -o t.sqlite t.nsys-rep
    python3 nsys2perfetto.py t.sqlite t.json
然后在 https://ui.perfetto.dev 上把 t.json 拖进去。
"""
import json
import sqlite3
import sys


def load_strings(db):
    return {i: v for i, v in db.execute("select id, value from StringIds")}


def main(sqlite_path, out_path):
    db = sqlite3.connect(sqlite_path)
    S = load_strings(db)

    events = []
    t_min = None

    def note_time(starts):
        nonlocal t_min
        for s in starts:
            if s is not None and (t_min is None or s < t_min):
                t_min = s

    # ---------- GPU kernel ----------
    rows = list(
        db.execute(
            """
            select start, end, streamId, demangledName, shortName,
                   gridX, gridY, gridZ, blockX, blockY, blockZ,
                   registersPerThread, staticSharedMemory, dynamicSharedMemory
            from CUPTI_ACTIVITY_KIND_KERNEL
            """
        )
    )
    note_time(r[0] for r in rows)
    kernels = []
    for r in rows:
        (start, end, stream, demangled, short, gx, gy, gz, bx, by, bz,
         regs, ssmem, dsmem) = r
        name = S.get(demangled) or S.get(short) or "<unknown kernel>"
        # 把模板参数剥掉，时间线上短一点
        short_name = name.split("(")[0]
        kernels.append(
            {
                "name": short_name,
                "cat": "kernel",
                "ph": "X",
                "ts": start / 1000.0,          # ns -> us
                "dur": (end - start) / 1000.0,
                "pid": 0,
                "tid": stream if stream is not None else 0,
                "args": {
                    "full_name": name,
                    "grid": [gx, gy, gz],
                    "block": [bx, by, bz],
                    "registers/thread": regs,
                    "static smem (B)": ssmem,
                    "dynamic smem (B)": dsmem,
                },
            }
        )

    # ---------- GPU memcpy ----------
    memcpy = []
    for start, end, stream, bytes_, copyKind in db.execute(
        "select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        note_time([start])
        memcpy.append(
            {
                "name": "memcpy",
                "cat": "memcpy",
                "ph": "X",
                "ts": start / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": 0,
                "tid": stream if stream is not None else 0,
                "args": {"bytes": bytes_, "copyKind": copyKind},
            }
        )

    # ---------- NVTX 区间（程序里打的版本标记）----------
    nvtx = []
    for start, end, text, text_id, global_tid in db.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"
    ):
        label = text or S.get(text_id) if text_id is not None else text
        if not label:
            continue
        note_time([start])
        nvtx.append(
            {
                "name": label,
                "cat": "nvtx",
                "ph": "X",
                "ts": start / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": 1,
                "tid": 0,
                "args": {"globalTid": global_tid},
            }
        )

    # ---------- 元数据：给轨道起名字 ----------
    meta = [
        {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "GPU (CUDA)"}},
        {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "NVTX 区间"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 0,
         "args": {"name": "各版本"}},
    ]
    streams = sorted({e["tid"] for e in kernels + memcpy})
    for s in streams:
        meta.append(
            {"name": "thread_name", "ph": "M", "pid": 0, "tid": s,
             "args": {"name": f"Stream {s}"}}
        )

    events = meta + nvtx + kernels + memcpy

    # 时间戳整体平移，让第一条事件从 0 开始（Perfetto 也会自动处理，这里更干净）
    if t_min:
        for e in events:
            if "ts" in e:
                e["ts"] = round(e["ts"] - t_min / 1000.0, 3)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)

    total_kernels = len(kernels)
    print(f"写出 {out_path}")
    print(f"  kernel 事件 : {total_kernels}")
    print(f"  memcpy 事件 : {len(memcpy)}")
    print(f"  NVTX 区间   : {len(nvtx)}")
    print(f"  流(stream)数: {len(streams)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
