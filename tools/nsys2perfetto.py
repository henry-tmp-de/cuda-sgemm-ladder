#!/usr/bin/env python3
"""nsys2perfetto.py —— 把 nsys 的 SQLite 导出转成 Chrome Trace Event Format。

为什么需要转：`nsys export --type=json` 产出的是 nsys 自己的 schema（顶层 {"data":[...]}），
Perfetto / ui.perfetto.dev 读不了。Perfetto 认的是 Chrome Trace Event Format：

    [ {"name":.., "cat":.., "ph":"X", "ts":<微秒>, "dur":<微秒>, "pid":.., "tid":.., "args":{..}}, ... ]

两个容易踩的点：
  * CUPTI 的时间戳单位是【纳秒】，Chrome Trace 要【微秒】—— 必须 /1000
  * 只导 CUPTI_ACTIVITY_KIND_KERNEL 会漏掉**整条 CPU 轨道**
    （CPU 侧的 CUDA API 调用在 CUPTI_ACTIVITY_KIND_RUNTIME 里）

本脚本导出的轨道：
    pid 0  GPU            —— kernel / memset / memcpy / 同步点，tid = CUDA stream
    pid 1  NVTX 区间      —— 程序里 nvtxRangePush 打的版本标记
    pid 2  CPU (CUDA API) —— 每个 cudaXxx 调用的耗时，tid = 主机线程

用法：
    nsys profile -o t ./app
    nsys export --type=sqlite -o t.sqlite t.nsys-rep
    python3 nsys2perfetto.py t.sqlite t.json
然后在 https://ui.perfetto.dev 上把 t.json 拖进去。
"""
import json
import sqlite3
import sys

# CUDA 用 0xFFFFFFFF 表示"没有绑定到具体流"（设备级同步）
SYNC_TID = 0xFFFFFFFF


def table_exists(db, name):
    return (
        db.execute(
            "select count(*) from sqlite_master where type='table' and name=?", (name,)
        ).fetchone()[0]
        > 0
    )


def main(sqlite_path, out_path):
    db = sqlite3.connect(sqlite_path)
    S = {i: v for i, v in db.execute("select id, value from StringIds")}

    events = []
    t_min = [None]

    def note(starts):
        for s in starts:
            if s is not None and (t_min[0] is None or s < t_min[0]):
                t_min[0] = s

    def ev(name, cat, start, dur_ns, pid, tid, args=None):
        note([start])
        e = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": start / 1000.0,
            "dur": dur_ns / 1000.0,
            "pid": pid,
            "tid": tid,
        }
        if args:
            e["args"] = args
        events.append(e)

    counts = {}

    # ================= pid 0：GPU =================
    # ---- kernel ----
    rows = list(
        db.execute(
            """select start, end, streamId, demangledName, shortName,
                      gridX, gridY, gridZ, blockX, blockY, blockZ,
                      registersPerThread, staticSharedMemory, dynamicSharedMemory
               from CUPTI_ACTIVITY_KIND_KERNEL"""
        )
    )
    counts["kernel"] = len(rows)
    for r in rows:
        (start, end, stream, demangled, short, gx, gy, gz, bx, by, bz,
         regs, ssmem, dsmem) = r
        full = S.get(demangled) or S.get(short) or "<unknown kernel>"
        ev(full.split("(")[0], "kernel", start, end - start, 0,
           stream if stream is not None else 0,
           {"full_name": full, "grid": [gx, gy, gz], "block": [bx, by, bz],
            "registers/thread": regs, "static smem (B)": ssmem,
            "dynamic smem (B)": dsmem})

    # ---- memset（原来漏了，cudaMemset 就落在这里）----
    if table_exists(db, "CUPTI_ACTIVITY_KIND_MEMSET"):
        rows = list(db.execute(
            "select start, end, streamId, bytes, value from CUPTI_ACTIVITY_KIND_MEMSET"))
        counts["memset"] = len(rows)
        for start, end, stream, nbytes, val in rows:
            ev("memset", "memset", start, end - start, 0,
               stream if stream is not None else 0,
               {"bytes": nbytes, "value": val})

    # ---- memcpy ----
    rows = list(db.execute(
        "select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
    counts["memcpy"] = len(rows)
    for start, end, stream, nbytes, kind in rows:
        ev("memcpy", "memcpy", start, end - start, 0,
           stream if stream is not None else 0,
           {"bytes": nbytes, "copyKind": kind})

    # ---- 同步点（原来漏了）----
    if table_exists(db, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        rows = list(db.execute(
            "select start, end, streamId, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"))
        counts["sync"] = len(rows)
        for start, end, stream, stype in rows:
            # 同步点的 streamId 是 0xFFFFFFFF（CUDA 的"无流"哨兵），
            # 不是真的流号；单独给它一条泳道，别混进 Stream 泳道里。
            ev("sync", "sync", start, end - start, 0, SYNC_TID,
               {"syncType": stype})

    # ================= pid 1：NVTX =================
    rows = list(db.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"))
    counts["nvtx"] = len(rows)
    for start, end, text, text_id, gtid in rows:
        label = text if text else (S.get(text_id) if text_id is not None else None)
        if not label:
            continue
        ev(label, "nvtx", start, end - start, 1, 0, {"globalTid": gtid})

    # ================= pid 2：CPU 侧的 CUDA API 调用 =================
    # 这是原来整条漏掉的轨道。名字来自 StringIds（cudaLaunchKernel / cudaMemset /
    # cudaEventRecord / cudaDeviceSynchronize ...）。
    g2t = {}
    if table_exists(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        rows = list(db.execute(
            "select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
        counts["runtime(CPU API)"] = len(rows)
        for start, end, gtid, name_id in rows:
            if gtid not in g2t:
                g2t[gtid] = len(g2t)
            ev(S.get(name_id, "<cuda api>"), "cuda_api", start, end - start,
               2, g2t[gtid], {"globalTid": gtid})

    # ================= 元数据：轨道命名 =================
    meta = [
        {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "GPU (CUDA)"}},
        {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "NVTX 区间"}},
        {"name": "process_name", "ph": "M", "pid": 2, "args": {"name": "CPU (CUDA API)"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "各版本"}},
    ]
    # GPU 各 stream 泳道
    streams = sorted({e["tid"] for e in events if e["pid"] == 0})
    for s in streams:
        nm = "设备级同步" if s == SYNC_TID else f"Stream {s}"
        meta.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": s,
                     "args": {"name": nm}})
    # CPU 各线程泳道：尽量用 ThreadNames 里的真实线程名
    tid_names = {}
    if table_exists(db, "ThreadNames"):
        for name_id, _prio, gtid in db.execute("select nameId, priority, globalTid from ThreadNames"):
            if gtid in g2t:
                tid_names.setdefault(g2t[gtid], S.get(name_id, ""))
    for gtid, tid in g2t.items():
        nm = tid_names.get(tid) or f"thread {gtid % 100000}"
        meta.append({"name": "thread_name", "ph": "M", "pid": 2, "tid": tid,
                     "args": {"name": nm}})

    events = meta + events

    # 时间戳整体平移到 0 起点
    if t_min[0]:
        off = t_min[0] / 1000.0
        for e in events:
            if "ts" in e:
                e["ts"] = round(e["ts"] - off, 3)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)

    print(f"写出 {out_path}")
    for k, v in counts.items():
        print(f"  {k:<18}: {v}")
    print(f"  GPU stream 数     : {len(streams)}")
    print(f"  CPU 线程数        : {len(g2t)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
