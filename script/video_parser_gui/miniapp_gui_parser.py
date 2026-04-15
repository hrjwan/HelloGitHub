import argparse
import inspect
import json
import os
import queue
import re
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from rapidfuzz import fuzz


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def frame_diff_score(img1, img2) -> float:
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
    h1 = cv2.normalize(h1, h1).flatten()
    h2 = cv2.normalize(h2, h2).flatten()
    return float(np.linalg.norm(h1 - h2) * 100)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def normalize_brand(text: str) -> str:
    text = clean_text(text)
    fixes = {
        "瑞幸咔啡": "瑞幸咖啡",
        "库迪咔啡": "库迪咖啡",
        "美宣佳超市": "美宜佳超市",
    }
    for src, dst in fixes.items():
        text = text.replace(src, dst)
    return text


def parse_blocks_from_text(text: str):
    pattern = re.compile(
        r"(?P<brand>[^\n]{2,20})\n?.{0,30}?在营门店数[:：]?\s*(?P<stores>\d{3,7})"
        r".{0,40}?口碑值[:：]?\s*(?P<score>\d+(?:\.\d+)?)"
        r".{0,40}?人均[:：]?\s*[¥￥]?\s*(?P<price>\d+(?:\.\d+)?)",
        re.S,
    )

    rows = []
    for item in pattern.finditer(text):
        rows.append(
            {
                "品牌": normalize_brand(item.group("brand")),
                "在营门店数": int(item.group("stores")),
                "口碑值": float(item.group("score")),
                "人均": float(item.group("price")),
            }
        )
    return rows


def dedup_rows(rows):
    merged = {}
    for row in rows:
        brand = row["品牌"]
        target = None
        for key in merged:
            if fuzz.ratio(brand, key) >= 88:
                target = key
                break
        if target is None:
            merged[brand] = row
        elif row["在营门店数"] > merged[target]["在营门店数"]:
            merged[target] = row
    return list(merged.values())


def _iter_ocr_texts(result):
    """
    兼容 PaddleOCR 不同版本返回结构。
    """
    if not result:
        return

    # 常见旧结构: list[list[[box, [text, score]], ...]]
    if isinstance(result, list):
        for block in result:
            if isinstance(block, list):
                for item in block:
                    if (
                        isinstance(item, list)
                        and len(item) >= 2
                        and isinstance(item[1], (list, tuple))
                        and len(item[1]) >= 2
                    ):
                        text = str(item[1][0]).strip()
                        try:
                            confidence = float(item[1][1])
                        except (TypeError, ValueError):
                            confidence = 0.0
                        yield text, confidence
            elif isinstance(block, dict):
                # 可能的新结构: [{"rec_texts": [...], "rec_scores": [...]}]
                rec_texts = block.get("rec_texts", [])
                rec_scores = block.get("rec_scores", [])
                for idx, text in enumerate(rec_texts):
                    score = rec_scores[idx] if idx < len(rec_scores) else 0.0
                    try:
                        confidence = float(score)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    yield str(text).strip(), confidence


def ocr_image(ocr: PaddleOCR, img_path: str) -> str:
    ocr_sig = inspect.signature(ocr.ocr)
    if "cls" in ocr_sig.parameters:
        result = ocr.ocr(img_path, cls=True)
    else:
        result = ocr.ocr(img_path)

    lines = []
    for text, confidence in _iter_ocr_texts(result):
        if text and confidence >= 0.5:
            lines.append(text)
    return "\n".join(lines)


def extract_keyframes(
    video_path: str,
    frame_dir: str,
    sample_every: int,
    diff_threshold: float,
    min_gap_sec: float,
    max_frames: int,
    logger,
    progress_cb=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    prev = None
    frame_index = 0
    save_index = 0
    last_saved_sec = -999.0
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % sample_every != 0:
            frame_index += 1
            if progress_cb and total_frames > 0 and frame_index % 30 == 0:
                progress_cb(min(35, int(frame_index / total_frames * 35)))
            continue

        sec = frame_index / fps
        should_save = False

        if prev is None:
            should_save = True
        else:
            diff = frame_diff_score(prev, frame)
            if diff >= diff_threshold and (sec - last_saved_sec) >= min_gap_sec:
                should_save = True

        if should_save:
            name = f"frame_{save_index:05d}_{sec:.2f}s.jpg"
            path = os.path.join(frame_dir, name)
            cv2.imwrite(path, frame)
            frames.append((name, path, round(sec, 2)))
            save_index += 1
            last_saved_sec = sec

            if save_index % 20 == 0:
                logger(f"已提取关键帧: {save_index}")
            if save_index >= max_frames:
                logger(f"达到 max_frames={max_frames}，停止继续抽帧。")
                break

        prev = frame
        frame_index += 1
        if progress_cb and total_frames > 0 and frame_index % 30 == 0:
            progress_cb(min(35, int(frame_index / total_frames * 35)))

    cap.release()
    logger(f"关键帧提取完成，共 {len(frames)} 张。")
    return frames


def run_pipeline(
    video_path: str,
    out_dir: str,
    sample_every: int,
    diff_threshold: float,
    min_gap_sec: float,
    max_frames: int,
    logger,
    progress_cb=None,
):
    start_time = time.time()

    ensure_dir(out_dir)
    frame_dir = os.path.join(out_dir, "frames")
    ensure_dir(frame_dir)

    logger("开始处理...")
    logger(f"视频: {video_path}")
    logger(f"输出目录: {out_dir}")

    if progress_cb:
        progress_cb(1)

    frames = extract_keyframes(
        video_path=video_path,
        frame_dir=frame_dir,
        sample_every=sample_every,
        diff_threshold=diff_threshold,
        min_gap_sec=min_gap_sec,
        max_frames=max_frames,
        logger=logger,
        progress_cb=progress_cb,
    )

    if progress_cb:
        progress_cb(40)

    logger("初始化 OCR 模型（首次会较慢）...")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    all_rows = []
    frame_logs = []

    total = max(1, len(frames))
    for index, (name, path, sec) in enumerate(frames, start=1):
        text = ocr_image(ocr, path)
        rows = parse_blocks_from_text(text)

        frame_logs.append(
            {"截图": name, "时间秒": sec, "识别记录数": len(rows), "OCR文本": text}
        )
        for row in rows:
            row["截图"] = name
            row["时间秒"] = sec
            all_rows.append(row)

        if index % 10 == 0 or index == total:
            logger(f"OCR进度: {index}/{total}")
        if progress_cb:
            progress = 40 + int(index / total * 50)
            progress_cb(min(90, progress))

    dedup = dedup_rows(all_rows)
    frame_df = pd.DataFrame(frame_logs)
    all_df = pd.DataFrame(all_rows, columns=["品牌", "在营门店数", "口碑值", "人均", "截图", "时间秒"])
    dedup_df = pd.DataFrame(dedup, columns=["品牌", "在营门店数", "口碑值", "人均"])

    if not dedup_df.empty:
        dedup_df = dedup_df.sort_values("在营门店数", ascending=False)

    excel_path = os.path.join(out_dir, "result.xlsx")
    csv_path = os.path.join(out_dir, "result_all_rows.csv")
    json_path = os.path.join(out_dir, "result_dedup.json")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        dedup_df.to_excel(writer, sheet_name="品牌汇总去重", index=False)
        all_df.to_excel(writer, sheet_name="原始识别记录", index=False)
        frame_df.to_excel(writer, sheet_name="帧OCR日志", index=False)

    all_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dedup, f, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100)

    logger("处理完成 ✅")
    logger(f"Excel: {excel_path}")
    logger(f"CSV:   {csv_path}")
    logger(f"JSON:  {json_path}")
    logger(f"截图目录: {frame_dir}")
    logger(f"总耗时: {time.time() - start_time:.1f} 秒")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("小程序录屏自动截图+数据整理")
        self.root.geometry("860x640")
        self.root.minsize(760, 560)

        self.log_queue = queue.Queue()
        self.running = False

        self.video_var = tk.StringVar()
        self.out_var = tk.StringVar(value=os.path.abspath("output_gui"))

        self.sample_var = tk.IntVar(value=8)
        self.diff_var = tk.DoubleVar(value=16.0)
        self.gap_var = tk.DoubleVar(value=0.8)
        self.maxf_var = tk.IntVar(value=600)

        self.build_ui()
        self.root.after(120, self.flush_logs)

    def build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        path_box = ttk.LabelFrame(main, text="输入输出")
        path_box.pack(fill=tk.X, pady=6)

        ttk.Label(path_box, text="视频文件").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(path_box, textvariable=self.video_var, width=80).grid(row=0, column=1, padx=8, pady=8, sticky="we")
        ttk.Button(path_box, text="选择视频", command=self.pick_video).grid(row=0, column=2, padx=8, pady=8)

        ttk.Label(path_box, text="输出目录").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(path_box, textvariable=self.out_var, width=80).grid(row=1, column=1, padx=8, pady=8, sticky="we")
        ttk.Button(path_box, text="选择目录", command=self.pick_out).grid(row=1, column=2, padx=8, pady=8)

        path_box.columnconfigure(1, weight=1)

        param_box = ttk.LabelFrame(main, text="参数（默认可直接用）")
        param_box.pack(fill=tk.X, pady=6)

        ttk.Label(param_box, text="sample_every").grid(row=0, column=0, padx=8, pady=8)
        ttk.Entry(param_box, textvariable=self.sample_var, width=12).grid(row=0, column=1, padx=8, pady=8)

        ttk.Label(param_box, text="diff_threshold").grid(row=0, column=2, padx=8, pady=8)
        ttk.Entry(param_box, textvariable=self.diff_var, width=12).grid(row=0, column=3, padx=8, pady=8)

        ttk.Label(param_box, text="min_gap_sec").grid(row=0, column=4, padx=8, pady=8)
        ttk.Entry(param_box, textvariable=self.gap_var, width=12).grid(row=0, column=5, padx=8, pady=8)

        ttk.Label(param_box, text="max_frames").grid(row=0, column=6, padx=8, pady=8)
        ttk.Entry(param_box, textvariable=self.maxf_var, width=12).grid(row=0, column=7, padx=8, pady=8)

        ctrl = ttk.Frame(main)
        ctrl.pack(fill=tk.X, pady=6)

        self.run_btn = ttk.Button(ctrl, text="开始处理", command=self.start)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="打开输出目录", command=self.open_out_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=4)

        self.progress = ttk.Progressbar(ctrl, mode="determinate", maximum=100)
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

        log_box = ttk.LabelFrame(main, text="运行日志")
        log_box.pack(fill=tk.BOTH, expand=True, pady=6)

        self.log_text = tk.Text(log_box, wrap="word", font=("Consolas", 10))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scroll.set)

        self.append_log("准备就绪。请选择视频并点击“开始处理”。")

    def pick_video(self):
        selected = filedialog.askopenfilename(
            title="选择录屏视频",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.flv"), ("All files", "*.*")],
        )
        if selected:
            self.video_var.set(selected)

    def pick_out(self):
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.out_var.set(selected)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def flush_logs(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, dict) and item.get("type") == "progress":
                    self.progress["value"] = item["value"]
                else:
                    self.append_log(str(item))
        except queue.Empty:
            pass
        self.root.after(120, self.flush_logs)

    def logger(self, msg: str):
        self.log_queue.put(msg)

    def set_progress(self, value: int):
        self.log_queue.put({"type": "progress", "value": value})

    def start(self):
        if self.running:
            messagebox.showinfo("提示", "任务正在运行中，请稍候。")
            return

        video_path = self.video_var.get().strip()
        out_dir = self.out_var.get().strip()

        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror("错误", "请先选择有效的视频文件。")
            return
        if not out_dir:
            messagebox.showerror("错误", "请先选择输出目录。")
            return

        try:
            sample_every = int(self.sample_var.get())
            diff_threshold = float(self.diff_var.get())
            min_gap_sec = float(self.gap_var.get())
            max_frames = int(self.maxf_var.get())
        except Exception:
            messagebox.showerror("错误", "参数格式不正确，请检查。")
            return

        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.append_log("启动任务...")

        def worker():
            try:
                run_pipeline(
                    video_path=video_path,
                    out_dir=out_dir,
                    sample_every=sample_every,
                    diff_threshold=diff_threshold,
                    min_gap_sec=min_gap_sec,
                    max_frames=max_frames,
                    logger=self.logger,
                    progress_cb=self.set_progress,
                )
                self.logger("你现在可以打开 result.xlsx 查看“品牌汇总去重”页。")
            except Exception as exc:
                self.logger("处理失败 ❌")
                self.logger(str(exc))
                self.logger(traceback.format_exc())
            finally:
                self.running = False
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def open_out_dir(self):
        target = self.out_var.get().strip()
        if not target:
            return
        ensure_dir(target)
        try:
            if os.name == "nt":
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                import platform

                if platform.system() == "Darwin":
                    os.system(f'open "{target}"')
                else:
                    os.system(f'xdg-open "{target}"')
        except Exception:
            messagebox.showinfo("输出目录", target)


def run_cli(args):
    run_pipeline(
        video_path=args.video,
        out_dir=args.out,
        sample_every=args.sample_every,
        diff_threshold=args.diff_threshold,
        min_gap_sec=args.min_gap_sec,
        max_frames=args.max_frames,
        logger=print,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="小程序录屏自动截图+数据整理（GUI/CLI）")
    parser.add_argument("--cli", action="store_true", help="使用命令行模式")
    parser.add_argument("--video", default="", help="输入视频路径（CLI模式必填）")
    parser.add_argument("--out", default="output_gui", help="输出目录")
    parser.add_argument("--sample_every", type=int, default=8)
    parser.add_argument("--diff_threshold", type=float, default=16.0)
    parser.add_argument("--min_gap_sec", type=float, default=0.8)
    parser.add_argument("--max_frames", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cli:
        if not args.video:
            raise SystemExit("CLI 模式必须提供 --video")
        run_cli(args)
        return

    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
