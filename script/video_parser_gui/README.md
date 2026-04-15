# miniapp_video_parser_gui

一个用于“小程序录屏自动截图 + OCR识别 + 数据整理”的 GUI 工具。

## 功能

- 从录屏中自动抽取关键帧（按画面变化）
- OCR 识别中文界面文本
- 抽取品牌、在营门店数、口碑值、人均（可继续扩展）
- 导出 `result.xlsx` / `result_all_rows.csv` / `result_dedup.json`
- GUI 一键使用，同时支持 CLI 模式

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动 GUI

```bash
python miniapp_gui_parser.py
```

## CLI 模式

```bash
python miniapp_gui_parser.py --cli --video your_video.mp4 --out output
```

## 打包为可执行文件（Windows/macOS/Linux）

先安装打包工具：

```bash
pip install pyinstaller
```

在当前目录执行：

```bash
pyinstaller --noconfirm --onefile --windowed --name miniapp_video_parser_gui miniapp_gui_parser.py
```

打包完成后可执行文件位于：

- Windows: `dist/miniapp_video_parser_gui.exe`
- macOS/Linux: `dist/miniapp_video_parser_gui`

## 参数说明

- `sample_every`: 每 N 帧采样一次，越小越精细，速度越慢
- `diff_threshold`: 帧差阈值，越大则截图越少
- `min_gap_sec`: 两次截图最小时间间隔（秒）
- `max_frames`: 最多保留截图数量

## 注意事项

- PaddleOCR 首次初始化会比较慢。
- 不同分辨率/字体可能影响识别率，建议录屏保持清晰。
- 如需适配更多字段（如 GMV、订单量、转化率），可在 `parse_blocks_from_text` 增加规则。
