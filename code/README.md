# 实验代码

大学生教育应激与亚临床心理危机四波纵向研究的采集、预处理和分析代码。设计与设备参数见上级目录的 `论文框架与实验设计.md`、`实验设备参数手册.docx`。

## 目录

```text
config/study_config.yaml       全部阈值与设备设置
acquisition/rest_eeg_protocol.py  静息脑电流程（配合 BrainVision Recorder）
acquisition/stress_task.py     心电流程时间线与算术应激任务
preprocessing/score_scales.py  量表评分
preprocessing/ecg_hrv.py       心电 → ecg_lab.csv
preprocessing/eeg_features.py  脑电 → eeg_lab.csv、eeg_channel_level.csv
preprocessing/wearable_nightly.py  Oura、Polar H10 → 夜间表
preprocessing/ema_passive.py   EMA 与手机被动数据
preprocessing/qc_report.py     分析前质控
analysis/main_models.R         测量模型、RI-CLPM、有调节的中介、亚型、稳健性
analysis/prediction_agreement.py  设备一致性与样本外预测
analysis/aux_fiml.R            辅助变量 FIML（饱和相关法），被两个主分析脚本调用
analysis/supplement_models.R   设计要求的补充分析（Holm、替代模型、多组、LPA 全指标、burst 多层模型）
analysis/dsem_burst.py         burst 数据的贝叶斯 DSEM（NumPyro）
analysis/ri_lta.py             随机截距潜转变分析 RI-LTA（JAX）
analysis/supplement_prediction.py  校准、净获益、SHAP、次要结局、流程与依从性
analysis/lab_capacity.py       实验室产能核算
preprocessing/safety_monitor.py    三源风险触发清单
preprocessing/export_bids.py   EEG 导出为 BIDS
run_pipeline.py                串联运行
figures/make_figures.py        从 06_Source_Data.xlsx 生成 Fig. 1–6（PDF、PNG、可编辑 PPT）；依赖 matplotlib 3.10、python-pptx、openpyxl、lxml、Arial 字体
```



## 环境

```bash
conda create -n expcode -c conda-forge python=3.11
conda activate expcode
pip install -r requirements.txt
```

```bash
conda create -n riclpm -c conda-forge r-base=4.4 r-lavaan r-mclust r-psych r-semtools r-data.table
```

```bash
conda create -n bayes -c conda-forge python=3.11
conda activate bayes
pip install numpyro jax arviz pandas scipy
```

分析时设置 Rscript 和该环境的 Python 路径（PowerShell）：

```powershell
$env:RSCRIPT = "C:\Users\<用户名>\.conda\envs\riclpm\Lib\R\bin\x64\Rscript.exe"
$env:BAYES_PYTHON = "C:\Users\<用户名>\.conda\envs\bayes\python.exe"
```

## 每次实验室测试

1. 佩戴 actiCAP slim 32，在 BrainVision Recorder 中查看阻抗（低于 30 kΩ），开始保存为 `<pid>_w<wave>`。Recorder 会把阻抗写入 `.vhdr`。
2. 运行静息流程：

```bash
python acquisition/rest_eeg_protocol.py --pid P0001 --wave 0 --out raw/eeg --trigger-port COM5
```

3. 开始 BIOPAC 记录后运行任务，时间零点要与导出文件的第一个样本对齐：

```bash
python acquisition/stress_task.py --pid P0001 --wave 0 --out raw/ecg
```

4. 确认 `.vhdr`、`.eeg`、`.vmrk` 和 `.json` 都已保存到 `raw/eeg`。

## 原始数据放置

```text
raw/participants.csv        pid, tier, sex, grade, age, ctq_total …
raw/scale_items.csv         pid, wave, 条目 <前缀>_<序号>, psqi_total, caffeine_cups_day, exercise_hours_week
raw/clinician_assessment.csv  pid, wave, cgi_s, cssrs_*, rater_id
raw/safety_events.csv       pid, wave, contacted, …
raw/burst_schedule.csv      pid, wave, start_date
raw/ecg/<pid>_w<wave>.acq 或 .txt（列 ecg, resp）与 <pid>_w<wave>_segments.csv
raw/eeg/<pid>_w<wave>.vhdr、.eeg、.vmrk 与 .json
raw/oura/<pid>.json         Oura sleep 文档；用 --fetch-oura 拉取时需 raw/oura/tokens.csv（pid, token）
raw/polar/<pid>_w<wave>_d<第几天>.txt   每行一个 RR（ms）
raw/ema_raw.csv             pid, wave, burst_day, prompt, date, stress, neg_affect, pos_affect, ideation_item
raw/sensor_raw.csv          pid, timestamp, screen_on_seconds, steps, lat, lon
raw/exam_days.csv           pid, date（可选）
```

## 运行

```bash
python run_pipeline.py --raw raw --processed processed --results results
```

只跑质控和分析：

```bash
python run_pipeline.py --processed processed --results results --steps qc analysis
```

表名带统一前缀时加 `--prefix`，例如 `--prefix study_`。

主分析结果在 `results/`：`t4_riclpm_within_paths.csv`（个体内路径）、`results_summary.txt`（拟合、中介、调节、亚型、稳健性）、`t7_prediction.csv`（预测）、`s38_dsem.csv`（DSEM）、`s36_rilta_fit.csv`（RI-LTA）。

## 已知注意事项

**RR 校正**：未使用 `nk.signal_fixpeaks`。已确认它会移动正常心搏且少报校正数，使 RMSSD 偏低约 5%。改用局部中位数 20% 规则（`preprocessing/common.py`）。

**坏道判定**：在 20–45 Hz 判断噪声，避免把 alpha 强的后部通道判为坏道。

**参考电极**：离线把 FCz 恢复为有效通道并转平均参考，分析用 33 个通道。

**亚型指标**：`main_models.R` 用基线水平和峰值跳变做 LPA。

## 代码验证情况

| 模块                     | 验证方式                    | 结果                       |
| ---------------------- | ----------------------- | ------------------------ |
| RMSSD                  | 已知 RR 序列的心电，真值 11–72 ms | 最大误差 0.9%                |
| HF 功率                  | 0.25 Hz 正弦调制 RR         | 786 ms²（理论 800）          |
| 漏搏                     | 去掉 30 个心搏               | 校正 7.2%，按规则判无效           |
| 脑电特征                   | 已知 10.2 Hz alpha 的信号    | 峰频 10.20 Hz，强弱和 FAA 方向正确 |
| 坏道                     | 噪声 T7 + 平坦 P3           | 准确检出，无误判                 |
| `rest_eeg_protocol.py` | 不接硬件运行                  | 流程和记录正常                  |
| BrainVision 文件读取       | 标准 `.vhdr` 三件套，含阻抗与打标   | 打标、阻抗、参考恢复正确             |
| 其余预处理、质控               | 小样本端到端运行                | 通过                       |
| 分析脚本                   | 在完整表格上运行                | 全部表格正常输出                 |
