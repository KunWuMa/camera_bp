# BSPC 论文代码复现说明

本目录是可上传 GitHub 的代码发布包，包含私有数据预处理、公开数据读取、
ANN/SNN/ResNet-1D/TCN/Transformer 训练、泄漏审计、统计检验和论文绘图脚本。

## 重要说明

仓库中没有任何真实数据。医院和实验室数据必须经通讯作者及数据管理方同意后
才能获取；公开 rPPG-BP-UKL 数据应从原作者仓库下载，本仓库不重新分发。

上传前务必运行：

```bash
python scripts/check_release.py
```

该检查会阻止常见视频、HDF5、Excel、CSV、NumPy 数组、模型权重和本机绝对路径
进入发布包。

## 推荐执行顺序

```bash
conda env create -f environment.yml
conda activate camera_bp_bspc
python scripts/run_pipeline.py --stage preflight
python scripts/run_pipeline.py --stage private-preprocessing
python scripts/run_pipeline.py --stage primary
python scripts/run_pipeline.py --stage analysis
python scripts/run_pipeline.py --stage figures
```

处理私有数据前必须在当前终端设置至少 16 个字符的随机 `BP_ID_SALT`，且不能将
该值、原始编号与哈希编号的映射表上传 GitHub。详细数据目录、标签格式、实验顺序、
结果文件与论文图表的对应关系见 `docs/`。

公开数据下载后放置为：

```text
data/public/rppg_bp_ukl/rPPG-BP-UKL_rppg_7s.h5
```

医院与实验室数据的申请要求及本地目录结构见 `data/private/README.md`。
