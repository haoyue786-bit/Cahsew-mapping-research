# 晚期自动标注环境

训练和标注运行在同一台机器上。标注 Skill 从有日期的高分辨率影像判读地类，需要边界细化时把带点或框提示的图像送给本机 SAM2。返回的标签投影回对应的 AEF 原始 128 × 128 网格。

## 服务依赖

1. 准备 Python 3.10 环境用于 SAM2。原服务记录的 PyTorch 和 torchvision 是 `2.5.1+cu121`、`0.20.1+cu121`。按上游服务的安装说明准备运行依赖和 SAM2。 [X-AnyLabeling Server](https://github.com/CVHub520/X-AnyLabeling-Server) · [SAM 2](https://github.com/facebookresearch/sam2)
2. 将服务源码放在资产根目录下的 `outputs/cashew_sample_assist_on_demand_context_20261116/_tools/remote_model_screen/X-AnyLabeling-Server-main`。原实验的 `app/models/segment_anything_2.py` SHA‑256 为 `abd6d6e0a6d3fa432cf810909f8556698014e8e7f2da326dce1372b667ee3412`。
3. 从官方渠道取得 `sam2.1_hiera_large.pt`，放在资产根下的 `outputs/tgo_visual_annotation_v4_20261115/_tools/sam2_weights/`，SHA‑256 为 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`。运行目录内的服务会写入实际插件、权重及设备身份。 [官方模型权重](https://github.com/facebookresearch/sam2#download-checkpoints)
4. 在训练 Python 中安装 Playwright 和 Chromium，或通过 `CASHEW_BROWSER_PYTHON` 指向装有 Playwright 和影像依赖的独立 Python 环境。查看日期与坐标注册依赖本机可访问的影像服务。 [Playwright 安装说明](https://playwright.dev/python/docs/intro)

## 启动

```bash
# 在安装了 X-AnyLabeling Server 和 SAM2 依赖的 Python 环境中运行
CUDA_VISIBLE_DEVICES=0 python \
  "$cashew_work/outputs/cashew_reference_library_v2_20261117/runtime/backend/serve_x_sam2.py"
```

服务监听 `127.0.0.1:19574`。如果服务与训练共用一个 GPU，可在调度终端将服务的进程号写入 `CASHEW_SAM_PID`，或用 `materialize.py --sam-unit` 记录对应的 systemd 用户单元。调度器继续检查显存余量。

如果使用 Esri 历史影像，可设置 `CASHEW_ESRI_PROXY`。Google 历史影像获取器使用 `CASHEW_GEHI_BIN` 及可选的 `CASHEW_GEHI_PROXY`。使用相应提供方授权的影像。参考图库可额外接入，用于对照疑难地类。

训练组在验收标注时检查日期、影像身份、预测轮次、原始网格及有效像元。样本被跳过的原因和已验收掩膜留在对应组的材料目录。
