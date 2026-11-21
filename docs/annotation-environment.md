# Late-Period Annotation Environment

Training and annotation run on the same host. The annotation skill interprets land cover from dated high-resolution imagery. When a boundary needs refinement, it sends the image and point or box prompts to a local SAM 2 service. The workflow projects the resulting labels back to the original 128 × 128 AEF grid.

## Service dependencies

1. Use Python 3.10 for SAM 2. The recorded PyTorch and torchvision versions were `2.5.1+cu121` and `0.20.1+cu121`. Follow the upstream installation instructions for the server dependencies and SAM 2. [X-AnyLabeling Server](https://github.com/CVHub520/X-AnyLabeling-Server) · [SAM 2](https://github.com/facebookresearch/sam2)
2. Place the server source under `outputs/cashew_sample_assist_on_demand_context_20261116/_tools/remote_model_screen/X-AnyLabeling-Server-main` in the asset root. In the original experiment, `app/models/segment_anything_2.py` had a SHA-256 digest of `abd6d6e0a6d3fa432cf810909f8556698014e8e7f2da326dce1372b667ee3412`.
3. Download `sam2.1_hiera_large.pt` from the official source and place it in `outputs/tgo_visual_annotation_v4_20261115/_tools/sam2_weights/` under the asset root. The SHA-256 digest is `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`. The service records the plugin, weights, and device it uses. [Official SAM 2 checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints)
4. Install Playwright and Chromium in the training Python environment, or set `CASHEW_BROWSER_PYTHON` to an executable in a separate environment with Playwright and the imagery dependencies. Acquisition date verification and coordinate registration require access to imagery services from the host. [Playwright installation](https://playwright.dev/python/docs/intro)

## Start the service

```bash
# Run in the Python environment with X-AnyLabeling Server and SAM 2 dependencies.
CUDA_VISIBLE_DEVICES=0 python \
  "$cashew_work/outputs/cashew_reference_library_v2_20261117/runtime/backend/serve_x_sam2.py"
```

The service listens on `127.0.0.1:19574`. When the service shares a GPU with training, set `CASHEW_SAM_PID` to its process ID in the scheduler terminal, or register its systemd user unit with `materialize.py --sam-unit`. The scheduler monitors available GPU memory.

For historical Esri imagery, set `CASHEW_ESRI_PROXY` as needed. The Google historical imagery downloader uses `CASHEW_GEHI_BIN` and optionally `CASHEW_GEHI_PROXY`. Use imagery authorized by the corresponding provider. An authorized reference image library can help interpret difficult land-cover classes.

Before accepting an annotation, the training job checks the acquisition date, imagery identity, prediction epoch, original grid, and valid pixels. Each group directory stores the accepted masks and reasons for skipping samples.
