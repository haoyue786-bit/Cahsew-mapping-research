# Run sample annotation

Use the annotation account's installed skill and the training Python environment. Provide the original georeferenced AEF128 TIFF, country, formal period and output directory. Use a new sample ID containing letters, digits, underscores or hyphens.

```bash
cd @WORKSPACE@/.codex_training_current_account/skills/cashew-sample-assist
export CASHEW_GEHI_BIN=/path/to/GEHistoricalImagery
python scripts/run_sample.py \
  --aef /path/to/original_AEF128.tif --case CASE_ID --country CIV \
  --period 2025 --run-dir /path/to/new_run
```

Use `--prefer-dry-season` to rank available dry scenes first on a tile already known to contain difficult woodland. The host still downloads and scores the date-qualified inventory. Sol can request a different local image before SAM2. Review `CASHEW_EVIDENCE.json` for the per-unit visual evidence and any proposals withheld.

The Codex app-server uses the configured annotation account and optional `CASHEW_CODEX_PROXY`. The first hosted turn is the land-cover plan. Its explicit skill input and base instructions bind `gpt-6.2-sol` at `max`. Check the returned model, effort and standard tier (`serviceTier=default`) in `thread_start_response.json`. The visual audit verifies image bytes in the recorded rollout. Repeat requests for identical evidence within that session return the previously registered image ID.

The standard first turn starts with one image, the full primary RGB. Use `view_details` for up to four native-detail boxes and `lookup_reference` for reviewed examples, optionally including a lookalike class. Request a wider view or regional background when it can resolve a doubt. `view_comparison` is offered when an aligned, date-qualified alternate is ready. Identical image requests reuse local results. Sol can submit a homogeneous label, simple polygons, or individually processed SAM2 units prompted by points or a box.

For a confirmed two-class scene, see the [foreground and background rule](code-selected-annotation.md). The host fills the confirmed remainder after SAM2 and keeps the explicitly uncertain areas and unresolved foreground footprints at class 10.

Add `--woody-review` when the selected area is a difficult woody mosaic. An independent Sol turn checks supported woody units at their marked interiors and records the available alternate date. Local clouds or unresolved crown identity can leave a unit as class 10. This option is used with the standard geometry workflow. For the detector and DINOv2 comparison use `--region-matching` as a separate mode.

Resume local work after a failure in a numbered annotation attempt containing `LOCAL_PLAN.json`. The frozen plan and already verified masks provide the retry inputs.

```bash
python scripts/run_sample.py \
  --resume-local /path/to/new_run/CASE_ID/annotation/002
```

X-AnyLabeling runs locally at `http://127.0.0.1:19574`. The optional proposal detector and local DINOv2 use `CASHEW_FIELDS_PYTHON` and `CASHEW_FIELDS_GPU`. The local DINOv2 trial expects the official `facebookresearch_dinov2_main` checkout under `~/.cache/torch/hub` and `dinov2_vits14_pretrain.pth` in its `checkpoints` directory. [Selection](imagery-selection.md) · [Segmentation](annotation-optional-sam.md)

Add `--region-matching` for the box-guided trial. The detector uses the full registered RGB and the X-AnyLabeling service produces one mask per box. All boxes are retained after screening. Requests with many boxes use bounded batches on the same image embedding. The first Sol turn selects one representative mask per confirmed class. DINOv2 compares interior features of the other masks. Woodland suggestions require verified dry imagery and separate Cashew and other-woodland representatives. A second Sol turn receives one group overview and returns only the IDs to retain. When there are no new suggestions, use the representatives already confirmed in the first turn. The point-guided workflow runs when detection provides no usable masks.

| Record | Contents |
| --- | --- |
| `CASE_ID/imagery/SEARCH.json` | Full local candidate ranking, date cutoff, quality and acquisition timing |
| `CASE_ID/imagery/{Google,Esri}/…/SELECTION.json` | Candidate RGB, actual dates, registration, quality measurements and season |
| `CASE_ID/annotation/…/LOCAL_PLAN.json` | Frozen Sol plan and visual evidence for resuming local work |
| `CASE_ID/annotation/…/woody_review/RECEIPT.json` | Independent review evidence, retained or withheld units and review time |
| `CASE_ID/annotation/…/result/` | RGB PNG, label PNGs, overlay, source label TIFF and 10 m label TIFF |
| `CASE_ID/annotation/…/proposal_masks/SAM_PROPOSALS.json` | Box and SAM2 mask counts, full-RGB predictor receipts and duration |
| `CASE_ID/annotation/…/REGION_REVIEW.json` | Local suggestions, selected groups and retained masks for the trial |
| `CASE_ID/RECEIPT.json` | Outcome, files, cumulative auxiliary queries and stage durations |

Check `PROGRESS.json` for annotation stages. The receipt records image preparation, Sol, SAM2, token usage and any reused masks. If both providers run out of eligible scenes, the receipt records the period and reason. Recover completed work from the plan, mask receipts and the recorded turn. When a hosted turn is unresolved, reconcile its saved thread and terminal status before sending another request.
