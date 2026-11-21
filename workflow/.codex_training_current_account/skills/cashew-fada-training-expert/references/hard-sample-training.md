# Cashew visual training samples

## Acquisition and intake

A chip is an original 128×128 AEF tile fragment. Choose acquisition budgets using candidate count, existing library coverage, observed image acceptance and prior response. Two probability screens seed the candidate pool:

- Competing uses Cashew in the top two, with a gap of at most 0.10 to the strongest other class and at least 0.55 combined probability.
- Diffuse covers a near tie among three or four classes, including Cashew within 0.12 of the leader, where the leading probability is at most 0.45.
- Both start with 5% valid chip coverage and a contiguous 64-pixel region. Prefer persistent uncertainty over multiple checkpoint screens and distribute selections across tiles. These are acquisition criteria.
- After the first acquisition, compare confident predictions with a source-only AEF embedding clue for a small missed/extra Cashew audit. A spatially spread random check can reveal other failures. Treat the hypotheses as inspection requests. The imagery reviewer receives only location and period.
- Start with backups from the same category. An image can provide accepted labels without resolving the main hard patch. A recorded duplicate, skip, shortfall or service issue changes both the expected turnaround and the expected evidence. Review actual category support before increasing volume.

The design draws on spatially balanced strata in the authors' [Strategic Sampling code](https://github.com/Remote-Sensing-of-Land-Resource-Lab/Training-Sample-Selection/blob/9293c3003afc213a6e75affb9f00bc27770bbd11/func/sample_temp/sample_crop.py), top-two and other uncertainty scores in [BSSAL](https://github.com/YXu556/BSSAL/blob/e978826ea907028c17155d140fbf181ea23c75d6/utils.py#L245-L351), and separate loss/dose accounting across classes inspired by [RS-DWL](https://github.com/zhu-xlab/RS-DWL/blob/fe6504b40c282c69535b3c4bc70307f24eaccecd/train.py#L146-L180). The trainer uses labels from image review, with the observed acquisition date, georeferencing, provenance and ambiguity mask.

## Training and decision evidence

- `annotated_training` shows the active library revision, actual optimizer steps and pixel counts by class, raw auxiliary loss, weighted/raw gradient comparison on the decoder/fusion modules, and checkpoint-bound training fits. Use `training`, `controls`, `source`, `unlabelled_panel`, `module_detail`, `module_gradients` and `interventions` as needed. Query relevant epochs explicitly.
- `hard_sample_weight` is the coefficient on the separately normalized four-class loss. `hard_sample_repeats` is the maximum number of exposures per accepted chip in an epoch when capacity permits. Compare exposure and loss per active update. Zero weight has zero supervised updates; fit still measures the corresponding predictor.
- For gradual/adversarial controls, compare scheduled and executed states. Match the intervention horizon to the underlying mechanism. An epoch currently represents around 63 generator updates for standard groups. Inspect source segmentation, local and global alignment and inference BN separately. Gradient norms, gradient cosine and optimizer changes are different measurements.
- Compare a set of earlier masks against the same masks now. Intake resets what constitutes the newest cohort; the raw loss of a newly added difficult label has a first-use baseline. Candidate selection uses common masks and reports prior sample exposure. A source-preservation concern or sustained novel failure may outweigh better fit on repeated cases.
