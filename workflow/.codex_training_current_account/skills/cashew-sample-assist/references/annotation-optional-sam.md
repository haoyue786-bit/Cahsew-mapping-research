# Direct labels and SAM2 masks

After [image selection](imagery-selection.md), the [sample runner](../scripts/run_sample.py) calls [annotation](../scripts/annotate.py). Sol receives the full registered 1024 × 1024 RGB and the four-class visual cues. It can make up to ten auxiliary queries per AEF tile to inspect native detail, a wider landscape, reviewed examples, country context or proposal boxes. Sol submits the class, interior points, useful exclusion points and an optional box for each confidently identified unit. It can directly assign one class to an entirely homogeneous tile.

[X-AnyLabeling SAM2](../scripts/xanylabeling_sam2.py) receives the full registered RGB and returns one mask per unit. The program clips the mask to its prompt box and checks it against the positive and negative points. The [two-class workflow](code-selected-annotation.md) fills the confirmed remainder with its class. Other uncovered areas, contradictory masks and unresolved units receive 10.

The program saves source labels and votes by area on the 10 m AEF grid as soon as SAM2 finishes. All cited images are audited before segmentation. The output includes an original-resolution overlay and a 10 m preview.

Operator-provided points or observation windows apply only to the exact case, date and RGB fingerprint. Include them in the annotation input and delivery audit when present. Record durations for Sol, SAM2 and export.
