# On-demand appearance references

The [compact appearance library](../assets/coordinate-reference-library-compact/index.html) includes sparse, variable-density and dense Cashew, woody lookalikes, fields and built-up land. Each example records its marked scope, provider and visible cues. The review site also retains earlier examples and similar records by ID. [Compact library review](http://127.0.0.1:18780/library?scope=compact)

Use `lookup_reference` by class or ID when an example can resolve a visual question. Compare the marked area in a positive example with a close negative. The tool checks live reviews, excluding evaluation-point origins, overlapping query footprints and rejected or flagged records. It supplies the RGB and scope overlay as typed images.

```bash
python scripts/reference_lookup.py --bounds -5.21 8.10 -5.19 8.12 --class-id 2 --limit 2
```

The bounds come from the target AEF footprint in WGS84. To request a specific eligible example, use `--reference-id`. A reference label applies to the marked scope. Judge the target using imagery for its own formal period. [Four-class cues](visual-cues.md) · [Selection windows](imagery-selection.md)
