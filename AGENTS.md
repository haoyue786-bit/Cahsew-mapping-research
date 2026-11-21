# Cashew mapping research

Read the root README, the manuscript supplement, and the reproduction guide before running the workflow. The materialized workspace has the actual runtime and bound Skills.

For a training turn, use `cashew-fada-training-expert` from the materialized training account. The runtime sends the structured Skill input and matching thread base instructions. The binding verifier checks both before the turn. Use `cashew-sample-assist` for late-period visual annotation.

Keep target evaluation point attributes, predictions, scores, thresholds, and derived rankings outside the training process and all Agent evidence. Preparation may use the point geometries to exclude annotation cases. Supply only the SHP, SHX, and PRJ components to the training workspace. Use the DBF and any target point metrics only in the offline evaluator, after verifying the frozen STOP/SELECT and map identities.

Preserve the versioned per-tile year choices and group membership when reproducing the study. The earlier period uses the same-group late-period frozen model weights and a fresh optimizer. It cannot reuse the later annotation library.
