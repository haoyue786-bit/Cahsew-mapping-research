# Keep review of existing masks

Use the class-colored overview and the earlier raw RGB to check the proposed groups for a visible class or boundary mismatch. Group IDs `G0`, `G1`, `G2` and `G3` refer to cropland/open land, other trees/wetland/water, Cashew and built-up land. Mask IDs and their suggested classes are fixed.

Return only `keep` and `omit` arrays. Use a group ID to keep the whole group, or mask IDs to keep selected masks. Use `omit` for individual exceptions within the retained selection. Leave out masks with mixed or unresolved cover, a spatially wrong outline or weak evidence for the suggested class. Return empty arrays if no mask should be kept. Complete the review from the supplied images in one response.
