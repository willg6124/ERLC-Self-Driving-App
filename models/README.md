# Optional: real object-detection weights

`erlc_autopilot/perception/object_detection.py` works out of the box with
a dependency-light HSV/contour heuristic detector (no files needed here).

For better accuracy against the real game, drop a pretrained MobileNet-SSD
(Caffe) model into this folder and it's picked up automatically:

- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`

These are the standard, widely-mirrored MobileNet-SSD (VOC-trained)
weights used all over OpenCV tutorials/projects — search for either
filename to find a mirror, e.g. the `chuanqi305/MobileNet-SSD` or
`C-Aniruddh/realtime_object_recognition` GitHub repos. Not vendored here
to keep the repo small and avoid license ambiguity.

Without these files, or on the **Lite** model tier (which skips them on
purpose for speed even if present), the app automatically falls back to
the heuristic detector — everything still works, just a bit less robust
to real-world clutter than a trained network would be.
