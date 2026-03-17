# Refactor Checklist

Ordered by priority and impact.

1. Unify `run_id` generation/registration to avoid mismatched runs between `main.cpp` and `Core/Scheduler.cpp`.
2. Parse and use `device_id` from requests and pass it through camera capture and DB writes.
3. Replace hardcoded serial port `\\.\COM17` with config or API parameter.
4. Fix potential leak in `parse_get` by freeing the `evhttp_uri` object.
5. Return explicit error when scheduler start fails in `/api/control/add`.
6. Enforce or document POST body size limit (8 KB) and return a clear error when exceeded.
7. Fix `Core/sqlite_herpler.cpp` filename typo and update references.
8. Align include paths with actual directory casing (`Detect/` vs `Core/detect`).
9. Align single-image pipeline ROI/QW handling with multi-image pipeline behavior.
10. Fix mismatch between group timeout comment and default value.
