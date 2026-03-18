# Camera Capture Flow (Slide + Camera)

This document defines the target capture workflow for integration with the slide controller (UART1). It is the reference flow for future code changes.

## Overview
Each **group** contains **4 faces**. The workflow is:

1. Load material
2. Move to camera position
3. Capture face 0
4. Rotate to next face
5. Capture face 1
6. Rotate to next face
7. Capture face 2
8. Rotate to next face
9. Capture face 3
10. Unload material
11. Repeat from step 1

## UART1 Commands (from `docs/serial_manual.md`)
- `LOAD`  : `id,20,0,0`
- `GOTO_CAM`: `id,21,0,0`
- `CAM_NEXT`: `id,23,0,0`
- `UNLOAD` : `id,22,0,0`

Recommended response gating:
- Wait for `FLOW LOAD DONE` after `LOAD`
- Wait for `FLOW GOTO_CAM DONE` after `GOTO_CAM`
- Wait for `Cam Next` or `M<id> POS_DONE` after `CAM_NEXT`
- Wait for `FLOW UNLOAD DONE` after `UNLOAD`

## Capture State Machine (per group)

State: IDLE
- Action: send `LOAD`
- Next: WAIT_LOAD

State: WAIT_LOAD
- On `FLOW LOAD DONE` -> send `GOTO_CAM` -> WAIT_GOTO_CAM
- On timeout -> ERROR

State: WAIT_GOTO_CAM
- On `FLOW GOTO_CAM DONE` -> CAPTURE_FACE(0)
- On timeout -> ERROR

State: CAPTURE_FACE(i)
- Action: trigger camera capture (Snap)
- On capture ok:
  - If `i < 3`: send `CAM_NEXT` -> WAIT_CAM_NEXT(i+1)
  - If `i == 3`: send `UNLOAD` -> WAIT_UNLOAD
- On capture fail -> ERROR

State: WAIT_CAM_NEXT(i)
- On `Cam Next` or `M<id> POS_DONE` -> CAPTURE_FACE(i)
- On timeout -> ERROR

State: WAIT_UNLOAD
- On `FLOW UNLOAD DONE` -> IDLE (next group)
- On timeout -> ERROR

State: ERROR
- Action: log, optionally stop scheduler/camera, require manual intervention

## Notes for Implementation
- The existing camera flow currently rotates per shot via serial and sleeps. Replace with UART1 flow above.
- Use event-driven gating rather than fixed sleeps whenever possible. A short safety delay is acceptable after confirmed move completion.
- Group boundary is at 4 faces; only after face 3 do we `UNLOAD` and return to `LOAD`.
- If UART1 reports `Flow Busy`, retry or wait until `FLOW ... DONE` for the last command before issuing a new one.

## Mapping to Current Modules
- Capture loop: `Core/camera_thread.cpp`
- Group completion: `Core/group_manager.cpp`
- Serial control: new/updated serial client (UART1)

This document is the baseline for future refactors of the capture pipeline.
