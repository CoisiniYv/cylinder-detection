# Architecture Overview

## Scope
This document describes the current runtime architecture of the inspection service in `E:\ydk\op\mt`. It reflects observed code behavior and is intended for developers and operators.

## High-Level Runtime
1. Program entry in `main.cpp` parses CLI args and loads `config.json` via `Core/Config.cpp`.
2. SQLite database is initialized and schema is ensured in `Core/db_utils.cpp`.
3. A `run_id` is generated and registered.
4. HTTP server starts in `Core/Server.cpp` and exposes REST-like endpoints.
5. Endpoints start or stop the camera + detection pipelines.

## Core Components
| Component | Files | Responsibility |
| --- | --- | --- |
| Config | `Core/Config.hpp`, `Core/Config.cpp`, `config.json` | Load host/port, model/output paths, database path |
| HTTP Server | `Core/Server.hpp`, `Core/Server.cpp` | libevent server, request parsing, task control APIs |
| Scheduler | `Core/Scheduler.hpp`, `Core/Scheduler.cpp` | Orchestrates camera, workers, group manager, DB writes |
| Camera Thread | `Core/camera_thread.hpp`, `Core/camera_thread.cpp` | Camera SDK integration, group capture, serial rotation |
| Detect Thread | `Core/detect_thread.hpp`, `Core/detect_thread.cpp` | Per-thread CUDA inference pipeline (crop, Halcon, YOLO, SAM) |
| Single Detect | `Core/single_detect_thread.hpp`, `Core/single_detect_thread.cpp` | Single-image inference pipeline for API usage |
| Group Manager | `Core/group_manager.hpp`, `Core/group_manager.cpp` | Aggregates 4-face results, handles timeouts, notifies camera |
| Queue Manager | `Core/queue_manager.hpp`, `Core/queue_manager.cpp` | Raw/result concurrent queues and blocking pop |
| DB Utilities | `Core/db_utils.hpp`, `Core/db_utils.cpp`, `Core/sqlite_helper.hpp`, `Core/sqlite_herpler.cpp` | SQLite initialization, run registration, SQL helper |
| Types | `Core/image_types.hpp` | Common image/result structures |

## Data Flow
### Multi-Face Pipeline (Camera)
1. `/api/control/add` stores parameters into server state and starts `Scheduler`.
2. `CameraThread` captures 4 images per group and pushes `ImageFrame` into the raw queue.
3. `DetectThread` workers pop raw frames, run crop and inference, and push `SingleImageResult` to the result queue.
4. `GroupManager` aggregates 4 faces, marks group GOOD/NG, and triggers DB write and next group capture.

### Single-Image Pipeline
1. `/api/single_detect/add` loads an image from disk.
2. A dedicated `SingleDetectThread` runs crop, Halcon, slice detection, and SAM segmentation.
3. Results are returned in the HTTP response and images are saved under the run output directory.

## Threading Model
- HTTP server runs in the libevent loop thread started by `Server::start`.
- `CameraThread` runs a capture loop thread when started.
- `DetectThread` runs N worker threads, each with its own CUDA context and model instance.
- `GroupManager` runs one aggregation thread.
- `SingleDetectThread` runs one worker thread on demand.

## Configuration
The service reads `config.json` and expects these keys:
- `ip`: Host IP to bind the HTTP server.
- `analyzerPort`: Port to bind.
- `modelDir`: Base directory for model files.
- `uploadDir`: Output directory for saved images.
- `dbPath`: SQLite database file path.

If `dbPath` is empty, the default is `my_inspection.db` in the working directory.

## Storage (SQLite)
Tables created in `ensure_db_initialized`:
- `inspection_runs`: Each program run (primary key `run_id`).
- `inspection_groups`: Group-level status (GOOD/NG).
- `inspection_faces`: Per-face record with saved paths.
- `defect_detections`: Per-detection record with bbox, length, area.

## Output Files
- Multi-face pipeline saves per-group images under `uploadDir\{run_id}\{group_id}`.
- Single capture saves `uploadDir\{run_id}\single\camera_image.png`.
- Single detect saves outputs under `uploadDir\{run_id}\single`.

## Logging
`Core/Utils/Log.hpp` defines `LOGI`, `LOGE`, `LOGC` for structured console logging.

## Dependencies
- OpenCV (CPU + CUDA)
- CUDA runtime
- libevent
- jsoncpp
- SQLite3
- Camera SDK (MPHdc_API)
- trtyolo (TensorRT)
- Halcon
- SAM segmenter

## Notes and Caveats
The items below reflect current code behavior and are candidates for cleanup.
- `run_id` is generated in `main.cpp` and again in `Core/Scheduler.cpp`, which can create mismatched runs.
- `device_id` is not parsed from HTTP requests and defaults to `CAM-001`.
- Serial port is hardcoded to `\\.\COM17` in `Core/Scheduler.cpp`.
- `parse_get` does not free the `evhttp_uri` object, which can leak.
- POST body is capped at 8 KB (`RECV_BUF_MAX_SIZE`).
- `Core/sqlite_herpler.cpp` has a naming typo (should be `sqlite_helper.cpp`).
- Case-sensitive systems may break due to `Detect/` include vs `Core/detect` folder.
