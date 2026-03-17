# API Reference

## Base URL
`http://{ip}:{analyzerPort}` from `config.json`.

## Response Envelope
All endpoints return JSON with a common envelope:
```json
{
  "code": 0,
  "msg": "OK",
  "data": { }
}
```
`code` is `0` for success and negative for errors. Some endpoints return `data` only on success.

## Endpoints
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Health and queue stats |
| GET, POST | `/api/control/add` | Start multi-face camera detection |
| GET | `/api/control/cancel` | Cancel multi-face detection |
| GET, POST | `/api/single_detect/add` | Run single-image detection |
| GET | `/api/single_detect/cancel` | Stop single-image worker |
| GET | `/api/camera/single_capture` | Capture one camera image |
| GET | `/api/camera/status` | Query camera status |
| GET | `/api/camera/stop` | Stop and release camera thread |

## Parameter Model (DetectParams)
The following parameters are accepted by `/api/control/add` and `/api/single_detect/add` via GET query or POST JSON. Defaults are the current server defaults.

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `model_name` | string | empty | If relative, joined with `modelDir`. If empty, `engine_file` becomes `modelDir`.
| `input_image_path` | string | empty | Required for `/api/single_detect/add` only.
| `enable_four_side_crop` | bool | false | Enables ROI crop.
| `crop_x` | int | 0 | ROI left.
| `crop_y` | int | 0 | ROI top.
| `crop_width` | int | 0 | ROI width.
| `crop_height` | int | 0 | ROI height.
| `is_qw` | bool | false | Enables QW circle crop logic.
| `circle_x1` | int | 0 | QW circle 1 center X.
| `circle_y1` | int | 0 | QW circle 1 center Y.
| `circle_x2` | int | 0 | QW circle 2 center X.
| `circle_y2` | int | 0 | QW circle 2 center Y.
| `radius` | int | 0 | QW circle radius.
| `enable_fourier_transform` | bool | false | Enables stripe removal.
| `filter_width` | int | 10 | FFT filter width.
| `attenuation_factor` | double | 0.0001 | FFT attenuation factor.
| `target_angle` | int | 90 | FFT target angle.
| `angle_tolerance` | int | 10 | FFT angle tolerance.
| `enable_denoising` | bool | false | Enables denoising.
| `denoise_h` | float | 5.0 | Denoise strength.
| `denoise_hColor` | float | 8.0 | Denoise color strength.
| `denoise_search_window` | int | 17 | Denoise search window.
| `denoise_template_window` | int | 11 | Denoise template window.
| `slice_width` | int | 640 | Slice width.
| `slice_height` | int | 640 | Slice height.
| `slice_distance` | int | 40 | Slice spacing.
| `nms_threshold` | float | 0.45 | Must be > 0.
| `conf_threshold` | float | 0.20 | Must be > 0.
| `pix_to_mm` | double | 0.021 | Pixel to mm scale.
| `min_area_mm2` | double | 0.0 | SAM min area filter.
| `min_diameter_mm` | double | 0.0 | SAM min diameter filter.
| `delay_ms` | int | 0 | Camera delay between shots.
| `qw_index` | int | 0 | QW camera index (0-3).
| `device_id` | string | `CAM-001` | Currently not parsed from requests and always defaults.

Note on SAM models: SAM paths are not request parameters. They are derived from `modelDir` and fixed names `SAM\SAM_encoder.engine` and `SAM\SAM_mask_decoder.engine`, with ONNX fallbacks.

## GET vs POST
- GET: parameters are read from the query string.
- POST: parameters are read from a JSON body (max size 8 KB).

## `/api/health`
Method: GET

Response `data`:
- `status`: `ok`
- `raw_queue_size`: approximate raw queue length
- `result_queue_size`: approximate result queue length
- `host`, `port`, `modelDir`, `uploadDir` if config is loaded

Example response:
```json
{
  "code": 0,
  "msg": "OK",
  "data": {
    "status": "ok",
    "raw_queue_size": 0,
    "result_queue_size": 0,
    "host": "10.37.57.112",
    "port": 9003,
    "modelDir": "E:\\ydk\\resource\\models",
    "uploadDir": "E:\\ydk\\resource\\output"
  }
}
```

## `/api/control/add`
Method: GET or POST

Purpose: Start multi-face camera detection. Parameters are stored in server state and used by the scheduler.

Response codes:
- `0`: accepted
- `-1`: request failed
- `-3`: `nms_threshold` or `conf_threshold` invalid (<= 0)

Example GET:
```
GET /api/control/add?model_name=v12.engine&delay_ms=100&qw_index=2
```

Example POST JSON:
```json
{
  "model_name": "v12.engine",
  "delay_ms": 100,
  "qw_index": 2,
  "nms_threshold": 0.45,
  "conf_threshold": 0.2
}
```

Note: If scheduler start fails internally, the current implementation still returns success.

## `/api/control/cancel`
Method: GET

Purpose: Cancel the current multi-face detection task and stop the scheduler.

Response codes:
- `0`: canceled
- `-1`: request failed

## `/api/single_detect/add`
Method: GET or POST

Purpose: Run detection on a single image file.

Required parameter:
- `input_image_path`

Response codes:
- `0`: success
- `-10`: missing `input_image_path`
- `-11`: file not found
- `-3`: `nms_threshold` or `conf_threshold` invalid
- `-12`: detection failed

Response `data`:
- `meta`: `{ device_id, group_id, index_in_group }`
- `detections`: list of `{ label_id, confidence, box, length, area }`
- `saved_path`: visualization path
- `skeleton_path`: skeleton path

Note: Current single-image pipeline ignores ROI and QW crop parameters and always uses the full image.

## `/api/single_detect/cancel`
Method: GET

Purpose: Stop and release the single-image detection thread.

## `/api/camera/single_capture`
Method: GET

Purpose: Capture a single camera image and save it as PNG.

Response codes:
- `0`: success
- `-20`: camera start failed
- `-21`: capture failed
- `-22`: output directory creation failed
- `-23`: image save failed

Response `data`:
- `saved_path`: saved image path
- `camera_running`: `true` or `false`

## `/api/camera/status`
Method: GET

Purpose: Return whether the camera thread is running.

Response `data`:
- `camera_running`: `true` or `false`

## `/api/camera/stop`
Method: GET

Purpose: Stop and release the camera thread.

Response codes:
- `0`: stopped
- `-1`: camera thread not found
