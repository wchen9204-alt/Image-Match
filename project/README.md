# Image Registration Experimental Platform

This project is a YAML-driven 2D image registration platform built around OpenCV,
C++17, and optional Python deep-learning matchers. It is organized as a set of
replaceable method families:

- Keypoint methods: SIFT, SURF, ORB, BRISK, KAZE, AKAZE.
- Structure methods: edge, line, and contour based pipelines.
- Direct methods: global, frequency, sparse, and dense direct registration.
- Learning methods: LoFTR, SuperPoint + LightGlue, SuperPoint + SuperGlue.

The C++ pipeline owns image loading, configuration, geometry estimation,
warping, metrics, visualization, batch execution, and result summaries. Learning
methods call Python scripts, read a unified matches JSON, and then reuse the same
C++ geometry and evaluation stages.

## Layout

The tree below intentionally uses ASCII characters only, so it remains readable
in terminals and editors with different encodings.

```text
project/
|-- CMakeLists.txt
|-- main.cpp
|-- apps/
|   |-- registration_app.h
|   `-- registration_app.cpp
|-- include/
|   |-- core/
|   |-- data/
|   |-- interfaces/
|   |-- keypoint/
|   |-- structure/
|   |-- direct/
|   |-- learning/
|   |-- matcher/
|   |-- filter/
|   |-- geometry/
|   |-- transform/
|   |-- pipeline/
|   |-- dataset/
|   |-- evaluator/
|   `-- utils/
|-- src/
|   |-- core/
|   |-- data/
|   |-- keypoint/
|   |-- structure/
|   |-- direct/
|   |-- learning/
|   |-- matcher/
|   |-- filter/
|   |-- geometry/
|   |-- transform/
|   |-- pipeline/
|   |-- dataset/
|   |-- evaluator/
|   `-- utils/
|-- configs/
|   |-- keypoint/
|   |-- structure/
|   |-- direct/
|   |-- learning/
|   |-- matcher/
|   |-- filter/
|   |-- geometry/
|   |-- evaluator/
|   `-- pipeline/
|       |-- keypoint/
|       |-- structure/
|       |-- direct/
|       |-- learning/
|       `-- batch/
|-- tools/
|   `-- deep/
|       |-- learning_backend.py
|       |-- loftr_infer.py
|       |-- superpoint_lightglue_infer.py
|       `-- superpoint_superglue_infer.py
|-- third_party/
|   |-- LightGlue/
|   `-- SuperGluePretrainedNetwork/
|-- datasets/
|-- outputs/
`-- build-mingw/
```

For the full directory map, see `PROJECT_DIRECTORY_STRUCTURE_CN.md`.

## Build

The current local build target is MinGW:

```powershell
cmake --build project/build-mingw
```

Main executable:

```text
project/build-mingw/bin/registration_app.exe
```

Core dependencies:

- OpenCV 4.x, including contrib modules used by the project.
- yaml-cpp.
- CMake with C++17 support.

## Single Run

Run one pipeline YAML:

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml
```

Examples by method family:

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/structure/line_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/direct/frequency_direct_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/loftr_learning_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/superpoint_lightglue_learning_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/superpoint_superglue_learning_pipeline.yaml
```

Each pipeline can also override input images and output directory through its
`io` block.

## Batch Run

Batch configs live in:

```text
project/configs/pipeline/batch/
```

Examples:

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_keypoint.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_structure.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_direct.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_learning.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/compare_direct.yaml
```

The batch runner scans the configured dataset root, runs the selected single
pipeline for every sample, and writes a summary CSV.

## Learning Methods

Learning pipelines use this flow:

```text
LearningPipeline
  -> PythonLearningMatcher
  -> learning_backend.py in single or worker mode
  -> matches JSON
  -> C++ correspondence view with source LEARNING
  -> geometry estimation
  -> warp, metrics, visualization, summaries
```

Implemented learning methods:

- `LOFTR`: backed by Kornia LoFTR.
- `SUPERPOINT_LIGHTGLUE`: backed by SuperPoint + LightGlue.
- `SUPERPOINT_SUPERGLUE`: backed by SuperPoint + SuperGlue.

The primary Python entry is:

```text
tools/deep/learning_backend.py
```

It supports two modes:

- `single`: start Python, load one model, process one image pair, then exit.
- `worker`: keep Python alive, load the model once, and process many image pairs.

The three method-specific scripts remain useful as compatibility and debugging
entry points, but the learning YAML configs use the unified backend.

The learning YAML files currently point to:

```text
C:/Users/wangchenyu/AppData/Local/Python/bin/python.exe
```

This avoids the WindowsApps Python alias and starts the real Python interpreter.

Python dependencies already needed by the learning scripts include:

- torch
- torchvision
- opencv-python
- kornia
- certifi

Source-only Python dependencies are stored under `third_party/` and are not
compiled by CMake:

```text
project/third_party/LightGlue
project/third_party/SuperGluePretrainedNetwork
```

The Python scripts add these directories to `sys.path` automatically.

## Outputs

Single-run outputs are written under:

```text
project/outputs/single/<method_family>/<pipeline>/<sample>/
```

Batch outputs are written under:

```text
project/outputs/batch/<method_family>/<pipeline>/
```

Typical files include:

- original images
- all-match visualization
- inlier visualization
- warped image
- blend image
- `run_summary.txt`
- `run_summary.json`
- batch `summary.csv`

## Extending

Common extension points:

- Add a keypoint extractor by implementing `IKeypointExtractor` and registering
  it in the factory.
- Add a structure extractor or associator through the structure interfaces.
- Add a direct aligner through `IDirectAligner` and direct pipeline config.
- Add a learning matcher by providing a Python script that writes the unified
  matches JSON format.
- Add or switch geometry estimators through the geometry YAML configs.

The shared `RegistrationContext` carries image data, matches, geometry, warp
results, metrics, and run summaries across pipeline stages.
