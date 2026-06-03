# Image Registration Framework

A modular, OpenCV-based feature registration playground for Windows + VSCode + CMake. The first slice covers the six classic point-feature methods (SIFT, SURF, ORB, BRISK, KAZE, AKAZE), full BFMatcher with norm auto-routing, RatioTest + CrossCheck filters, and the four staple geometric estimators (Homography, Affine2D, Rigid2D, Similarity2D), wrapped in a YAML-driven pipeline.

The architecture is intentionally interface-first so later phases (line / region / deep features, evaluators, datasets) can be slotted in without touching the existing components.

## Layout

```
project/
鈹溾攢鈹€ CMakeLists.txt              modern CMake, OpenCV + yaml-cpp
鈹溾攢鈹€ main.cpp                    thin entry point -> RegistrationApp
鈹溾攢鈹€ apps/                       CLI driver (registration_app)
鈹溾攢鈹€ configs/                    YAML for every component & pipeline
鈹?  鈹溾攢鈹€ feature/                sift / surf / orb / brisk / kaze / akaze
鈹?  鈹溾攢鈹€ matcher/                bf
鈹?  鈹溾攢鈹€ filter/                 ratio_test / cross_check
鈹?  鈹溾攢鈹€ geometry/               homography / affine / rigid / similarity
鈹?  鈹斺攢鈹€ pipeline/               sift_pipeline / orb_pipeline
鈹溾攢鈹€ include/                    public headers
鈹?  鈹溾攢鈹€ interfaces/             pure-virtual contracts
鈹?  鈹溾攢鈹€ core/                   types, context, factory, config, result
鈹?  鈹溾攢鈹€ data/                   KeypointData / KeypointMatchData / StructureMatchData / GeometryData
鈹?  鈹溾攢鈹€ feature/                concrete extractors
鈹?  鈹溾攢鈹€ matcher/                BFMatcher
鈹?  鈹溾攢鈹€ filter/                 RatioTest / CrossCheck
鈹?  鈹溾攢鈹€ geometry/               Homography / Affine / Rigid / Similarity
鈹?  鈹溾攢鈹€ transform/              warpers
鈹?  鈹溾攢鈹€ pipeline/               BasePipeline / KeypointPipeline
鈹?  鈹斺攢鈹€ utils/                  logger / timer / yaml_utils / draw_matches
鈹溾攢鈹€ src/                        implementations mirrored from include/
鈹斺攢鈹€ outputs/                    matches/ and warped/ are written here
```

## Build (Windows / VSCode + CMake Tools)

Prerequisites:

- Visual Studio 2019 / 2022 with the C++ workload (MSVC 19.x)
- CMake >= 3.16
- OpenCV 4.x with `xfeatures2d` (any prebuilt that includes `opencv_xfeatures2dXXX.lib`)
- yaml-cpp prebuilt or installed via vcpkg

Set the environment variables once so `find_package` can locate them, e.g.:

```powershell
setx OpenCV_DIR "C:\opencv\build"          # contains OpenCVConfig.cmake
setx CMAKE_PREFIX_PATH "C:\yaml-cpp\install"
```

Or, when using vcpkg:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
    -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
```

Without vcpkg:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
    -DOpenCV_DIR="C:/opencv/build" `
    -DCMAKE_PREFIX_PATH="C:/yaml-cpp/install"
cmake --build build --config Release
```

The build copies `configs/` next to the produced `registration_app.exe` so the relative paths inside the pipeline YAMLs work out of the box.

## Run

```powershell
cd build\bin\Release
.\registration_app.exe configs\pipeline\keypoint\sift_pipeline.yaml path\to\imgA.jpg path\to\imgB.jpg
```

Or rely entirely on the YAML I/O block:

```powershell
.\registration_app.exe configs\pipeline\keypoint\orb_pipeline.yaml
```

Outputs land in `outputs/matches/` and `outputs/warped/`.

## Switching feature / geometry method

Edit a pipeline YAML directly. For SIFT, switch the geometric model by modifying the `geometry` line inside `configs/pipeline/keypoint/sift_pipeline.yaml`:

```yaml
keypoint: configs/keypoint/akaze.yaml      # one of: sift, surf, orb, brisk, kaze, akaze
matcher:  configs/matcher/bf.yaml
filters:
  - configs/filter/ratio_test.yaml
  - configs/filter/cross_check.yaml
geometry: configs/geometry/homography.yaml    # homography / affine / rigid / similarity
```

The `BFMatcher` reads `descriptor_norm` from the active feature YAML, so SIFT/SURF/KAZE auto-pick `L2` and ORB/BRISK/AKAZE(MLDB) auto-pick `HAMMING` without further configuration.

## Extending

- Add a new extractor: derive from `IKeypointExtractor`, register a string in `Factory::createKeypointExtractor`, drop a YAML in `configs/keypoint/`.
- Add a new filter / matcher / geometry: same recipe against the matching interface and factory branch.
- Add a new pipeline variant: derive from `BasePipeline` (or implement `IPipeline`) and reuse `Config::loadPipeline`.

The `RegistrationContext` is the single shared mutable state. Every stage receives it by reference and reads/writes the relevant `data/` struct in place, avoiding copy-by-value of large matrices.

