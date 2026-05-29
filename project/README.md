# Image Registration Framework

A modular, OpenCV-based feature registration playground for Windows + VSCode + CMake. The first slice covers the six classic point-feature methods (SIFT, SURF, ORB, BRISK, KAZE, AKAZE), full BFMatcher with norm auto-routing, RatioTest + CrossCheck filters, and the four staple geometric estimators (Homography, Affine2D, Fundamental, Essential), wrapped in a YAML-driven pipeline.

The architecture is intentionally interface-first so later phases (line / region / deep features, evaluators, datasets) can be slotted in without touching the existing components.

## Layout

```
project/
├── CMakeLists.txt              modern CMake, OpenCV + yaml-cpp
├── main.cpp                    thin entry point -> RegistrationApp
├── apps/                       CLI driver (registration_app)
├── configs/                    YAML for every component & pipeline
│   ├── feature/                sift / surf / orb / brisk / kaze / akaze
│   ├── matcher/                bf
│   ├── filter/                 ratio_test / cross_check
│   ├── geometry/               homography / affine / fundamental / essential
│   └── pipeline/               sift_pipeline / orb_pipeline
├── include/                    public headers
│   ├── interfaces/             pure-virtual contracts
│   ├── core/                   types, context, factory, config, result
│   ├── data/                   FeatureData / MatchData / GeometryData
│   ├── feature/                concrete extractors
│   ├── matcher/                BFMatcher
│   ├── filter/                 RatioTest / CrossCheck
│   ├── geometry/               Homography / Affine / Fundamental / Essential
│   ├── transform/              warpers
│   ├── pipeline/               BasePipeline / FeaturePipeline
│   └── utils/                  logger / timer / yaml_utils / draw_matches
├── src/                        implementations mirrored from include/
└── outputs/                    matches/ and warped/ are written here
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
.\registration_app.exe configs\pipeline\sift_pipeline.yaml path\to\imgA.jpg path\to\imgB.jpg
```

Or rely entirely on the YAML I/O block:

```powershell
.\registration_app.exe configs\pipeline\orb_pipeline.yaml
```

Outputs land in `outputs/matches/` and `outputs/warped/`.

## Switching feature / geometry method

Edit a pipeline YAML (or copy one of the templates). Every component is a single line:

```yaml
feature:  configs/feature/akaze.yaml      # one of: sift, surf, orb, brisk, kaze, akaze
matcher:  configs/matcher/bf.yaml
filters:
  - configs/filter/ratio_test.yaml
  - configs/filter/cross_check.yaml
geometry: configs/geometry/fundamental.yaml   # homography / affine / fundamental / essential
```

The `BFMatcher` reads `descriptor_norm` from the active feature YAML, so SIFT/SURF/KAZE auto-pick `L2` and ORB/BRISK/AKAZE(MLDB) auto-pick `HAMMING` without further configuration.

## Extending

- Add a new extractor: derive from `IFeatureExtractor`, register a string in `Factory::createFeatureExtractor`, drop a YAML in `configs/feature/`.
- Add a new filter / matcher / geometry: same recipe against the matching interface and factory branch.
- Add a new pipeline variant: derive from `BasePipeline` (or implement `IPipeline`) and reuse `Config::loadPipeline`.

The `RegistrationContext` is the single shared mutable state. Every stage receives it by reference and reads/writes the relevant `data/` struct in place, avoiding copy-by-value of large matrices.
