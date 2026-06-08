<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore Changelog

All notable changes to the NCore project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

- - -

## [v19.3.0](https://github.com/NVIDIA/ncore/compare/c69e32a2e3d576745211e3dd66f9fa8142037154..v19.3.0) - 2026-06-05

### Highlights

- nuScenes V4 converter + lidar model tooling (#128): Full nuScenes dataset support in NCore V4 format, shipping with a new generic lidar model library and evaluation tool:
  - Converts nuScenes to V4 (closing #123), covering 6 cameras, HDL-32E lidar, 5 radars, and 3D cuboid annotations. Achieves 0.029deg far-range angular error on the lidar model - on par with PAI extraction quality.
  - Change was a little involved on the lidar model side as nuScenes does only expose motion-compensated data without per-point timestamps, but effort resulted in a couple of additional re-usable features / tools:
    - Structured lidar model estimation library ([docs](https://github.com/NVIDIA/ncore/tree/main/tools/data_converter#structured-lidar-model-extraction)): Generic, reusable library for deriving spinning lidar models from motion-compensated point clouds. Works without raw per-point timestamps (but still relies on some additional assumptions for now). Composable steps for azimuth extraction, column alignment, resolution upsampling, and multi-frame optimization. HDL-32E presets included.
    - Lidar model evaluation [tool](https://nvidia.github.io/ncore/tools/lidar_model_eval.html) (`ncore_evaluate_lidar_model`): CLI tool that compares model-predicted vs. native ray directions on any sequence and validates against target thresholds. Reports angular error (mean/median/p95), per-row and per-frame breakdowns, and optionally renders overlay and error heatmap images onto camera frames.
- Generic Rolling-shutter solver refactor (#119): Extracted a shared and abstract `RollingShutterSolver` implementation from previously hardcoded camera and lidar models into a generic utility class, unifying both (camera convergence was based on pixel-deltas, whereas lidar uses the more general time-convergence criteria working for all sensors). This is groundwork for eventually adding support for solid-state lidars (#125). No observerable public behavior change, but includes a deeper [analysis](https://github.com/NVIDIA/ncore/pull/119#issuecomment-4553040552) of the convergence pattern we see in practice for currently "time-linear" camera/lidar sensors, and also implements additional higher-order (approximate) differential solver variants that would handle non-linear measurement time readout behaviors better also.

- OpenCV fisheye `compute_max_angle` fix (#140): Replaced Newton-Raphson inversion with a hybrid analytical solver that correctly handles non-monotone distortion polynomials (images wider than the valid FOV). Up to 1667x faster by determining inflexion points via Eigenvalue decomposition; fixes nonsensical output angles (e.g. 451 deg) for affected cameras (example image of now-supported fisheye lenses of the model).  
Note that the API location changed from the model to the parameter pack, which is a mild breaking change, but as arguments remain compatible and usage is likely low opted for a minor instead of a major version bump.

#### ➕ Added
- (**data_converter**) add structured lidar model extraction library - ([d818a9e](https://github.com/NVIDIA/ncore/commit/d818a9e13dd01637d94edd1f0f644a053da791c9)) - Janick Martinez Esturo
- (**nuscenes**) add nuScenes to NCore V4 converter - ([827506d](https://github.com/NVIDIA/ncore/commit/827506d36f38f926ec348a9b89bd900dd08aca8d)) - Janick Martinez Esturo
- (**tools**) add lidar model evaluation tool - ([b694dea](https://github.com/NVIDIA/ncore/commit/b694dea445421747f9bfe5bb1b11b61a205fd0d9)) - Janick Martinez Esturo
- Add OpenCV-based colormap utilities, remove matplotlib dependency - ([c8fdd35](https://github.com/NVIDIA/ncore/commit/c8fdd350635504e111e4f8ec510b6d55a57034ac)) - Janick Martinez Esturo
#### 🪲 Fixed
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) (**data**) respect monotonicity in OpenCV fisheye compute_max_angle - ([c67ecfa](https://github.com/NVIDIA/ncore/commit/c67ecfac779539831ba21c40bc72b7f14de9cd08)) - Janick Martinez Esturo
- (**pillow**) Update Pillow to 12.2.0 to address CVE-2026-42311 - ([1497969](https://github.com/NVIDIA/ncore/commit/149796983a4fdb909be57e57c156d6d10ed01de8)) - Janick Martinez Esturo
- Lockfile update - ([0f82eae](https://github.com/NVIDIA/ncore/commit/0f82eaed635be626fa0b0e5f06931b7010392e11)) - Janick Martinez Esturo
#### ⚡ Performance
- (**ci**) drop repository-cache to eliminate 3 min download overhead - ([e234b54](https://github.com/NVIDIA/ncore/commit/e234b54a4cefcc7be137ef38a539bfceac00f343)) - Janick Martinez Esturo
- (**ci**) commit MODULE.bazel.lock to skip PyPI resolution in CI - ([0498325](https://github.com/NVIDIA/ncore/commit/049832552f120d7c432a717f305f8d36145cb6b5)) - Janick Martinez Esturo
#### 🔄 Changed
- (**sensors**) extract shared rolling-shutter solver into generic utility - ([c1e21ea](https://github.com/NVIDIA/ncore/commit/c1e21ea034eef6f75ea4540f72d6bd59149688c2)) - Janick Martinez Esturo
#### ⚙️ CI
- remove redundant tag trigger from CI workflow - ([c69e32a](https://github.com/NVIDIA/ncore/commit/c69e32a2e3d576745211e3dd66f9fa8142037154)) - Janick Martinez Esturo
#### 🏗️ Build
- (**bazelignore**) Ignore .venv directory in .bazelignore - ([bc07978](https://github.com/NVIDIA/ncore/commit/bc079781f73f8ab644a4f592445b462fe5b91879)) - Janick Martinez Esturo

- - -

## [v19.2.1](https://github.com/NVIDIA/ncore/compare/0cc85ea8a054470f223106ebf3ffe3a5d704c1e5..v19.2.1) - 2026-05-21

### Highlights

- Remove `ffmpeg` dependency in favor of `PyNvVideoCodec` for H.264 decode (used in PAI converter)

- Support empty chunk arrays (some recent use cases might produce empty arrays, e.g., camera-labels)

- Fix various security and static type issues

#### ➕ Added
- (**components**) support empty chunk arrays via _normalize_chunks - ([b4e9a3c](https://github.com/NVIDIA/ncore/commit/b4e9a3c7ba6981bacaaf82dbca563630cd2632ee)) - Janick Martinez Esturo
- (**pai**) replace imageio-ffmpeg with PyNvVideoCodec for H.264 decode - ([711ff8e](https://github.com/NVIDIA/ncore/commit/711ff8ebf84c6e61376174daae545964884ab538)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**ci**) restrict cache-save to push events only - ([808e776](https://github.com/NVIDIA/ncore/commit/808e776798c66835c8c1ca5f7bfca78b7b2a1e7b)) - Janick Martinez Esturo
- (**components**) replace eval() with ast.literal_eval() for safe metadata parsing - ([e31874b](https://github.com/NVIDIA/ncore/commit/e31874ba2e8a23b0405e5ee4e50d19f31b66716a)) - Janick Martinez Esturo
- (**components**) replace mutable default arguments with safe alternatives - ([499b509](https://github.com/NVIDIA/ncore/commit/499b5090186741ce99a9537ea2e3e62a4a4e49fe)) - Janick Martinez Esturo
- (**types**) correct dtype validation in BivariateWindshieldModelParameters - ([0cc85ea](https://github.com/NVIDIA/ncore/commit/0cc85ea8a054470f223106ebf3ffe3a5d704c1e5)) - Janick Martinez Esturo
#### 📚 Documentation
- (**deps**) add licensing notes for imageio-ffmpeg dependency - ([6ea118e](https://github.com/NVIDIA/ncore/commit/6ea118eb423094d8b63871823057db6b251dc363)) - Janick Martinez Esturo
- fix typos in windshield distortion model comments - ([59bfc70](https://github.com/NVIDIA/ncore/commit/59bfc708a6bcac1fd660391a933d71c262717f71)) - Janick Martinez Esturo
#### ⚙️ CI
- add merge_group triggers for GitHub merge queue - ([1a9889f](https://github.com/NVIDIA/ncore/commit/1a9889f6a8fef339ed35bec35077cc47161f6166)) - Janick Martinez Esturo
#### 🏗️ Build
- (**bazelignore**) Add .worktrees to .bazelignore - ([43c05b5](https://github.com/NVIDIA/ncore/commit/43c05b5239c9d0dc65135c13d1c273a62a7d1336)) - Janick Martinez Esturo
#### 🔧 Chore
- (**tools**) remove unused transform_point_cloud import - ([76ee08f](https://github.com/NVIDIA/ncore/commit/76ee08f72ce719d262764248132f7bd2750f4ce9)) - Janick Martinez Esturo
- (**types**) add missing return type annotations across impl/ - ([19622dc](https://github.com/NVIDIA/ncore/commit/19622dc4346cd0ab48355c60e458dde9dd152e5b)) - Janick Martinez Esturo

- - -

## [v19.2.0](https://github.com/NVIDIA/ncore/compare/d606bf47b50b5350562f485b6e4dd0a57c44fdef..v19.2.0) - 2026-05-12

### Highlights

- Add support for KITTI raw dataset conversion to NCore V4 format, enabling users to easily convert and utilize the popular KITTI dataset in the latest NCore format with native support for camera labels and efficient storage.

- Add component-level generic_data support in NCore V4, allowing components to store and access arbitrary user-defined data in a flexible and extensible way, enabling new use cases and easier integration of custom data types without modifying the core library.

- Replaced mypy with ty for static type checking, resulting in significantly faster type checking times (approximately 9x faster) while maintaining strong type safety and improving developer productivity.

#### ➕ Added
- (**data_converter**) add KITTI raw dataset to NCore V4 converter - ([f425aa9](https://github.com/NVIDIA/ncore/commit/f425aa97392b317fe282ca475ffabd7528f4fee9)) - Janick Martinez Esturo
- (**v4**) add component-level generic_data support - ([5ec4d7b](https://github.com/NVIDIA/ncore/commit/5ec4d7b798d37449c7fd7c88313161f631fe7e32)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**format**) use portable shebang in ruff_isort.sh - ([a5024ee](https://github.com/NVIDIA/ncore/commit/a5024ee92e96da3a7cd84b022f476edcf9125e92)) - Janick Martinez Esturo
- (**ncore_vis**) handle grayscale camera images via PIL conversion - ([c7826d2](https://github.com/NVIDIA/ncore/commit/c7826d241102ea70898d0708141b334c1fa6a271)) - Janick Martinez Esturo
- (**ncore_vis**) default cuboid source filter to first available source - ([23f435f](https://github.com/NVIDIA/ncore/commit/23f435fc0c834dac638616dfa2cafb35c744d386)) - Janick Martinez Esturo
#### 🏗️ Build
- (**deps**) replace archived rules_proto with protobuf 30.0, bump TF - ([40e0b2d](https://github.com/NVIDIA/ncore/commit/40e0b2dfb3ccbe7b574eac07308bafedd2e998fa)) - Janick Martinez Esturo
- remove mypy in favor of ty (~9x faster type checking) - ([d606bf4](https://github.com/NVIDIA/ncore/commit/d606bf47b50b5350562f485b6e4dd0a57c44fdef)) - Janick Martinez Esturo

- - -

## [v19.1.1](https://github.com/NVIDIA/ncore/compare/5a278ff4d250f0ee6c1a2a81bdeb7c40367c2f90..v19.1.1) - 2026-05-08

This is a bugfix release to allow using the V4 compat APIs in `PYTHONOPTIMIZE=1` settings.

#### 🪲 Fixed
- (**data**) extract walrus-operator assignments from assert statements - ([5a278ff](https://github.com/NVIDIA/ncore/commit/5a278ff4d250f0ee6c1a2a81bdeb7c40367c2f90)) - Janick Martinez Esturo

- - -

## [v19.1.0](https://github.com/NVIDIA/ncore/compare/bb21e4a5ed5eb686be50ee29498eac84ce785574..v19.1.0) - 2026-05-07

### Highlights

- Add new first class citizen V4 `CameraLabelsComponent` for native camera labels and `CameraLabelsProtocol` for unified access.
  
  The new component is based on a generic tagged-union label type system, supports array and image-encoded labels, as well as optional quantization and compression for efficient storage.
  
  Migrate Waymo panoptic segmentation to CameraLabelsComponent and add support for visualization in `ncore_vis`.

- Enable GPU-based unit tests in the CI using NVIDIA self-hosted runners with Blackwell GPU support, and upgrade PyTorch to 2.7+cu128 for Python 3.11 to ensure compatibility.

- Fix an itar regression to serialize PointCloudsComponent metadata, and preload it in the reader to avoid redundant I/O and improve performance.

- Update waymo converstion to support partial sequence conversions and refine visualizations in `ncore_vis`.

#### ➕ Added
- (**ncore_vis**) Add option to recenter world coordinates near the origin - ([9f20e3a](https://github.com/NVIDIA/ncore/commit/9f20e3af67b2d29b1ea8fc7a3025d2aafef75ecf)) - Janick Martinez Esturo
- (**waymo**) add --seek-sec and --duration-sec CLI options for time-restricted conversion - ([b3e6695](https://github.com/NVIDIA/ncore/commit/b3e6695838fcd644cabdbdfc67afe3481f044a7a)) - Janick Martinez Esturo
- add camera label overlay support to ncore_vis - ([7b7f228](https://github.com/NVIDIA/ncore/commit/7b7f228bbadda728f81008bfaf0cbde11aeab3be)) - Janick Martinez Esturo
- migrate Waymo panoptic segmentation from generic_data to CameraLabelsComponent - ([177b9ec](https://github.com/NVIDIA/ncore/commit/177b9ec24d2ebae77f2571b50882942d422c551e)) - Janick Martinez Esturo
- add CameraLabelsComponent with tagged-union type system, compat layer, and tests - ([8b8a607](https://github.com/NVIDIA/ncore/commit/8b8a6076147ca1ddab7c6f2f46c8d86104cda8c2)) - Janick Martinez Esturo
- upgrade PyTorch to 2.7+cu128 for Python 3.11 with Blackwell GPU support - ([8f85584](https://github.com/NVIDIA/ncore/commit/8f855847c473edc52b5b218f5190f2f635f7e189)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**ci**) move --stamp before -- so Bazel processes it - ([fb18a16](https://github.com/NVIDIA/ncore/commit/fb18a1610022bb8b3be31e0968774f02cebbb5f6)) - Janick Martinez Esturo
- (**ty**) Fix ty-reported static type issues - ([b0f54f0](https://github.com/NVIDIA/ncore/commit/b0f54f02c1ca2c0575bd3b76d4e4ca9510ca8810)) - Janick Martinez Esturo
- (**unit-tests**) consistently check all store types - ([77144a9](https://github.com/NVIDIA/ncore/commit/77144a9b2b4bca1c84703e48031a7f6f0aa65bbd)) - Janick Martinez Esturo
- (**unit-tests**) Prevent relative imports in tests - ([e4de909](https://github.com/NVIDIA/ncore/commit/e4de909d1d0ca53b4f878abbc4b40a01484b3e47)) - Janick Martinez Esturo
- (**waymo-converter**) skip frames that are outside of the sequence timestamp interval - ([f4b9d65](https://github.com/NVIDIA/ncore/commit/f4b9d65acd6a5dc575e90120bfa6150853b27ce6)) - Janick Martinez Esturo
- Preload point cloud component metadata in reader - ([13bbf78](https://github.com/NVIDIA/ncore/commit/13bbf78c2df302fb1e767d2f4bfc3fdb786e6d59)) - Janick Martinez Esturo
- Point cloud component meta data group - ([52fe0cd](https://github.com/NVIDIA/ncore/commit/52fe0cdfc36f5d94450ecdbbcb82be0cafa2adbc)) - Janick Martinez Esturo
#### 🔄 Changed
- (**v4-components**) use module-level logger - ([ae7e990](https://github.com/NVIDIA/ncore/commit/ae7e99008cfb4c68c596c727a2143753fa97680e)) - Emmanuel Attia
#### 📚 Documentation
- add project site link to README - ([041f003](https://github.com/NVIDIA/ncore/commit/041f00338abbea0a20075096032d7bc34af55789)) - Janick Martinez Esturo
#### ⚙️ CI
- switch CI to NVIDIA self-hosted runners - ([9372229](https://github.com/NVIDIA/ncore/commit/9372229269dfa0a6036c41636efb8efe99892308)) - Janick Martinez Esturo
- add copy-pr-bot configuration for self-hosted runners - ([bb21e4a](https://github.com/NVIDIA/ncore/commit/bb21e4a5ed5eb686be50ee29498eac84ce785574)) - Janick Martinez Esturo
#### 🏗️ Build
- Specify wheel version via embed label - ([cc440a4](https://github.com/NVIDIA/ncore/commit/cc440a46fe45e2a543e42ff899bb0e1165dbae6d)) - Janick Martinez Esturo
- Suppress rules_py warning on RECORD file changes - ([bcebfde](https://github.com/NVIDIA/ncore/commit/bcebfdeeeb041852301d4a4a413a052fe2444123)) - Janick Martinez Esturo
#### 🧪 Tests
- add V4 compat layer coverage for camera labels, radar sensor, and get_sequence_meta - ([d91fcc9](https://github.com/NVIDIA/ncore/commit/d91fcc9593b7fb53e00f2741c6edfbb63cfc1a75)) - Janick Martinez Esturo

- - -

## [v19.0.0](https://github.com/NVIDIA/ncore/compare/34aabcc3e0d7d55aff7e9434d6b5e091549405fd..v19.0.0) - 2026-04-24

### Highlights

- Add new first class citizen V4 `PointCloudsComponent` for native point clouds and `PointCloudsSourceProtocol` for unified
  point cloud (native / lidar / radar) access. Support transformation of per-point attributes in a consistent way
  for invariant (no transformation) / direction-like (rotation only) and point-like (full transformation) attributes.

  Mild breaking change only for data-converters having to provide the list of native point-clouds (similar to sensors)
  in `ComponentGroupAssignments.create()` (using an empty list will be sufficient for most cases - this is just to not
  deviate from the existing sensor conventions).

- Add further `.itar` init performance improvements by re-using tail-buffer for lookup of compressed consolidated meta-data
  (if possible)

#### ➕ Added
- (**build**) Set use_default_shell_env to True for Sphinx actions - ([f1d8741](https://github.com/NVIDIA/ncore/commit/f1d8741125db0bd7155292ecd1a1c44f03b94d5e)) - Janick Martinez Esturo
- (**converters**) add --world-global-mode option for optional world_global pose storage - ([8408067](https://github.com/NVIDIA/ncore/commit/8408067db4e9cdff490a25648d4b760eeda29aa3)) - Janick Martinez Esturo
- ![BREAKING](https://img.shields.io/badge/BREAKING-red) add PointCloudsComponent with PointCloud type, PointCloudsSourceProtocol, V4 storage, adapter, tools, and visualizer - ([8acc007](https://github.com/NVIDIA/ncore/commit/8acc007946d6bb95264ad971290516bd04b3160a)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**docs**) update ncore_docs_data to v0.6 to fix missing camera.jpg image - ([34aabcc](https://github.com/NVIDIA/ncore/commit/34aabcc3e0d7d55aff7e9434d6b5e091549405fd)) - Janick Martinez Esturo
#### ⚡ Performance
- IndexedTarStore cache tail read to avoid duplicate I/O for in-range keys - ([6a989f2](https://github.com/NVIDIA/ncore/commit/6a989f277917d843ac037a9e2fdc016020bb7a32)) - Emmanuel Attia
#### 📚 Documentation
- add S3 read performance section with benchmark results - ([30483b2](https://github.com/NVIDIA/ncore/commit/30483b2f31053d829805e9561dba39e2aee04877)) - Janick Martinez Esturo

- - -

## [v18.9.0](https://github.com/NVIDIA/ncore/compare/cf53a926c0410dae0ed50cbe8dcac96f1d696894..v18.9.0) - 2026-04-14

### Highlights

- Add support for radar sensors in the PAI converter and visualization tools

  ![radar-6x-half-short](https://github.com/user-attachments/assets/95a94c04-7934-4196-bc81-7cae0beaab24)

- Add performance improvements when accessing cloud storage to the IndexedTarStore, improving initialization performance by 2x

#### ➕ Added
- Allow optional itar index_tail_read_size optional - ([a96915d](https://github.com/NVIDIA/ncore/commit/a96915d5467f61c9a4b84a749dd8b0e78afb1374)) - Janick Martinez Esturo
- add radar sensor support to PAI converter and ncore_vis - ([e7b4a2e](https://github.com/NVIDIA/ncore/commit/e7b4a2e57c01f2b9ce221b4085ee6433239fd504)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**converters**) remove superfluous identity world_global pose from waymo and colmap converters - ([cf53a92](https://github.com/NVIDIA/ncore/commit/cf53a926c0410dae0ed50cbe8dcac96f1d696894)) - Janick Martinez Esturo
#### ⚡ Performance
- (**stores**) optimize IndexedTarStore open with single-read index and lazy TarFile - ([e24a13e](https://github.com/NVIDIA/ncore/commit/e24a13e1eba24f5fa433240997c49bda2d05cc72)) - Emmanuel Attia
#### 🔄 Changed
- expose index tail read size parameter, add tests, refine logic - ([400bd40](https://github.com/NVIDIA/ncore/commit/400bd402109bf939f1352521c85908ceaa9e18b8)) - Janick Martinez Esturo
- remove the _NullTarFile stub, only instantiate TarFile in write mode - ([87e9874](https://github.com/NVIDIA/ncore/commit/87e987437d5ca7c1e739cc90564774aae0116950)) - Janick Martinez Esturo

- - -


## [v18.8.0](https://github.com/NVIDIA/ncore/compare/9de4d6cd368ba00699c467cbd1fa13646e2d58e9..v18.8.0.1) - 2026-03-31

### Highlights

- Added `compute_max_angle` computation API for OpenCV fisheye camera model,
  which can be used to determine the default valid field of view for projection
  and visualization
- Added support to convert Scannet++ datasets to NCore via colmap converter
- Correctly interpolate cuboid tracks for camera overlays in `ncore_vis`
- Fix PAI conversion to validate presence of required offline features and
  update to default HuggingFace dataset revision 'main'
- Extend itar documentation on performance and cloud storage access

#### ➕ Added
- (**colmap**) add OPENCV_FISHEYE support, configurable paths, and scannetpp-v4 subcommand - ([6c5beee](https://github.com/NVIDIA/ncore/commit/6c5beee8cd16f68d5e8e17fc47abd3b20c812ca9)) - Janick Martinez Esturo
- (**colmap**) support per-image masks in colmap converter - ([7e05282](https://github.com/NVIDIA/ncore/commit/7e052820c345a0a247d7539c6ca5dab883cb37d0)) - Janick Martinez Esturo
- (**colmap**) Include downsampled sensors by default - ([ca411b9](https://github.com/NVIDIA/ncore/commit/ca411b9d7b40b6e338ae0facf05cd393c676b2be)) - Janick Martinez Esturo
- (**ncore_vis**) Interpolate cuboid tracks to camera mid-frame timestamp - ([afae717](https://github.com/NVIDIA/ncore/commit/afae7170f03b3611c7c963bfbf72cb58532cb883)) - Janick Martinez Esturo
- (**opencv-fisheye**) compute_max_angle for OpenCVFisheyeCameraModel - ([ac30d7a](https://github.com/NVIDIA/ncore/commit/ac30d7a262158ef9ddf949452d7f6e19f65cc308)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**PAI-conversion**) Validate presence of required offline features - ([397cbc6](https://github.com/NVIDIA/ncore/commit/397cbc627b11a14740e26c5a6d9599615c741bde)) - Janick Martinez Esturo
- (**camera_test**) Fix ruff E402 import not at top of file - ([68650e1](https://github.com/NVIDIA/ncore/commit/68650e1fd41b4967605076146fae81d3ea36e222)) - Janick Martinez Esturo
- (**colmap**) use se3_inverse and correct transformation names - ([0a26d69](https://github.com/NVIDIA/ncore/commit/0a26d692a8e412cfbfab4b8d6a5d7b71e2c20185)) - Janick Martinez Esturo
- (**colmap**) pass image_path to pycolmap SceneManager to suppress warning - ([9b8935b](https://github.com/NVIDIA/ncore/commit/9b8935b71729197f01636f903683c1bed93aa7af)) - Janick Martinez Esturo
- (**colmap**) Use flags for boolean options in converter - ([25730d9](https://github.com/NVIDIA/ncore/commit/25730d9f8b9de591ff3b93e959a98e46613bcacf)) - Janick Martinez Esturo
- (**deps**) bump cbor2 >= 5.9.0 for Python 3.9+ to address CWE-674 / CVE-2026-26209 - ([ff30199](https://github.com/NVIDIA/ncore/commit/ff30199e01dc22b6448523e129dd7175403797a0)) - Janick Martinez Esturo
- (**ncore_vis**) Fix cuboid overlay projection and eager track initialization - ([4c3d50f](https://github.com/NVIDIA/ncore/commit/4c3d50f961b86afc0e47ca21d4262a9cae113ce5)) - Janick Martinez Esturo
- (**ncore_vis**) Use frame_idx consistently - ([f9a3a14](https://github.com/NVIDIA/ncore/commit/f9a3a14b26d4ae91ae635c533b0bbc03fae5a30a)) - Janick Martinez Esturo
- (**ncore_vis**) Replace se3_inverse with get_frames_T_source_sensor in lidar projection - ([b953857](https://github.com/NVIDIA/ncore/commit/b953857e27fad5000c49bd0f5df2ad69a2e94107)) - Janick Martinez Esturo
- (**ncore_vis**) Use per-cuboid observation timestamps for projection and rendering - ([9de4d6c](https://github.com/NVIDIA/ncore/commit/9de4d6cd368ba00699c467cbd1fa13646e2d58e9)) - Janick Martinez Esturo
- (**pai**) Update to default HF revision 'main', update caching - ([f040823](https://github.com/NVIDIA/ncore/commit/f040823d6c84c78024c9adb840392449df3e90ab)) - Janick Martinez Esturo
- (**pycolmap**) patch Python 3 map() compatibility in text format parsers - ([2d36b2a](https://github.com/NVIDIA/ncore/commit/2d36b2af70bd16c35be5c172482217acaefd851f)) - Janick Martinez Esturo
- (**ty**) Fix type error in context manager exit method - ([d07839e](https://github.com/NVIDIA/ncore/commit/d07839e5d9ebcc37d9aa9a7cb0173b1301ff7229)) - Janick Martinez Esturo
#### 📚 Documentation
- (**colmap**) document storate of `rgb` point colors in generic lidar frame data - ([06f904e](https://github.com/NVIDIA/ncore/commit/06f904e7da43a2121bf4943006fb5da346a70131)) - Janick Martinez Esturo
- add itar read performance benchmark to storage documentation - ([9c5d194](https://github.com/NVIDIA/ncore/commit/9c5d1947111ed900a5e8f51e4fc5f091d9c8289a)) - Janick Martinez Esturo
- split formats.rst into data formats and storage/access pages, add cloud storage - ([42a4420](https://github.com/NVIDIA/ncore/commit/42a44203a7736af234f42d6efcaf484568b31063)) - Janick Martinez Esturo

- - -

## [v18.7.0](https://github.com/NVIDIA/ncore/compare/84e13dfc7dc773eb065d0f5182641d32b20cdcbd..v18.7.0) - 2026-03-17

### Highlights
- Added timestamp interval filtering and caching to cuboid tracks compat API to accelerate local or repeated lookups (mostly intended to accelerate older non-OSS NCore data-format versions like V3)

#### ➕ Added
- add timestamp_interval_us filtering to get_cuboid_track_observations() - ([9832a44](https://github.com/NVIDIA/ncore/commit/9832a441d05f1da5ca37c99bf1e934cc25700496)) - Janick Martinez Esturo
#### 🪲 Fixed
- (**colmap**) Don't include downsampled images by default - ([27856f8](https://github.com/NVIDIA/ncore/commit/27856f89f3a65e99fe71af3090376cbf5b9b030f)) - Janick Martinez Esturo
#### 📚 Documentation
- (**conventions**) Refine specs of coordinate systems and transformations - ([84e13df](https://github.com/NVIDIA/ncore/commit/84e13dfc7dc773eb065d0f5182641d32b20cdcbd)) - Janick Martinez Esturo
- (**lidar-model**) Document that per-row azimuth offsets are optional / can be zero if applicable - ([a76bccb](https://github.com/NVIDIA/ncore/commit/a76bccb8885ce233e8d60c19ba5c109a5380cec7)) - Janick Martinez Esturo
#### ⚙️ CI
- Prevent execution of non-build/test jobs in forks - ([007b0db](https://github.com/NVIDIA/ncore/commit/007b0dbf7fb19796fc8a620c68dc7ff845e19caa)) - Janick Martinez Esturo

- - -

## [v18.6.0](https://github.com/NVIDIA/ncore/compare/c10047427a14c8bfeaee1960d1dd922b5ceaa011..v18.6.0) - 2026-03-12

### Highlights
- New data converters for physical AI and colmap datasets, and made various improvements to existing tools and documentation.
- Improve waymo converter performance by optimizing lidar processing, resulting in ~23x speedup.
- Added `ncore_vis`, a viser-based tool for visualizing NCore datasets.
- Performance improvements of the camera / lidar sensor models by skipping updates for hidden sensors and reducing redundant allocations and kernel launches.

#### ➕ Added
- (**ncore_vis**) Allow metadata to color lidar points - ([242f0bb](https://github.com/NVIDIA/ncore/commit/242f0bb25dc2f3e38d6477cb891628849a795611)) - Michael Shelley
- (**ncore_vis**) Add up direction dropdown - ([0cdbb5b](https://github.com/NVIDIA/ncore/commit/0cdbb5b0e0fb439d22ece5b8df58c470123d23a7)) - Michael Shelley
- (**ncore_vis**) Add ncore_vis, a viser-based tool for visualizing NCore datasets - ([52a2fe4](https://github.com/NVIDIA/ncore/commit/52a2fe44617922ecf82e58c16e1a9d7b8b4ec304)) - Janick Martinez Esturo
- (**viser**) cache CameraModel instances, add device selector - ([3be074a](https://github.com/NVIDIA/ncore/commit/3be074afb97672d9fc340f6500f1ea7f8d35ee86)) - Janick Martinez Esturo
- Add bazel binary for downloading clips for PAI - ([8eaae76](https://github.com/NVIDIA/ncore/commit/8eaae76765d8d07061feb4537df8c1396a3d1e82)) - Michael Shelley
- Add physical AI to ncore v4 converter - ([43bec06](https://github.com/NVIDIA/ncore/commit/43bec0610ab72bc0d4ba78ff74e1553b04d8f38b)) - Michael Shelley
- Add colmap to ncore converter - ([6a19ab4](https://github.com/NVIDIA/ncore/commit/6a19ab4206d9158a8a3961b0f880933d5a305fc0)) - Michael Shelley
#### 🪲 Fixed
- (**basedataconverter**) don't use paths in generic logic / switch to string IDs - ([3230ce9](https://github.com/NVIDIA/ncore/commit/3230ce9b44c6340a68039cd8941e0498cf03b6a2)) - Janick Martinez Esturo
- (**ci**) Also free up disk space for wheel deployment - ([3628c2f](https://github.com/NVIDIA/ncore/commit/3628c2f51db1b44d4b18087c0cd68c09e5b3b693)) - Janick Martinez Esturo
- (**ci**) Free up disc space also for export docs - ([c909253](https://github.com/NVIDIA/ncore/commit/c909253ab02cbbd348b34822c063a6b07487cfa3)) - Janick Martinez Esturo
- (**colmap**) Final cleanups (code / docs) - ([14a172e](https://github.com/NVIDIA/ncore/commit/14a172eb635d856b1ddf47cd46b38ed7b416667d)) - Janick Martinez Esturo
- (**colmap**) Prevent division by zero and crash on missing images - ([54d729f](https://github.com/NVIDIA/ncore/commit/54d729f354b2af64534fc29d50b9030961118220)) - Michael Shelley
- (**docs**) Update bazel invocation in docs - ([c100474](https://github.com/NVIDIA/ncore/commit/c10047427a14c8bfeaee1960d1dd922b5ceaa011)) - Janick Martinez Esturo
- (**ncore_vis**) Prevent race conditions when removing nodes in scene - ([a1a4ad4](https://github.com/NVIDIA/ncore/commit/a1a4ad4ece86b372ef779d150501e5588440b210)) - Michael Shelley
- (**pai**) use '--hf-token' consistently - ([46a76c2](https://github.com/NVIDIA/ncore/commit/46a76c2c93e4ee07dee8fe2e0e05bb7e4c2bc3eb)) - Janick Martinez Esturo
- (**pai**) Update readme, default dataset revision to 'main', fix copyrights - ([7cc4ffc](https://github.com/NVIDIA/ncore/commit/7cc4ffcacc6bc5255724c9df9fc405a5f0eda6df)) - Janick Martinez Esturo
- (**pai**) Require DracoPy 2.0.0 or higher - ([10bd676](https://github.com/NVIDIA/ncore/commit/10bd676027bc8bed37586df3ccb2adf5e5e0a17a)) - Janick Martinez Esturo
- (**pai**) Remove fallbacks, add platform_class to metadata, and close streaming provider - ([7f1fe35](https://github.com/NVIDIA/ncore/commit/7f1fe359d4352dfe7c9e48ceb17cce5f9b33dc8a)) - Janick Martinez Esturo
- (**pai**) Licenses and static type updates - ([7afbc85](https://github.com/NVIDIA/ncore/commit/7afbc85078071cb94448f5d62257d2cd68d40044)) - Janick Martinez Esturo
- (**pai**) Make sure to clean up temporary directories after conversion - ([10a2437](https://github.com/NVIDIA/ncore/commit/10a243775d4d06cfa27e3b435393f0af3e1e28ac)) - Janick Martinez Esturo
- (**waymo-converter**) add type annotations and enable mypy checking - ([44dd7b2](https://github.com/NVIDIA/ncore/commit/44dd7b2052333fe82ba3cd4078ef8bb6656e742f)) - Janick Martinez Esturo
- Add early exit on missing frames / minor cleanups - ([0bc9282](https://github.com/NVIDIA/ncore/commit/0bc9282d38212c6f516729b8dd604fde298ca5e9)) - Janick Martinez Esturo
#### ⚡ Performance
- (**ncore_vis**) skip updates for hidden cameras and lidars - ([251a916](https://github.com/NVIDIA/ncore/commit/251a91699b1602ded9e8ab1bb522d2c141de0615)) - Janick Martinez Esturo
- (**sensors**) reduce redundant allocations and kernel launches in sensor models - ([625929f](https://github.com/NVIDIA/ncore/commit/625929f40e7adf4a257e5f9a26f65dafa713d06d)) - Janick Martinez Esturo
- (**waymo-converter**) optimize lidar processing ~23x speedup - ([644dc06](https://github.com/NVIDIA/ncore/commit/644dc069d48b5fd3eb871011a7726cc0b2a80719)) - Zan Gojcic, Cursor
#### 🔄 Changed
- (**colmap**) Various improvements / cleanups - ([476bd1c](https://github.com/NVIDIA/ncore/commit/476bd1ccb2b9af845d42c7c1dfc8599f270176ee)) - Janick Martinez Esturo, Copilot, Copilot
- (**data_converter**) make --root-dir optional by introducing FileBasedDataConverter - ([16acb07](https://github.com/NVIDIA/ncore/commit/16acb07ce95ef7c5765b0a2ec19a0437eaa43fba)) - Janick Martinez Esturo
- (**ncore_vis**) Cleanup metadata colorization code, only find valid metadata at init - ([62f0257](https://github.com/NVIDIA/ncore/commit/62f02572a6f5b0614679448b658fec69537e11e9)) - Michael Shelley
- (**pai**) Add missing copyright, remove toml file - ([75916b9](https://github.com/NVIDIA/ncore/commit/75916b9dc4b25dbc1c3caa8fc10e5affd36e9a93)) - Michael Shelley
- (**pai**) Remove unused code in bazel file - ([a0f1039](https://github.com/NVIDIA/ncore/commit/a0f103931be8e9fad327704aa74738134e03b4a6)) - Michael Shelley
- (**pai**) Rename pai_remote, flatten structure, remove python module - ([02e2740](https://github.com/NVIDIA/ncore/commit/02e27404bf5f7a6fd59a52740bb0b9015224bba6)) - Michael Shelley
- (**pai**) Rename pai conversion commands - ([95336a7](https://github.com/NVIDIA/ncore/commit/95336a74d636803beb094301b0c1d9d4f8169e6b)) - Michael Shelley
- (**tools**) remove ncore_visualize_labels and open3d dependency - ([6fc43b3](https://github.com/NVIDIA/ncore/commit/6fc43b3e82620b728c7db98a67487fa00fb21ad3)) - Janick Martinez Esturo
- (**waymo-converter**) separate Waymo-derived and NVIDIA-original code - ([c1b4b65](https://github.com/NVIDIA/ncore/commit/c1b4b65338571526039a6539d06b505ccd822f99)) - Janick Martinez Esturo
- split PAI converter into separate local and streaming variants with distinct config factory - ([92a1729](https://github.com/NVIDIA/ncore/commit/92a1729c883495a0790ff3939c8acbf396c7c32b)) - Janick Martinez Esturo
- clean up unused code and rename input parameter and add readme - ([f5cca96](https://github.com/NVIDIA/ncore/commit/f5cca96c79763a972cf659908b9e71f837f4a7ca)) - Michael Shelley
#### 📚 Documentation
- (**contributing**) add conventional commits and linear history guidelines - ([1355d88](https://github.com/NVIDIA/ncore/commit/1355d8853d1af242d6c257dce1679048091742ca)) - Janick Martinez Esturo
- (**converters**) Format updates - ([da7f5f7](https://github.com/NVIDIA/ncore/commit/da7f5f7a8006e2f7cb344ca5ee02b3af4dc5cefd)) - Janick Martinez Esturo
- (**converters**) Updated converter docs to remove code samples; combined pai readmes into one. - ([a525e0d](https://github.com/NVIDIA/ncore/commit/a525e0d5caf60094c0687f668e9f20eb34e6c84e)) - Michael Shelley
- (**pai**) indicate current default HuggingFace dataset revision - ([d179209](https://github.com/NVIDIA/ncore/commit/d179209f661a272b3e634809b092c2536b7f595e)) - Janick Martinez Esturo
- (**pai**) Add PAI to NCore docs - ([132e538](https://github.com/NVIDIA/ncore/commit/132e5389ac5a432edda80a2c988902fb81af3cab)) - Michael Shelley
- (**readme**) overhaul README.md for OSS readiness - ([d763103](https://github.com/NVIDIA/ncore/commit/d763103dddd6c406b7f9bf95d0252196a4fe4c48)) - Janick Martinez Esturo
- (**sphinx**) migrate to nvidia_sphinx_theme for NVIDIA corporate branding - ([aa9c3bc](https://github.com/NVIDIA/ncore/commit/aa9c3bc0ed54b9ce6a3199d0ceb378157355bf48)) - Janick Martinez Esturo
- (**waymo**) Update documentation for Waymo conversion tool - ([cceee95](https://github.com/NVIDIA/ncore/commit/cceee9516eecc6562b8ab2c92ed55f0d32da6d56)) - Janick Martinez Esturo
- comprehensive documentation overhaul - ([3eda9b0](https://github.com/NVIDIA/ncore/commit/3eda9b0dacb639d8d5d5406afe71c81e0296cac5)) - Janick Martinez Esturo
- fix duplicate data_conversions label in waymo.rst and colmap.rst - ([35c02b7](https://github.com/NVIDIA/ncore/commit/35c02b7e808e4342bfdbc9ae24daee275caa9c84)) - Janick Martinez Esturo
- add SECURITY.md per SCM standard (NRE-2883) - ([cbd904d](https://github.com/NVIDIA/ncore/commit/cbd904d317f201ec01e998205c12c4e952c703db)) - Jonas Toelke, Claude Opus 4.6 (1M context)
- Minor readme updates - ([1968e10](https://github.com/NVIDIA/ncore/commit/1968e10efb721d4344baaa0c28034bc9e45ab2a4)) - Janick Martinez Esturo
- update CHANGELOG and README for v18.5.0 release - ([4a268ed](https://github.com/NVIDIA/ncore/commit/4a268ed246696d0ede32d1754d3c4e112d06650d)) - Janick Martinez Esturo
- add CHANGELOG.md following Keep a Changelog format - ([2097286](https://github.com/NVIDIA/ncore/commit/20972864817ebf83c534c3b8d3357be4be538b10)) - Janick Martinez Esturo
- fix typo - ([bad48b1](https://github.com/NVIDIA/ncore/commit/bad48b1a3a9205f7aa21fbd68c1925039b245fe4)) - Janick Martinez Esturo
- fix copyright year and add SIL Lab link in documentation - ([ecc2290](https://github.com/NVIDIA/ncore/commit/ecc229090cbe65fb10fc840aae7e5afc881c3b7f)) - Janick Martinez Esturo
- update commit signing requirements from DCO sign-off to GPG signatures - ([3fde27a](https://github.com/NVIDIA/ncore/commit/3fde27a287d43ec988bd65de533cca1f5d758837)) - Janick Martinez Esturo
- update contributions on merge commits - ([3ded866](https://github.com/NVIDIA/ncore/commit/3ded86635bca49a780df72a3d4bf49e930f116ca)) - Janick Martinez Esturo
#### ⚙️ CI
- (**bazel**) Cache external dependencies - ([f97c1d6](https://github.com/NVIDIA/ncore/commit/f97c1d6c129523b3053064eb4819786d9850f3b1)) - Janick Martinez Esturo
- (**bazel-setup**) Remove waymo modules repo cache and set bazelrc - ([e2282ba](https://github.com/NVIDIA/ncore/commit/e2282bacb34a428c903ae95f8f17aa3a77e40041)) - Janick Martinez Esturo
- (**pipy**) publish to regular PyPI - ([a1a18d6](https://github.com/NVIDIA/ncore/commit/a1a18d62021666022ed86c893a0e6adbbb431146)) - Janick Martinez Esturo
- support fork CI by falling back to NVIDIA_PACKAGES_TOKEN for GitHub Packages auth - ([75ba65a](https://github.com/NVIDIA/ncore/commit/75ba65aa4eab88863463518212487a162cdf8f62)) - Janick Martinez Esturo
- add GitHub issue and pull request templates - ([5a7c905](https://github.com/NVIDIA/ncore/commit/5a7c90559ffdbb7a8892f2ec414f71865ee1598b)) - Janick Martinez Esturo
- patch rules_python for --repo_contents_cache support - ([3cd8809](https://github.com/NVIDIA/ncore/commit/3cd8809a9e83c1b1ebb45b3920197a0a32ba8217)) - Janick Martinez Esturo
- add conventional commits check for pull requests - ([eb7bc62](https://github.com/NVIDIA/ncore/commit/eb7bc62c39eabcd11161efdc35d558460ec7ff47)) - Janick Martinez Esturo
#### 🏗️ Build
- (**bazel**) Update to bazel 8.5.1 - ([60811b3](https://github.com/NVIDIA/ncore/commit/60811b3901dbd63fdfc985ef26d342b6ec58d784)) - Janick Martinez Esturo
- (**proto**) Update to latest rules_proto - ([26972cc](https://github.com/NVIDIA/ncore/commit/26972ccba2878ab4eb7b48b3c716691d81c3a816)) - Janick Martinez Esturo
- Add cog configuration for conventional commit linting and changelog generation - ([4822d29](https://github.com/NVIDIA/ncore/commit/4822d2922b59e4e26f5b30621bcf7cfd0e5832aa)) - Janick Martinez Esturo
- remove pyside6 dependency (LGPL-licensed) - ([8c27d3d](https://github.com/NVIDIA/ncore/commit/8c27d3da9449f161eff214803cb4487d4b6aabe0)) - Janick Martinez Esturo
#### 🎨 Style
- (**waymo-converter**) modernize type annotations for Python 3.11 - ([c01da20](https://github.com/NVIDIA/ncore/commit/c01da202fd6ad8ea72d82412f13a4a9c08a901cc)) - Janick Martinez Esturo
#### 🔧 Chore
- (**waymo**) update Waymo Open Dataset to version 1.6.1 in MODULE.bazel and related files - ([fb04dfa](https://github.com/NVIDIA/ncore/commit/fb04dfa467e8ab95879099ff88574a9314281b03)) - Janick Martinez Esturo
- Clean up Python code for public release - ([21751a1](https://github.com/NVIDIA/ncore/commit/21751a1603b4e90f9fdaeaa053323dd42cf27bbb)) - Janick Martinez Esturo

- - -


## [v18.5.0](https://github.com/NVIDIA/ncore/releases/tag/v18.5.0) - 2026-02-17
#### ➕ Added
- Initial open-source release
- V4 component-based data format specification
- Data reading/writing APIs (`ncore.data.v4`)
- Sensor model APIs (`ncore.sensors`)
- Data conversion tools for Waymo Open Dataset
- Data visualization tools
- Sphinx documentation with tutorials
