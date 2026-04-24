<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore Changelog

All notable changes to the NCore project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
