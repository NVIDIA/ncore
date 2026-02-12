# GitHub Migration Guide

This document describes the migration from GitLab CI to GitHub Actions and the changes required to work with the new infrastructure.

## Overview

The ncore repository has been migrated from GitLab to GitHub. This guide explains the changes and how to work with the new infrastructure.

| Aspect | GitLab (Old) | GitHub (New) |
|--------|--------------|--------------|
| Repository | gitlab-master.nvidia.com/nrs/ncore | github.com/NVIDIA/ncore |
| CI/CD | GitLab CI (`.gitlab-ci.yml`) | GitHub Actions (`.github/workflows/`) |
| Docker Registry | gitlab-master.nvidia.com:5005/nrs/ncore | ghcr.io/nvidia/ncore |
| Python Packages | GitLab Package Registry | Test PyPI (test.pypi.org) |
| Test Data | GitLab Generic Package Registry | GitHub Packages (Maven) |
| Documentation | GitLab Pages | GitHub Pages |
| GPU Testing | GPU runners available | CPU-only (GPU tests skipped) |

## CI/CD Workflow

All CI/CD is handled by a single unified workflow: `.github/workflows/ci.yml`

| GitLab CI Job | GitHub Actions Job | Branch | Purpose |
|---------------|-------------------|--------|---------|
| `build+test` | `build-and-test` | All | Format check, build all targets, run tests |
| `pages` | `build-docs` + `deploy-docs` | main only | Documentation deployment to GitHub Pages |
| `wheel` | `publish-testpypi` | main only | Python package publishing to Test PyPI |

The workflow uses the shared composite action `.github/actions/setup-bazel/action.yml` for common setup steps (disk cleanup, GitHub Packages auth, pandoc installation).

**Job Dependencies:**
- `build-and-test` runs on all branches and PRs
- `build-docs`, `deploy-docs`, and `publish-testpypi` only run on pushes to `main` after `build-and-test` succeeds

## Developer Setup

### 1. GitHub Personal Access Token (PAT)

You need a PAT with `read:packages` scope for downloading test data and Python packages.

**Create PAT:**
1. Go to GitHub Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. Generate new token with scopes: `read:packages`, `write:packages` (if publishing)
3. Save token securely

### 2. Configure Test Data Authentication

GitHub Packages requires authentication for downloads. Configure `~/.netrc`:

```
machine maven.pkg.github.com
  login YOUR_GITHUB_USERNAME
  password YOUR_GITHUB_PAT
```

**For CI workflows:** Automatically configured using `GITHUB_TOKEN`.

### 3. Install ncore from Test PyPI

During development/testing, ncore is published to Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ncore
```

**Note:** The `--extra-index-url https://pypi.org/simple` flag is required because Test PyPI doesn't host dependencies like numpy, torch, etc. They will be downloaded from the main PyPI.

**View package:** https://test.pypi.org/project/ncore/

### 4. Docker Image Access

```bash
# Login to GitHub Container Registry
echo $GITHUB_PAT | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Pull the dev image
docker pull ghcr.io/nvidia/ncore:dev
```

## Local Development

No changes needed for basic Bazel operations:

```bash
bazel build ...
bazel test ...
bazel run //:format
```

**Note:** Test data downloads now require GitHub authentication (see step 2 above).

## GPU Testing

### What Changed

GitHub Actions runners do not have GPUs. Tests that use `device="cuda"` are automatically skipped when the environment variable `CI_NO_GPU=1` is set.

Local developers are expected to have a GPU available.

### Affected Tests

- `ncore/impl/sensors/camera_test.py` - 8 parameterized test classes
- `ncore/impl/sensors/lidar_test.py` - 1 parameterized test class

### Implementation

Tests use `_get_test_devices()` function that returns:
- `("cpu", "cuda")` when `CI_NO_GPU` is not set
- `("cpu",)` when `CI_NO_GPU=1`

### Local GPU Testing

GPU tests always run locally (assuming CUDA is available):

```bash
# Run all tests including GPU
bazel test //ncore/impl/sensors:all

# Force CPU-only mode (for testing CI behavior locally)
CI_NO_GPU=1 bazel test //ncore/impl/sensors:all
```

## Test Data Packages

The project depends on **4 external tarball packages**:

| Package | Version | Location |
|---------|---------|----------|
| test-data-v3-shards | 0.7 | `bazel/repo/repos.bzl` |
| test-data-v4 | 0.1 | `bazel/repo/repos.bzl` |
| test-data-v4 | 0.2 | `MODULE.bazel` |
| ncore-docs-data | 0.2 | `MODULE.bazel` |

All packages migrated from GitLab Generic Package Registry to GitHub Packages (Maven-style).

**New URL format:**
```
https://maven.pkg.github.com/NVIDIA/ncore/com/nvidia/ncore/<package>/<version>/<package>-<version>.tar.gz
```

### Uploading New Test Data Packages

To upload new or updated test data packages to GitHub Packages, use the following commands.

**Step 1: Download from GitLab (if migrating existing packages)**

```fish
# Create a temporary directory
mkdir -p ~/ncore-test-data-migration
cd ~/ncore-test-data-migration

# Download packages using wget
wget -O test-data-v3-shards-0.7.tar.gz \
  "https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v3-shards/0.7/test-data-v3-shards.tar.gz"

wget -O test-data-v4-0.1.tar.gz \
  "https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v4/0.1/test-data-v4.tar.gz"

wget -O test-data-v4-0.2.tar.gz \
  "https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v4/0.2/test-data-v4.tar.gz"

wget -O ncore-docs-data-0.2.tar.gz \
  "https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/ncore-docs-data/0.2/ncore-docs-data.tar.gz"

# Verify downloads
ls -lh *.tar.gz
sha256sum *.tar.gz
```

**Step 2: Upload to GitHub Packages**

```fish
# Set your GitHub PAT (needs write:packages scope)
set -x GITHUB_TOKEN "your_github_pat_here"

# Upload test-data-v3-shards v0.7
curl -X PUT \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Content-Type: application/gzip" \
  --data-binary "@test-data-v3-shards-0.7.tar.gz" \
  "https://maven.pkg.github.com/NVIDIA/ncore/com/nvidia/ncore/test-data-v3-shards/0.7/test-data-v3-shards-0.7.tar.gz"

# Upload test-data-v4 v0.1
curl -X PUT \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Content-Type: application/gzip" \
  --data-binary "@test-data-v4-0.1.tar.gz" \
  "https://maven.pkg.github.com/NVIDIA/ncore/com/nvidia/ncore/test-data-v4/0.1/test-data-v4-0.1.tar.gz"

# Upload test-data-v4 v0.2
curl -X PUT \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Content-Type: application/gzip" \
  --data-binary "@test-data-v4-0.2.tar.gz" \
  "https://maven.pkg.github.com/NVIDIA/ncore/com/nvidia/ncore/test-data-v4/0.2/test-data-v4-0.2.tar.gz"

# Upload ncore-docs-data v0.2
curl -X PUT \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Content-Type: application/gzip" \
  --data-binary "@ncore-docs-data-0.2.tar.gz" \
  "https://maven.pkg.github.com/NVIDIA/ncore/com/nvidia/ncore/ncore-docs-data/0.2/ncore-docs-data-0.2.tar.gz"

echo "Upload complete!"
```

**Step 3: Verify uploads**

```fish
wget --header="Authorization: Bearer $GITHUB_TOKEN" \
  -O /dev/null \
  "https://maven.pkg.github.com/NVIDIA/ncore/com/nvidia/ncore/test-data-v4/0.2/test-data-v4-0.2.tar.gz" \
  && echo "Download successful!"
```

**Note:** The GitHub repository must exist before uploading packages.

## Parallel Operation Mode

During the transition period, both CI systems are active:

- **GitLab CI:** `.gitlab-ci.yml` (original, kept for reference)
- **GitHub Actions:** `.github/workflows/` (new)

Both configurations include comments documenting the equivalent in the other system.

### Divergence Documentation

Key differences are documented inline:
- GitHub workflow files reference GitLab CI line numbers
- GitLab CI file includes migration notice header
- Bazel files include comments showing old vs new URLs

## Troubleshooting

### Test Data Download Fails

**Error:** `Failed to download https://maven.pkg.github.com/...`

**Solution:** Configure `~/.netrc` with GitHub credentials (see Developer Setup #2)

### Test PyPI Upload Fails

**Error:** `403 Forbidden` when uploading to Test PyPI

**Solution:** Ensure the `TEST_PYPI_API_TOKEN` secret is configured in GitHub repository settings:
1. Create API token at https://test.pypi.org/manage/account/token/
2. Add to GitHub: Settings -> Secrets and variables -> Actions -> New repository secret
3. Name: `TEST_PYPI_API_TOKEN`, Value: (your token starting with `pypi-`)

**Error:** `400 File already exists`

**Solution:** The version is already published to Test PyPI. Either:
- Update version in `ncore/BUILD.bazel`
- The workflow uses `--skip-existing` to handle this gracefully

### Docker Pull Fails

**Error:** `unauthorized: access denied`

**Solution:** Login to GitHub Container Registry:
```bash
echo $GITHUB_PAT | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

### GPU Tests Failing Locally

**Check:** Is CUDA available?
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Force CPU mode:**
```bash
export CI_NO_GPU=1
bazel test ...
```

### Bazel Download Authentication

If Bazel fails to download test data, ensure `~/.netrc` is configured:
```bash
cat ~/.netrc
# Should show:
# machine maven.pkg.github.com
#   login YOUR_USERNAME
#   password YOUR_PAT
```

## Migration Timeline

1. **Phase 1:** GitHub repository created
2. **Phase 2:** GitHub Actions workflows created
3. **Phase 3:** Parallel operation (GitLab + GitHub) - **CURRENT**
4. **Phase 4:** Team validation and testing
5. **Phase 5:** GitLab CI archived
6. **Phase 6:** Update all external references

## Files Changed in Migration

### New Files
- `.github/actions/setup-bazel/action.yml` - Shared composite action for Bazel environment setup
- `.github/workflows/ci.yml` - Unified CI pipeline (build, test, docs deployment, PyPI publishing)
- `GITHUB_MIGRATION.md` - This documentation

### Modified Files
- `MODULE.bazel` - Updated test data URLs
- `bazel/repo/repos.bzl` - Updated test data URLs
- `ncore/BUILD.bazel` - Updated homepage URL
- `.gitlab-ci.yml` - Added migration notice header
- `ncore/impl/sensors/camera_test.py` - GPU detection for test skipping
- `ncore/impl/sensors/lidar_test.py` - GPU detection for test skipping

## Questions & Support

- **GitHub Issues:** https://github.com/NVIDIA/ncore/issues
- **Repository:** https://github.com/NVIDIA/ncore
- **Documentation:** https://nvidia.github.io/ncore

---

**Last Updated:** 2026-02-12
**Migration Status:** In Progress (Parallel Operation)
