# Fork CI Mirror — CPU-Only Gap Documentation

This document outlines the scope of CPU-feasible CI checks mirrored in this fork's `fork-ci-mirror-fast.yml` workflow, and the permanent gaps due to lack of GPU/self-hosted runners.

## Mirrored Checks

### Linting & Code Style (`fork-ci-mirror-fast.yml`)
- **Tool**: `uv sync --locked --only-group linting` + `bash tools/autoformat.sh` (CHECK_ONLY mode)
- **Runtime**: `ubuntu-latest` (GitHub-hosted Linux runner, ~2–3 minutes)
- **Purpose**: Detects code formatting, import sorting (isort), linting (pylint/flake8), and style violations
- **Trigger**: 
  - Auto-runs on all `pull_request` events (opened, synchronize, reopened)
  - Auto-runs on `workflow_dispatch` (manual trigger)
  - Can also be triggered via PR comment with `/run-ci-mirror`

## Permanent Gaps — GPU/Distributed Training Infrastructure

The following critical CI jobs **cannot** be mirrored on GitHub-hosted runners and remain an acknowledged permanent limitation:

### 1. Container Build (`cicd-container-build`)
- **Requires**: NVIDIA self-hosted GPU runner (`nvidia-ci-aws-gpu-x8`)
- **Purpose**: Builds Docker container with CUDA/NVIDIA dependencies
- **Gap Reason**: Requires Docker registry push permissions and AWS ECR access; GPU not needed but NVIDIA self-hosted runner required for auth

### 2. Unit Tests (`cicd-unit-tests-latest`)
- **Requires**: NVIDIA self-hosted GPU runner (`nvidia-ci-aws-gpu-x8`)
- **Purpose**: Validates CUDA kernels, distributed training ops, model correctness
- **Gap Reason**: Tests depend on NVIDIA GPU hardware; no substitute on CPU

### 3. Integration Tests (`cicd-integration-tests-latest`)
- **Requires**: NVIDIA self-hosted GPU runner (`nvidia-ci-aws-gpu-x8`)
- **Purpose**: Validates distributed training across multiple GPUs/nodes, DDP, FSDP workflows
- **Gap Reason**: Tests depend on Slurm cluster with H100 GPUs; no CPU equivalent

### 4. Installation Tests (`install-test.yml`)
- **Requires**: NVIDIA self-hosted CPU runner (`linux-amd64-cpu16`) + NGC PyTorch container
- **Purpose**: Validates pip/UV installs on NGC PyTorch images with CUDA libraries present
- **Gap Reason**: Requires NVIDIA self-hosted runner and NGC private container access

### 5. Copyright Header Checks (`copyright-check.yml`)
- **Requires**: NVIDIA-NeMo/FW-CI-templates shared workflow (private org access)
- **Purpose**: Validates NVIDIA copyright headers on new files
- **Gap Reason**: Template workflow is in private NVIDIA org; fork cannot access without auth

---

## Summary

| Check | Mirrored | Reason |
|-------|----------|--------|
| **Linting** | ✅ Yes | Runs on `ubuntu-latest` with standard build tools |
| **Container Build** | ❌ No | Requires NVIDIA self-hosted runner + AWS ECR auth |
| **Unit Tests** | ❌ No | GPU hardware required |
| **Integration Tests** | ❌ No | Distributed GPU training cluster required |
| **Install Tests** | ❌ No | NVIDIA self-hosted runner + NGC container auth |
| **Copyright Checks** | ❌ No | NVIDIA private workflow dependency |

This fork CI mirror is intended **solely for pre-validation before upstreaming PRs**. It catches common lint/style issues early, but full validation still requires running tests on NVIDIA's CI infrastructure before merging upstream.
