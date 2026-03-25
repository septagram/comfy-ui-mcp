---
name: release
description: Bump version, build release + installer, tag and push
argument-hint: "[major|minor|patch|none|skip]"
allowed-tools: Read, Edit, Bash, Glob
---

# Release Pipeline

## Step 1: Determine version bump

Argument: `$ARGUMENTS`

- `major` → bump major (X.0.0 → X+1.0.0)
- `minor` → bump minor (X.Y.0 → X.Y+1.0)
- `patch` → bump patch (X.Y.Z → X.Y.Z+1)
- `none` or `skip` → use current version
- Empty / no argument → default to **patch** bump

Read `Cargo.toml` to get the current `version = "X.Y.Z"`. Parse the three semver
components. Compute the new version. Edit `Cargo.toml` with the new version.
Run `cargo check` to update `Cargo.lock`.

## Step 2: Build release + installer

```bash
cargo wix -p comfy-ui-mcp
```

This builds the release binary and produces an MSI in `target/wix/`.
If it fails, stop and report the error. Do NOT continue.

## Step 3: Commit, tag, push

1. Stage: `git add Cargo.toml Cargo.lock`
2. Commit: `Release X.Y.Z`
3. Tag: `git tag -a X.Y.Z -m "X.Y.Z"` (no `v` prefix)
4. Push: `git push && git push --tags`

## Step 4: Report

Print:
- Final version number
- Path to the built MSI (`target/wix/comfy-ui-mcp-X.Y.Z-x86_64.msi`)
- Size of the MSI
