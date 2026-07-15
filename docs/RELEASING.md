# Releasing a new version

The PCM addon zip is built by CI (`.gitea/workflows/build-pcm.yml`).
Every push to `main` builds it as a downloadable artifact; pushing a
`v<version>` tag additionally creates a Gitea release with the zip
attached. The release job checks that the tag matches `metadata.json`
and fails on a mismatch.

## Steps

1. **Bump the version** in `metadata.json` — the single entry in
   `versions` (plain `MAJOR.MINOR.PATCH`, no `v` prefix; the PCM schema
   rejects anything else):

   ```json
   "versions": [
       {
           "version": "1.0.2",
           "status": "stable",
           "kicad_version": "10.0",
           "runtime": "ipc"
       }
   ]
   ```

2. **Commit, tag, push** (tag = `v` + the manifest version):

   ```powershell
   git add metadata.json
   git commit -m "Release 1.0.2"
   git tag v1.0.2
   git push
   git push origin v1.0.2
   ```

3. **Verify**: the Actions run for the tag builds
   `th.co.b4l.fill-resistance_<version>.zip` and publishes it at
   <https://git.b4l.co.th/B4L/kicad-zone-resistance/releases>, together
   with `metadata-registry.json`. The zip installs directly via
   Plugin and Content Manager → *Install from File*.

## Publishing to the official KiCad registry (optional)

The attached `metadata-registry.json` already carries the release
`download_url`, `download_sha256` and sizes. Submit it as
`packages/th.co.b4l.fill-resistance/metadata.json` in a merge request
to <https://gitlab.com/kicad/addons/metadata>. The registry keeps every
published version: append the new entry to the `versions` array of the
registry copy instead of replacing the previous one (the repo's own
`metadata.json` only ever holds the current version —
`tools/build_package.py` reads `versions[0]`).

## Local build (no CI)

```powershell
python tools/build_package.py    # writes dist/<identifier>_<version>.zip
```

Pure stdlib — no venv needed. `dist/` is gitignored.

## CI prerequisites (one-time, server side)

- Actions enabled for the repo (Settings → Actions unit).
- A runner registered with the `ubuntu-latest` label; the default
  act_runner image works — the build needs only Python 3.
- Workflow actions are pinned to commit SHAs; when bumping them, update
  the SHA and the trailing version comment together.
