# PUSH — how to create the remote and push

This repo was prepared locally. The push step requires **your** GitHub credentials and has to run on your machine. Two paths.

## Path A — `gh` CLI (recommended)

If you have the GitHub CLI installed and authenticated (`gh auth login`):

```bash
# From inside the SI/ directory:
cd SI

# Create the remote repo and push in one shot.
# Replace YOUR_GH_USER with your handle (or omit --public for private).
gh repo create YOUR_GH_USER/SI \
    --public \
    --source=. \
    --remote=origin \
    --description "Home-scale compounding self-improvement: AZR + Elo + Islands + In-Place TTT" \
    --push
```

That's it. The repo exists at `https://github.com/YOUR_GH_USER/SI` with the full commit history.

## Path B — manual (no gh CLI)

1. Create the empty repo on GitHub via the web UI:
   - Name: `SI`
   - Description: `Home-scale compounding self-improvement: AZR + Elo + Islands + In-Place TTT`
   - **Do not** initialize with README, .gitignore, or LICENSE — we already have them.

2. From inside the `SI/` directory:

```bash
git remote add origin git@github.com:YOUR_GH_USER/SI.git
git push -u origin main
```

## Before you push — a 90-second review checklist

```bash
cd SI
git log --oneline          # there should be one commit ("initial commit")
git status                 # should be clean
ls docs/                   # 00-overview.md through 06-risks.md should all exist
pytest tests/ -q           # Elo math tests should pass (no GPU needed)
```

If any of those fail, fix before pushing — a public repo with broken tests on the first commit is worse than a slightly delayed push.

## After pushing — immediate next actions

1. **Enable branch protection on `main`.** Settings → Branches → require PR review. Self-improvement research accumulates accidents; branch protection prevents accidental force-pushes over committed anchor-passing generations.

2. **Add a `VERSIONS.lock`** by running `scripts/bootstrap.sh` on your actual hardware, then commit it. This pins every dependency to a specific commit so Phase 1 results are reproducible later.

3. **Read `docs/00-overview.md` end-to-end** before writing any code. The whole point of the doc pass was to specify the system precisely; deviating from it silently defeats the purpose.

4. **Do not run Phase 1 on a machine you care about** until you've confirmed the sandboxfusion container is isolated. Verify with:
   ```bash
   docker inspect si-sandbox | grep -E '(NetworkMode|CapAdd|CapDrop|Privileged|SecurityOpt)'
   ```

5. **Star the upstream repos** — LeapLabTHU/Absolute-Zero-Reasoner, verl-project/verl, andborth/RoboPhD, codelion/openevolve. This project depends on their ongoing maintenance; a star is the cheapest possible contribution back.

## If you plan to make this repo private

The Apache 2.0 license we inherit requires attribution but allows closed derivative works. You can push as `--private` above and still be compliant. If you later open-source it, check that the `NOTICE` file (not yet in this repo — add one if you distribute binaries) lists the upstream attributions.
