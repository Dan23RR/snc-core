# Release guide — Zenodo + GitHub (v0.4.0)

Step-by-step actions Daniel performs to ship the package + paper publicly.
Estimated total time: **45–90 minutes** of hands-on work.

---

## Repository state at hand-off

```
snc_core_v04/                              # → becomes the GitHub repo root
├── README.md                               # ← top-level README (badges + abstract + quickstart)
├── LICENSE                                 # MIT
├── CITATION.cff                            # GitHub citation metadata
├── CHANGELOG.md
├── RELEASE_GUIDE.md                        # ← this file
├── pyproject.toml                          # build config (v0.4.0)
├── .gitignore
├── .zenodo.json                            # auto-fills Zenodo metadata when GitHub→Zenodo integration triggers
├── .github/
│   └── workflows/
│       └── test.yml                        # CI: pytest on push (Python 3.9 → 3.12)
├── src/snc_core/                           # package source (15 files, 1500+ LOC)
├── tests/                                  # 34 tests, all PASS
├── examples/                               # 3 standalone examples
├── benchmarks/                             # full HumanEval reproducibility scripts
├── paper/
│   ├── snc-trust-layer-paper.pdf          # 15 pages, citable preprint
│   ├── snc-trust-layer-paper.md           # markdown source
│   └── preamble.tex
└── dist/
    ├── snc_core-0.4.0-py3-none-any.whl    # 20 KB pip-installable wheel
    └── snc_core-0.4.0.tar.gz              # source dist
```

Everything reproducible, MIT licensed, tested.

---

## Track A — Zenodo (publishes the citable preprint)

The cleanest path is the **GitHub→Zenodo integration**, which auto-archives every GitHub release as a Zenodo deposit with a DOI. The `.zenodo.json` in the repo root pre-fills the metadata. Two-step process.

### Step A1 — Connect Zenodo to GitHub (one-time, 5 minutes)

1. Go to https://zenodo.org/account/settings/github/
2. Sign in with GitHub (creates Zenodo account on first use, free)
3. Toggle the integration **ON** for the soon-to-be-created `snc-core` repository (after Track B Step B1 below)

Once enabled, every GitHub Release tagged with a version (e.g. `v0.4.0`) automatically creates a Zenodo deposit with a DOI.

### Step A2 — Manual Zenodo upload (alternative path, 10 minutes)

If the GitHub integration is not desired or available:

1. Go to https://zenodo.org/uploads/new
2. Upload `paper/snc-trust-layer-paper.pdf`
3. Fill in fields (most are pre-filled by `.zenodo.json` metadata if uploading via the API):
   - **Title**: *Behavioral Trust Clustering: A Thermodynamic Governance Layer for Production LLMs*
   - **Authors**: Culotta, Daniel — affiliation: Independent researcher, Italy
   - **Description**: copy from `.zenodo.json` `description` field
   - **Upload type**: Publication → Preprint
   - **License**: MIT
   - **Keywords**: copy from `.zenodo.json` `keywords` field
4. Click **Publish**. Receive DOI (e.g. `10.5281/zenodo.14500000`).

### Step A3 — Update README and CITATION.cff with the DOI

After Zenodo issues the DOI, replace `PLACEHOLDER` with the actual DOI in:
- `README.md` (badge + citation block)
- `CITATION.cff` (`doi` field at root level and in `preferred-citation`)
- `.zenodo.json` does not need updating (it is the source, not the consumer).

Quick replace:
```bash
DOI="10.5281/zenodo.14500000"  # replace with actual
sed -i "s|10.5281/zenodo.PLACEHOLDER|${DOI}|g" README.md CITATION.cff
```

Commit with message: `docs: add Zenodo DOI`.

---

## Track B — GitHub (publishes the package + paper repo)

### Step B1 — Create the GitHub repository (5 minutes)

1. Go to https://github.com/new
2. Repository name: `snc-core`
3. Description: *Behavioral Trust Clustering — a thermodynamic governance layer that reduces LLM hallucination by 52% on HumanEval. Drop-in wrapper for any decoder. MIT license.*
4. **Public**, **Do not** initialize with README/LICENSE/.gitignore (we already have them).
5. Click **Create repository**.

### Step B2 — Initial push from local repo (5 minutes)

From the `snc_core_v04/` directory on your machine:

```bash
cd C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\Startup\normaai\snc_core_v04

git init -b main
git add .
git commit -m "Initial public release v0.4.0

- snc-core Python package: HybridLayer + behavioral_governance + trust_thermodynamic
- 3 backends: Ollama, OpenAI-compatible, Callable (testing)
- 34 unit tests, all PASS
- Companion paper: 15 pages, full HumanEval n=164 results
- HumanEval benchmark scripts with reproducible candidate cache
- Examples and reproducibility infrastructure"

git remote add origin https://github.com/<your-username>/snc-core.git
git push -u origin main
```

Replace `<your-username>` with your GitHub handle. After push, the CI workflow runs automatically; verify the green checkmark on the GitHub Actions tab within 2–3 minutes.

### Step B3 — Tag and create Release (5 minutes)

Tagging a release is what triggers the Zenodo archive (if Track A1 is configured).

```bash
git tag -a v0.4.0 -m "v0.4.0 — first public release with HumanEval n=164 results"
git push origin v0.4.0
```

Then on GitHub:
1. Go to **Releases** → **Draft a new release**
2. Select tag `v0.4.0`
3. Release title: `v0.4.0 — Initial public release`
4. Release notes: paste the abstract from the paper or the summary from `CHANGELOG.md`
5. Attach the wheel: drag `dist/snc_core-0.4.0-py3-none-any.whl` and `paper/snc-trust-layer-paper.pdf` into the release attachments box.
6. Click **Publish release**.

Within 1–2 minutes, Zenodo (if integrated) creates the deposit and emails the DOI.

### Step B4 — Update DOI references and re-push (3 minutes)

After the Zenodo DOI is known, run the `sed` commands from Track A3 above, then:

```bash
git add README.md CITATION.cff
git commit -m "docs: add Zenodo DOI ${DOI}"
git push origin main
```

The DOI badge in the README will now resolve correctly.

---

## Track C — PyPI (optional but recommended, 10 minutes)

Publish the wheel to PyPI for `pip install snc-core` to work without the GitHub URL.

### One-time setup

1. Create account at https://pypi.org/account/register/
2. Generate an API token at https://pypi.org/manage/account/token/ (scope = entire account; restrict to project after first upload)
3. Save token in `~/.pypirc`:
   ```ini
   [pypi]
     username = __token__
     password = pypi-<your-token-here>
   ```

### Upload

```bash
cd snc_core_v04/
pip install twine
twine upload dist/snc_core-0.4.0-py3-none-any.whl dist/snc_core-0.4.0.tar.gz
```

After upload, `pip install snc-core` works for everyone.

---

## Track D — Announcements (optional, 30 minutes)

Once the repository is public and the DOI resolves, announce in the right channels.

### Show HN (Hacker News)
- URL submission: https://news.ycombinator.com/submit
- Format: title `Show HN: snc-core – behavioral trust clustering reduces LLM hallucination by 52% on HumanEval` and link to the GitHub repo
- Best time to post: weekday morning US Eastern time

### Reddit
- /r/LocalLLaMA — practical, frequent paper discussion, 200K+ subscribers
- /r/MachineLearning — academic audience
- Title: same as Show HN; body: 3-paragraph summary (problem + method + result), link to paper PDF and repo

### Twitter/X
- Thread of 5–7 tweets with the headline number, the formula, the Pareto curve screenshot, and the GitHub link
- Tag relevant accounts: @huggingface @arxiv_org

### Anthropic / OpenAI / Mistral newsletters
- Submit to AIWeekly, ImportAI, The Batch — they take community submissions

---

## Sanity checklist before pushing

| Check | How |
|-------|-----|
| Tests green locally | `pytest tests/ -v` shows 34/34 PASS |
| Wheel installs from clean venv | `pip install dist/snc_core-0.4.0-py3-none-any.whl` |
| README renders correctly | Open on GitHub after push, verify badges and code blocks |
| LICENSE in place | MIT, with year 2026 and "Daniel Culotta" |
| `.gitignore` excludes `__pycache__/`, `*.egg-info/`, `dist/` artifacts when not committed | inspect file |
| Paper PDF opens correctly | `paper/snc-trust-layer-paper.pdf`, 15 pages, no broken refs |
| `CITATION.cff` parses | https://citation-file-format.github.io/cff-initializer-javascript/ for validation |
| `.zenodo.json` is valid JSON | `python -m json.tool .zenodo.json` |

---

## Post-release follow-ups (week 1)

1. **Monitor the CI runs** on the first push and on the v0.4.0 tag. Fix any platform-specific test failure (Windows path separators, encoding) within 24 h.
2. **Respond to issues** on GitHub within 48 h. Even a "thanks, will look at this next week" reply is better than silence.
3. **Add a `pinned` issue** titled "Roadmap & known limitations" linking to the paper Section 5 and Appendix B (mode-collapse list).
4. **Track downloads**: Zenodo dashboard shows download counts; PyPI shows install counts via https://pypistats.org/packages/snc-core.
5. **First citation alert**: enable Google Scholar alerts on the paper title.

---

## Files Daniel must edit before publishing

Only one search-and-replace is needed across the repo: replace `dculotta` with the actual GitHub username if different. Affected files:
- `README.md` (badge URLs and clone instructions)
- `pyproject.toml` (`[project.urls]` section)
- `CITATION.cff` (`repository-code` and `url` fields)
- `.zenodo.json` (`related_identifiers[0].identifier`)

Single command (replace `<your-username>`):
```bash
grep -rl "dculotta" --exclude-dir=node_modules --exclude-dir=.git . | \
  xargs sed -i "s|dculotta|<your-username>|g"
```

That is it. The release pipeline is end-to-end automatable from this point.
