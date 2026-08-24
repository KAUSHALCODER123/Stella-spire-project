# SpireDossier

Turns a raw CV and a client job description into a branded, evidence-cited
candidate dossier in about 30 seconds — the document a recruiter currently
spends 20–45 minutes assembling by hand.

Built for [Stellaspire](https://stellaspire.com)'s workflow: AI/ML engineering,
data, FinTech and finance leadership searches across Bengaluru, Hyderabad,
Gurgaon, Mumbai and Dubai.

---

## The idea in one paragraph

Most AI recruiting tools hand you a number — a match score, a risk rating — and
ask you to trust it. A recruiter with real pattern recognition will not, and
they are right not to. So this tool never asks. **Every judgement in the output
carries the verbatim line from the CV it rests on, and every number is
arithmetic you can check.** A `strong` verdict with no supporting quote is
treated as a bug and automatically demoted before the dossier is rendered.

## What it produces

- **Executive summary and fit rationale** written for a client hiring manager
- **Requirement-by-requirement match table** — every requirement in the brief
  rated strong / partial / absent / unclear, each with the CV quote behind it
- **Must-have coverage**, reported separately from nice-to-haves, because a
  blended percentage is a number nobody can act on
- **Flags for recruiter review** — employment gaps, short tenure, title
  inflation, claims the CV does not support, and job specs asking for something
  that cannot exist
- **Career timeline** with gaps marked inline
- **Skill matrix** distinguishing skills *demonstrated in described work* from
  skills merely *listed* — the single fastest tell for an AI-inflated CV
- **Screening-call questions** targeted at what the dossier could not resolve
- **Blind mode** (default): identity stripped from the header *and* from the
  generated prose, for client-side blind review

## The design decision that matters

The pipeline runs in a fixed order, and the ordering is the argument:

```
1. read the document          no interpretation
2. extract facts              model  -> evaluable against ground truth
3. compute the timeline       arithmetic
4. derive arithmetic flags    arithmetic
5. parse the client brief     model
6. assess against the brief   model  -> every claim carries a quote
7. merge flags                computed first, then judgement
```

Steps 3, 4 and 7 are deliberately **not** given to the model. Employment gaps,
total experience and average tenure are date arithmetic, so they are done in
Python, tagged `COMPUTED` in the output, and covered by ordinary unit tests. An
LLM asked to subtract dates will sometimes be wrong, and there is no way for the
reader to know which time. Arithmetic that can be derived should never be
generated.

Two consequences worth noting:

- **Overlapping roles are merged before counting.** A consulting engagement
  running alongside a day job is not extra career experience, and it must not
  manufacture a phantom gap either. Both are tested.
- **The `jd_market_impossibility` flag critiques the client, not the
  candidate.** When a brief asks for "10+ years of hands-on Generative AI", the
  dossier says so and suggests a renegotiated requirement. That is often the
  most valuable line in the document.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
playwright install chromium

cp .env.example .env              # then put your API key in .env
python -m scripts.check_setup     # verifies key, model, and a structured call
```

`check_setup` reports which models your key can actually reach and makes one
real structured call, so a bad key surfaces in ten seconds rather than
mid-demo.

### Tests

```bash
pytest -q
```

397 tests, **no API key and no network required**. They cover the deterministic
layer, the redaction guarantee, and the eval scorer — the parts of the output
most likely to be challenged, and the parts that must never be wrong.

### Measuring extraction accuracy

```bash
python -m eval.run_eval                 # all cases, default extraction model
python -m eval.run_eval --model gpt-4o  # compare models
python -m eval.run_eval --case 05_table_roles
```

`eval/cases/` holds ten finance CVs written to break the extractor in ten
specific ways — two-column interleaving, `Mar'21 – Present`, year-only ranges,
roles in a pipe table, an explicit career break, headings like "WHERE I'VE
WORKED", three promotions inside one employer, a concurrent directorship, and a
CV terse enough to be one line per role. `eval/ground_truth.json` holds the
hand-labelled answers; the runner scores name, email, phone, position count,
company, title, start date, end date, key-skill recall, and the stated CTC and
notice figures, then prints per-field accuracy and names every miss against the
case that produced it.

The scorer is itself tested (`tests/test_eval_scoring.py`, no API calls): a
profile built from the ground truth must score 100%, and each single-field
mutation — a dropped role, an invented role, an off-by-one-month start date, a
truncated title — must be caught in the right field. A scorer nobody checked
produces a number that merely looks like evidence.

## Layout

```
app/
  schemas.py            Pydantic models: extraction (facts) vs assessment (judgement)
  analysis.py           Date arithmetic. No LLM. Fully unit-tested.
  pipeline.py           Orchestration, in the order above
  extract/
    documents.py        PDF / DOCX / TXT -> text, with honest warnings
    prompts.py          The three system prompts, as reviewable constants
    llm.py              The only module that knows which vendor we call
  render/
    dossier.py          Jinja2 -> HTML -> PDF via Chromium
    redact.py           Blind-profile redaction of generated prose
  templates/
    dossier.html        The client-facing document
tests/
  fixtures.py           A complete hand-built dossier, so the renderer can be
                        iterated with zero token spend
eval/
  cases/                Ten CVs, each breaking the extractor a different way
  ground_truth.json     Hand-labelled answers, with the limitation stated in it
  run_eval.py           Per-field scoring; names every miss
scripts/
  check_setup.py        Preflight
data/samples/           Sample CV + GCC brief in Stellaspire's verticals
```

`app/extract/llm.py` is the only file that imports a model vendor. Everything
else — schemas, arithmetic, prompts, ingestion, rendering — is provider-
agnostic. Swapping providers is a change to one file, which is how this project
moved from Anthropic to OpenAI in about twenty minutes.

## Known limitations

Stated plainly, because a tool whose failure modes are documented is more
useful than one that claims not to have any.

- **Two-column CV templates** degrade text extraction. PyMuPDF's reading-order
  heuristic can interleave lines from adjacent columns. The extraction prompt is
  told to flag suspected interleaving rather than silently repair it.
- **Scanned / image-only PDFs** produce nothing. Detected and reported rather
  than allowed to yield an empty dossier. No OCR yet.
- **`title_inflation` uses a keyword seniority ladder**, which will misjudge
  unusual title conventions. An explicitly stated team size suppresses the flag.
- **The eval cases are synthetic.** They were written to exercise known
  failure modes, not sampled from real applications, so the numbers describe the
  extractor against those failure modes rather than against the true
  distribution of CVs. A real-world sample would be the honest next step.
- **Extraction accuracy is measured but not yet published here.** The harness
  above runs; the numbers are pending a model budget (the OpenRouter free tier
  caps at 50 requests a day). Until a run is published in this section, treat
  the extraction as unquantified — the harness existing is not the same as the
  measurement having been taken.

## Status

Working end to end: ingestion, extraction, arithmetic, assessment, HTML and PDF
rendering, blind mode. On top of that: the web review UI, company accounts with
a single sign-in, role posting and applications, batch mode (N candidates
against one brief), M×N matching with the free constraint gate and affinity
screen in front of it, TOON-encoded prompts, and Supabase persistence for
accounts, roles, applicants and dossiers.

Next, in order:

1. **Publish measured extraction accuracy.** The harness and its ground truth
   are written and the scorer is tested; the run itself is pending a model
   budget. Until the numbers are in this file, the accuracy claim is not made.
2. **A real-world CV sample.** The current eval cases are synthetic by
   construction, which bounds what the numbers can be said to prove.
3. **OCR for scanned PDFs**, which today are detected and refused rather than
   silently producing an empty dossier.
