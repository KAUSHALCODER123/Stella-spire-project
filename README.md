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

41 tests, **no API key and no network required**. They cover the deterministic
layer and the redaction guarantee — the two parts of the output most likely to
be challenged, and the two parts that must never be wrong.

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
- **Extraction accuracy is not yet measured.** The eval harness is the next
  piece of work; measured per-field numbers will be published in this section.
  Until they are, treat the extraction as unquantified.

## Status

Working end to end: ingestion, extraction, arithmetic, assessment, HTML and PDF
rendering, blind mode.

Next: the evaluation harness (hand-labelled ground truth over ~20 CVs, per-field
accuracy reported here), the web review UI, and batch mode — ranking N
candidates against one brief.
