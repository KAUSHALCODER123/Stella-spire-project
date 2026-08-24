# Demo script

Five minutes, seven clicks, no typing beyond two sign-ins.

Every step below runs **without spending a single token**, so it works on a
dead API key, an exhausted quota, or hotel wifi. That is deliberate: the
failure mode on the day is not a missing feature, it is a provider outage
while someone is watching.

Read the "say this" lines out loud roughly as written. They are short on
purpose — the screen is doing the work.

---

## Before you start

| | |
|---|---|
| **URL** | your Render URL, or `http://127.0.0.1:8000` |
| **Agency sign-in** | `admin` / `admin123` |
| **Employer sign-in** | `careers@alderline.example` / `admin123` |

Run through it once end to end an hour beforehand. Not to check it works —
to check *you* work.

Have a second browser tab already signed in as the employer. Switching
accounts live is the one place this drags.

---

## The 30 seconds before you click anything

Say this:

> Stellaspire places finance leaders. A shortlist of six people is a day of
> reading CVs, and the write-up you send the client is done from memory at
> the end of it. This does the reading and produces the document — but every
> claim in it points back at the line in the CV it came from.

Do not say "AI-powered", "leverage", or "platform". She has heard those.

---

## 1 · The board — 20 seconds

Sign in as `admin`. You land on the workspace.

Say this:

> Six live roles, five candidates in the pool. The brief box is prefilled
> with a real CFO mandate so we can go straight to the interesting part.

Do not read the roles out. Let her eyes do it.

---

## 2 · One candidate against one role — 90 seconds

Click **View sample report** (the fixture path — no model call).

You are on Meera Ramanathan against a CFO brief. Three things, in order:

**a. The score, then immediately its parts.**

> 72%, strong fit. Ten of thirteen must-haves. That number is not a vibe —
> it is coverage, strength of evidence and experience fit, and you can see
> all three next to it.

**b. The experience card.** This is the one to slow down on.

> She has thirteen years worked and fifteen years of career span, because
> she took nineteen months out. The bar was fifteen. Most systems would
> compute 13.3 and screen her out. This measures the calendar, and it says
> why on the card: a break is time away, not seniority lost.

Pause here. This is the beat that lands.

**c. The verification card.**

> Thirteen of fourteen quotes located in the source CV. The fourteenth is
> flagged unverified rather than passed on as fact.

Then click the **Evidence** tab and click any quote — it highlights in the
CV on the right.

> Click a claim, it shows you where it came from. Nothing in this document
> is unsourced.

---

## 3 · Blind mode — 30 seconds

Toggle **Blind**.

> Same assessment, name and contact details stripped — from the prose too,
> not just the header, and from the filename in the footer. For clients who
> want a first pass without demographic signal.

If she asks whether it is really clean: the source pane and the PDF footer
are redacted as well, and there is a test that fails if a name survives
anywhere.

---

## 4 · Many roles against many CVs — 60 seconds

Click **View sample match run** — it is on the workspace and on the
Shortlists page.

Say this:

> Three roles, four candidates. Twelve possible pairs.

Then point at the top row of cards, left to right:

> Six of those twelve were ruled out for free — on stated salary, notice
> period, location or seniority. No model call, no cost. *"Will not go below
> 95 LPA; this role tops out at 80."* Thirteen model calls instead of
> nineteen.

> The matrix underneath is every candidate against every role, so you can
> see the ones we chose not to spend money on and disagree with the reason.

The cost figure reads **₹0** on the free tier because it genuinely was free.
If you have switched to GPT-4o for the demo it shows the real rupee figure.

---

## 5 · What the client sees — 45 seconds

Switch to the tab signed in as `careers@alderline.example`.

> This is the client's own login. Alderline posted two roles, so Alderline
> sees two roles and the candidates matched to them — ranked. They cannot
> see Meridian's CFO shortlist, and Meridian cannot see theirs.

This is the beat that makes it a product rather than a script. Do not rush
it.

---

## 6 · The deliverable — 30 seconds

Back on a dossier, click **Download PDF**.

> This is what goes to the client. Your branding, the computed figures
> marked as computed, the model-generated parts marked as generated, and a
> footer saying which model wrote it and what it cost.

Hand her the laptop at this point if the room allows it.

---

## If something breaks

| Problem | What to do |
|---|---|
| Model provider down / quota gone | Nothing in steps 1–6 calls a model. Carry on. |
| Render cold start (~30s) | Load the URL before you start talking. |
| A dossier 404s | It was lost on restart. Click **View sample report** again. |
| Employer dashboard empty | Run step 4 first — it populates the shortlists. |
| Asked something you don't know | "I don't know, let me check" costs you nothing here. Guessing costs you everything. |

---

## Questions she will probably ask

**"How accurate is the extraction?"**

Be straight: there is a harness — ten CVs written to break it in specific
ways, hand-labelled ground truth, per-field scoring, and the scorer is
itself tested. The run is pending a model budget. Say the number is not
measured yet rather than implying it is. Then say what you *do* know: every
quote is checked against the source, so a wrong extraction shows up as an
unverified claim instead of a confident one.

**"Is this better than the ATS we already use?"**

No, and don't claim it. An ATS is a system of record — pipelines,
compliance, scheduling, client portals. This does one thing an ATS does
badly: turning a CV and a brief into a defensible written assessment. It
would sit next to the ATS, not replace it.

**"What did it cost to run?"**

Point at the cost card. On the free tier, ₹0. On GPT-4o, roughly a rupee or
two per dossier — and the constraint gate means you only pay for the pairs
worth assessing.

**"Could it handle scanned CVs?"**

Not yet. They are detected and refused rather than silently producing an
empty report. OCR is the next piece of work.

**"Who built this?"**

You did. Say so plainly, and say how long it took.

---

## What not to do

- Do not demo a live upload unless you have credit and have tested it that
  morning. A spinner that never resolves is the only unrecoverable failure.
- Do not open the code unless she asks. If she asks, open
  `app/analysis.py` — the date arithmetic with no model in it — and
  `tests/`.
- Do not oversell the eval. "The harness exists, the numbers don't yet" is a
  stronger sentence than a vague claim, because she can check the first one.
- Do not apologise for what is missing. Name it once, move on.
