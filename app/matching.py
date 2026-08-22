"""Many candidates against many roles.

The naive shape of this feature is a full cross product: every CV assessed
against every job description. For ten roles and twenty candidates that is
200 assessment calls, several hundred rupees and quarter of an hour per run.
It also spends most of that budget on pairs no recruiter would ever look at.

So the work is split by what actually depends on what:

    parse each brief      once per role        M model calls
    extract each CV       once per candidate   N model calls
    affinity screen       every pair           0 model calls   <- this module
    full assessment       promising pairs only K model calls

Affinity is deliberately crude and deliberately free. It is a ROUTER, not a
verdict: it decides which pairs are worth spending a real assessment on, and
its output is labelled as screening order rather than as a match score. The
LLM assessment remains the only thing allowed to call a candidate a fit.

That distinction matters, because affinity is term overlap -- the same keyword
matching this product exists to replace. It is acceptable here only because
being wrong costs an extra assessment (or a missed one the recruiter can
trigger by hand), never a wrong verdict shown to a client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.analysis import TimelineAnalysis
from app.schemas import CandidateProfile, JobBrief

# Words that carry no signal when deciding whether a CV touches a requirement.
_STOP = {
    "and", "or", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "as", "is", "are", "be", "been", "has", "have", "had", "will", "would",
    "must", "should", "can", "able", "experience", "experienced", "years", "year",
    "strong", "proven", "solid", "deep", "hands", "hands-on", "excellent", "good",
    "working", "knowledge", "understanding", "familiarity", "familiar", "ability",
    "skills", "skill", "expertise", "expert", "background", "track", "record",
    "plus", "bonus", "preferred", "required", "essential", "desirable", "least",
    "using", "used", "use", "across", "within", "including", "such", "etc",
    "role", "team", "teams", "work", "environment", "large", "scale", "high",
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

# Below this, a pair is not worth an assessment call unless asked for by hand.
DEFAULT_MIN_AFFINITY = 0.15
# A pair must touch at least this share of the role's requirements. Without
# it, the experience component alone lifts a completely unrelated role above
# the affinity floor -- an ML engineer scored 25% against a frontend role
# purely for having enough years.
DEFAULT_MIN_TERM_RATIO = 0.10
# How many roles each candidate is assessed against by default.
DEFAULT_TOP_ROLES = 3


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) > 2 and t not in _STOP]


def _phrases(text: str) -> Set[str]:
    """Unigrams plus bigrams, so "feature store" is not reduced to "store"."""
    toks = _tokens(text)
    out: Set[str] = set(toks)
    out.update("{} {}".format(a, b) for a, b in zip(toks, toks[1:]))
    return out


def candidate_corpus(profile: CandidateProfile) -> Set[str]:
    """Everything a candidate demonstrably did, as searchable phrases.

    Includes achievements and skill evidence, not just the skills list -- the
    list is the part of a CV most easily inflated, so weighting it equally
    would reward exactly the padding this product flags.
    """
    parts: List[str] = []
    for skill in profile.skills:
        parts.append(skill.name)
        if skill.evidence:
            parts.append(skill.evidence)
    for pos in profile.positions:
        parts.extend([pos.title, pos.company, pos.summary or ""])
        parts.extend(pos.achievements)
    parts.append(profile.headline or "")
    for edu in profile.education:
        parts.extend([edu.degree or "", edu.field_of_study or ""])
    return _phrases(" ".join(parts))


@dataclass
class Affinity:
    """A cheap estimate of whether a pair is worth assessing properly."""

    score: float
    covered: List[str] = field(default_factory=list)
    uncovered: List[str] = field(default_factory=list)
    experience_ratio: float = 1.0
    term_ratio: float = 0.0

    @property
    def percent(self) -> int:
        return round(self.score * 100)


def _requirement_hit(requirement: str, corpus: Set[str]) -> bool:
    """True when the CV mentions something distinctive from the requirement.

    Bigrams are checked first: a requirement asking for "feature store" should
    not be satisfied by a CV that merely says "store".
    """
    toks = _tokens(requirement)
    if not toks:
        return False

    bigrams = ["{} {}".format(a, b) for a, b in zip(toks, toks[1:])]
    if any(b in corpus for b in bigrams):
        return True

    # No phrase match, so fall back to single terms -- but require TWO of them
    # when the requirement has two or more. One generic word in common is not
    # evidence: "Feature store engineering" must not be satisfied by a CV that
    # only says "retail store rota".
    distinctive = [t for t in toks if len(t) > 3]
    if not distinctive:
        return False
    hits = sum(1 for t in set(distinctive) if t in corpus)
    return hits >= min(2, len(set(distinctive)))


def score_affinity(
    profile: CandidateProfile,
    timeline: TimelineAnalysis,
    brief: JobBrief,
) -> Affinity:
    """Screen one candidate against one role. No model call."""
    corpus = candidate_corpus(profile)

    must = [r.text for r in brief.requirements if r.kind == "must_have"]
    checked = must or [r.text for r in brief.requirements]

    covered, uncovered = [], []
    for text in checked:
        (covered if _requirement_hit(text, corpus) else uncovered).append(text)

    term_ratio = len(covered) / len(checked) if checked else 0.0

    required = brief.stated_min_years
    if required:
        exp_ratio = min(timeline.total_experience_years / required, 1.0)
    else:
        exp_ratio = 1.0

    # Terms dominate; experience is a modifier, not a gate. A brief asking for
    # an impossible number of years should not zero out a strong candidate --
    # that is a flag for the account manager, not a screening decision.
    score = 0.75 * term_ratio + 0.25 * exp_ratio

    return Affinity(
        score=round(score, 4),
        covered=covered,
        uncovered=uncovered,
        experience_ratio=round(exp_ratio, 3),
        term_ratio=round(term_ratio, 3),
    )


# --------------------------------------------------------------------------
# Choosing which pairs to spend money on
# --------------------------------------------------------------------------


@dataclass
class PairPlan:
    candidate_index: int
    requisition_index: int
    affinity: Affinity
    selected: bool
    reason: str


def plan_pairs(
    affinities: Dict[Tuple[int, int], Affinity],
    *,
    n_candidates: int,
    n_requisitions: int,
    top_roles: int = DEFAULT_TOP_ROLES,
    min_affinity: float = DEFAULT_MIN_AFFINITY,
    min_term_ratio: float = DEFAULT_MIN_TERM_RATIO,
    assess_all: bool = False,
) -> List[PairPlan]:
    """Decide which (candidate, role) pairs get a full assessment.

    Two guarantees beyond "take the top N per candidate":

    * Every role gets at least its single best candidate assessed. Otherwise a
      niche role whose candidates all rank it second would come back empty,
      which reads as "no one is suitable" rather than "we did not look".
    * Nothing below `min_affinity`, or with essentially no term overlap, is
      selected automatically -- however few pairs that leaves. Spending a call
      on a pair with nothing in common buys an assessment nobody asked for.
    """
    def worth_it(a: Affinity) -> bool:
        return a.score >= min_affinity and a.term_ratio >= min_term_ratio

    plans: List[PairPlan] = []
    chosen: Set[Tuple[int, int]] = set()

    if assess_all:
        for (ci, ri), aff in affinities.items():
            chosen.add((ci, ri))
    else:
        # Top roles per candidate.
        for ci in range(n_candidates):
            ranked = sorted(
                ((ri, affinities[(ci, ri)]) for ri in range(n_requisitions) if (ci, ri) in affinities),
                key=lambda pair: -pair[1].score,
            )
            for ri, aff in ranked[:top_roles]:
                if worth_it(aff):
                    chosen.add((ci, ri))

        # Fill EMPTY roles only. A role that already picked up candidates from
        # the pass above needs nothing more -- topping every role up regardless
        # would quietly select the whole cross product whenever there are fewer
        # candidates than roles, defeating the budget entirely.
        for ri in range(n_requisitions):
            if any(r == ri for _, r in chosen):
                continue
            ranked = sorted(
                ((ci, affinities[(ci, ri)]) for ci in range(n_candidates) if (ci, ri) in affinities),
                key=lambda pair: -pair[1].score,
            )
            if ranked and worth_it(ranked[0][1]):
                chosen.add((ranked[0][0], ri))

    for (ci, ri), aff in sorted(affinities.items(), key=lambda kv: (kv[0][0], -kv[1].score)):
        selected = (ci, ri) in chosen
        if assess_all:
            reason = "every pair requested"
        elif selected:
            reason = "screened in"
        elif aff.term_ratio < min_term_ratio:
            reason = "nothing in this CV touches the role's requirements"
        elif aff.score < min_affinity:
            reason = "too little overlap to be worth assessing"
        else:
            reason = "outranked by this candidate's stronger role matches"
        plans.append(PairPlan(candidate_index=ci, requisition_index=ri, affinity=aff,
                              selected=selected, reason=reason))
    return plans


def estimate_calls(*, n_candidates: int, n_requisitions: int, n_selected: int) -> Dict[str, int]:
    """What a run will cost, so the number can be shown before it is spent."""
    return {
        "briefs": n_requisitions,
        "extractions": n_candidates,
        "assessments": n_selected,
        "total": n_requisitions + n_candidates + n_selected,
        "naive_total": n_requisitions + n_candidates + (n_candidates * n_requisitions),
    }
