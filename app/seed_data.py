"""Sample roles and applicants, so a fresh start is explorable.

Everything here is invented. The point is that opening the app for the first
time shows a populated board rather than five empty states, and that the
constraint gate has something real to bite on -- a candidate priced out of one
role and comfortably inside another, a notice period that busts one limit,
a location mismatch. An empty demo hides exactly the behaviour worth seeing.

Disable with SEED_DEMO_DATA=0.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict

from app.accounts import COMPANIES, find_by_email
from app.config import settings
from app.intake import CandidatePreferences, RoleConstraints

log = logging.getLogger(__name__)

ROLES = [
    ("hiring@fintrail.example", dict(
        role_title="Staff Machine Learning Engineer",
        client_name="Fintrail Technologies", location="Bengaluru", work_mode="hybrid",
        min_years=6, max_years=12, ctc_min_lpa=55, ctc_max_lpa=85, max_notice_days=90,
        domain="FinTech",
        must_have_skills=["Real-time inference serving", "PyTorch", "Kubernetes",
                          "Feature store engineering"],
        nice_to_have_skills=["Go", "Fraud and risk modelling"],
        notes="Replacing a departing lead on the fraud platform.")),
    ("hiring@fintrail.example", dict(
        role_title="Data Platform Engineer",
        client_name="Fintrail Technologies", location="Bengaluru", work_mode="hybrid",
        min_years=4, max_years=9, ctc_min_lpa=35, ctc_max_lpa=60, max_notice_days=60,
        domain="FinTech",
        must_have_skills=["Spark", "Airflow orchestration", "Streaming ingestion"],
        nice_to_have_skills=["Delta Lake", "dbt"])),
    ("talent@meridiangcc.example", dict(
        role_title="GenAI Platform Lead",
        client_name="Meridian Global Capability Centre", location="Hyderabad", work_mode="hybrid",
        min_years=9, max_years=16, ctc_min_lpa=85, ctc_max_lpa=115, max_notice_days=60,
        domain="Insurance",
        must_have_skills=["RAG systems at enterprise scale", "LLM fine-tuning",
                          "MLOps and drift detection", "Kubernetes", "Model governance"],
        nice_to_have_skills=["Setting up an AI function inside a GCC", "Vector databases at scale"],
        notes="0-to-1 build. Will hire a team of six in the first year.")),
    ("careers@alderline.example", dict(
        role_title="Vice President, Finance",
        client_name="Alderline NBFC", location="Mumbai", work_mode="onsite",
        min_years=12, ctc_min_lpa=70, ctc_max_lpa=95, max_notice_days=90,
        domain="Non-banking finance",
        must_have_skills=["IFRS and Ind AS reporting", "Treasury and capital markets",
                          "Statutory audit management"],
        nice_to_have_skills=["Investor relations"])),
    ("people@northwind.example", dict(
        role_title="Senior Analytics Engineer",
        client_name="Northwind Analytics", location="Gurgaon", work_mode="remote",
        min_years=5, max_years=10, ctc_min_lpa=40, ctc_max_lpa=65, max_notice_days=45,
        domain="Data and analytics",
        must_have_skills=["SQL", "dbt", "Data warehouse modelling"],
        nice_to_have_skills=["Looker", "Python"])),
    ("jobs@qadira.example", dict(
        role_title="Principal Cloud Architect",
        client_name="Qadira Digital", location="Dubai", work_mode="onsite",
        min_years=10, ctc_min_lpa=95, ctc_max_lpa=140, max_notice_days=60,
        domain="Enterprise technology",
        must_have_skills=["Kubernetes", "Terraform", "Multi-cloud architecture"],
        nice_to_have_skills=["FinOps"],
        notes="Relocation to the UAE supported.")),
]

# (display name, CV file, preferences). Chosen so the free constraint gate has
# something real to do: Meera is priced out of nothing but overlaps with
# little, Rohit's expectation exceeds every band, Daniel is in the wrong city
# without relocation.
APPLICANT_SPECS = [
    ("cv_arjun_menon.txt", "Arjun Menon", CandidatePreferences(
        full_name="Arjun Menon", email="arjun.menon.ml@example.com",
        target_roles=["ML Platform Lead", "Head of AI Engineering"],
        current_location="Bengaluru", preferred_locations=["Bengaluru", "Hyderabad"],
        work_mode="hybrid", notice_period_days=90,
        current_ctc_lpa=68, expected_ctc_lpa=90, min_acceptable_ctc_lpa=80,
        years_experience=8.4)),
    ("batch/cv_priya_raghavan.txt", "Priya Raghavan", CandidatePreferences(
        full_name="Priya Raghavan", email="priya.raghavan@example.com",
        target_roles=["GenAI Platform Lead", "Principal ML Engineer"],
        current_location="Hyderabad", preferred_locations=["Hyderabad", "Bengaluru"],
        work_mode="hybrid", notice_period_days=60,
        current_ctc_lpa=78, expected_ctc_lpa=98, min_acceptable_ctc_lpa=88,
        years_experience=11.9)),
    ("batch/cv_daniel_okonkwo.txt", "Daniel Okonkwo", CandidatePreferences(
        full_name="Daniel Okonkwo", email="d.okonkwo@example.com",
        target_roles=["Staff Engineer", "Cloud Architect"],
        current_location="Dubai", preferred_locations=["Dubai"],
        open_to_relocate=False, work_mode="onsite", notice_period_days=30,
        expected_ctc_lpa=105, min_acceptable_ctc_lpa=95, years_experience=8.2)),
    ("batch/cv_meera_kulkarni.txt", "Meera Kulkarni", CandidatePreferences(
        full_name="Meera Kulkarni", email="meera.k@example.com",
        target_roles=["Data Scientist", "Analytics Engineer"],
        current_location="Pune", preferred_locations=["Pune", "Gurgaon"],
        open_to_relocate=True, work_mode="remote", notice_period_days=90,
        current_ctc_lpa=32, expected_ctc_lpa=45, min_acceptable_ctc_lpa=38,
        years_experience=7.0)),
    ("batch/cv_rohit_bansal.txt", "Rohit Bansal", CandidatePreferences(
        full_name="Rohit Bansal", email="rohit.bansal@example.com",
        target_roles=["Director of Engineering", "VP Engineering"],
        current_location="Gurgaon", preferred_locations=["Gurgaon", "Bengaluru"],
        work_mode="hybrid", notice_period_days=90,
        current_ctc_lpa=110, expected_ctc_lpa=135, min_acceptable_ctc_lpa=125,
        years_experience=14.8)),
]


def seed_board(role_library: Dict[str, RoleConstraints], applicants: Dict[str, dict],
               force: bool = False) -> dict:
    """Populate the role board and applicant pool. Idempotent unless forced."""
    if (role_library or applicants) and not force:
        return {"roles": 0, "applicants": 0, "skipped": True}

    created_roles = 0
    for email, fields in ROLES:
        company = find_by_email(email)
        if company is None:
            continue
        role_library[uuid.uuid4().hex[:10]] = RoleConstraints(company_id=company.id, **fields)
        created_roles += 1

    created_applicants = 0
    for relative, display, prefs in APPLICANT_SPECS:
        source = settings.sample_dir / relative
        if not source.exists():
            log.warning("seed: sample CV missing at %s", source)
            continue
        # Copy into the uploads directory so the pipeline reads it exactly as
        # it would a real upload.
        target = settings.upload_dir / "seed{}_{}".format(uuid.uuid4().hex[:6], Path(relative).name)
        target.write_bytes(source.read_bytes())
        applicants[uuid.uuid4().hex[:10]] = {
            "prefs": prefs,
            "path": target,
            "filename": "{} CV.txt".format(display),
            "stored": None,
        }
        created_applicants += 1

    log.info("seeded %d roles and %d applicants", created_roles, created_applicants)
    return {"roles": created_roles, "applicants": created_applicants, "skipped": False}
