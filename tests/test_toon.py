"""TOON encoding.

TOON only earns its place if it is (a) smaller than JSON on this project's
actual payloads and (b) unambiguous. Both are asserted here: a round of
structural tests, then a measured comparison on the real schemas.
"""

from __future__ import annotations

import pytest

from app.toon import encode, encode_model, encode_scalar


# --- scalars ---------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (None, "null"), (True, "true"), (False, "false"),
    (42, "42"), (-7, "-7"), (0, "0"),
    (3.5, "3.5"), (8.0, "8"), (-0.0, "0"),
    ("Ada", "Ada"), ("Hello world", "Hello world"),
])
def test_scalars(value, expected):
    assert encode_scalar(value) == expected


@pytest.mark.parametrize("value", [
    "", " padded", "padded ", "true", "false", "null", "42", "1e-6",
    "has: colon", "has,comma", 'has"quote', "has\\backslash",
    "has[bracket]", "has{brace}", "-leading", "#hash",
])
def test_ambiguous_strings_are_quoted(value):
    """Anything that could read back as something else must be quoted."""
    out = encode_scalar(value)
    assert out.startswith('"') and out.endswith('"'), (value, out)


def test_control_characters_are_escaped():
    out = encode_scalar("line1\nline2\ttab")
    assert "\\n" in out and "\\t" in out
    assert "\n" not in out and "\t" not in out


def test_quotes_and_backslashes_are_escaped():
    assert encode_scalar('say "hi"') == '"say \\"hi\\""'
    assert encode_scalar("a\\b") == '"a\\\\b"'


# --- structure -------------------------------------------------------------


def test_flat_object():
    assert encode({"a": 1, "b": "x"}) == "a: 1\nb: x"


def test_nested_object_uses_indentation():
    assert encode({"user": {"id": 1, "name": "Ada"}}) == "user:\n  id: 1\n  name: Ada"


def test_primitive_array_is_inline_with_a_length():
    assert encode({"tags": ["a", "b", "c"]}) == "tags[3]: a,b,c"


def test_empty_array_is_explicit():
    assert encode({"tags": []}) == "tags: []"


def test_uniform_object_array_becomes_a_table():
    out = encode({"rows": [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]})
    assert out == "rows[2]{a,b}:\n  1,x\n  2,y"


def test_the_table_header_names_fields_once():
    """This is the whole point: N rows, one field list."""
    rows = [{"name": "n{}".format(i), "kind": "k", "cat": "c"} for i in range(20)]
    out = encode({"items": rows})
    assert out.count("name") == 1
    assert out.count("kind") == 1


def test_a_ragged_array_falls_back_to_list_form():
    out = encode({"items": [{"a": 1}, {"a": 1, "b": 2}]})
    assert "items[2]:" in out and "- " in out
    assert "{a}" not in out


def test_an_array_of_nested_objects_falls_back_safely():
    out = encode({"items": [{"a": {"deep": 1}}, {"a": {"deep": 2}}]})
    assert "items[2]:" in out
    assert "deep" in out


def test_mixed_scalar_and_object_array():
    out = encode({"items": [1, {"a": 2}, "three"]})
    assert "items[3]:" in out


def test_values_containing_the_delimiter_are_quoted_inside_a_table():
    """Otherwise a row would gain a phantom column."""
    out = encode({"rows": [{"a": "x,y", "b": 1}]})
    row = out.splitlines()[-1]
    assert row.count(",") == 2, row          # one real separator, one inside quotes
    assert '"x,y"' in row


def test_top_level_list():
    assert encode([1, 2, 3]) == "items[3]: 1,2,3"


def test_top_level_scalar():
    assert encode("hello") == "hello"


def test_empty_object():
    assert encode({}) == ""


def test_deeply_nested_structure_does_not_crash():
    data = {"a": {"b": {"c": {"d": {"e": [1, 2]}}}}}
    assert "e[2]: 1,2" in encode(data)


# --- the payloads this exists for -----------------------------------------


def test_a_profile_encodes_its_arrays_as_tables():
    from tests.fixtures import sample_profile
    out = encode_model(sample_profile())
    assert "positions[" in out
    assert "skills[" in out and "]{" in out


def test_nulls_are_dropped_from_models_by_default():
    from tests.fixtures import sample_profile
    assert "null" not in encode_model(sample_profile())


def test_toon_is_smaller_than_json_on_the_real_schemas():
    """If it were not, there would be no reason to carry the encoder."""
    from tests.fixtures import sample_brief, sample_profile

    for model in (sample_profile(), sample_brief()):
        as_json = model.model_dump_json(indent=2, exclude_none=True)
        as_toon = encode_model(model)
        assert len(as_toon) < len(as_json) * 0.85, (
            "TOON should be meaningfully smaller: {} vs {}".format(len(as_toon), len(as_json)))


def test_every_field_value_survives_the_encoding():
    """Compact is worthless if it loses data."""
    from tests.fixtures import sample_profile

    profile = sample_profile()
    out = encode_model(profile)
    for position in profile.positions:
        assert position.company in out
        assert position.title in out
    for skill in profile.skills:
        assert skill.name in out
    assert profile.full_name in out
    assert str(profile.notice_period_days) in out


def test_the_assessment_prompt_explains_the_format():
    """A format the model has not been told about is a decoding risk."""
    from app.analysis import build_timeline
    from app.extract import llm
    from app.schemas import Assessment
    from tests.fixtures import sample_brief, sample_profile

    seen = {}

    class Resp:
        def parse(self, **kw):
            seen.update(kw)
            return type("R", (), {"status": "completed", "usage": None,
                                  "output_parsed": Assessment(executive_summary="s",
                                                              fit_rationale="r")})()

    llm._client = type("C", (), {"responses": Resp()})()
    try:
        p = sample_profile()
        llm.assess(profile=p, timeline=build_timeline(p), brief=sample_brief(), cv_text="CV")
    finally:
        llm._client = None

    body = seen["input"]
    assert "TOON" in body
    # A worked example of the table header, not just the format's name.
    assert "]{" in body and "}:" in body
    assert "comma-separated" in body
    # And the explanation must come before the data it describes.
    assert body.index("TOON") < body.index("<client_brief>")
