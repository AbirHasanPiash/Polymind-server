"""User preferences are validated before they are stored."""

from __future__ import annotations

import pytest

from app.schemas.user import Preferences


def test_defaults_are_sensible():
    prefs = Preferences()
    assert prefs.default_model == "auto"
    assert prefs.default_effort == "medium"
    assert prefs.saved_prompts == []


def test_default_model_must_exist():
    with pytest.raises(ValueError):
        Preferences(default_model="gpt-9-ultra")


def test_legacy_default_model_is_accepted():
    assert Preferences(default_model="claude-4.5-sonnet").default_model == "claude-4.5-sonnet"


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        Preferences(theme="dark")


def test_saved_prompts_are_bounded():
    prompts = [{"id": str(i), "title": "t", "content": "c"} for i in range(51)]
    with pytest.raises(ValueError):
        Preferences(saved_prompts=prompts)
