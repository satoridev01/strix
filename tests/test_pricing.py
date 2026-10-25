from __future__ import annotations

from unittest.mock import patch

import litellm
import pytest
from agents.usage import Usage

from strix.report.pricing import resolve_litellm_model
from strix.report.usage import LLMUsageLedger


def test_resolves_common_bare_model_names() -> None:
    resolve_litellm_model.cache_clear()
    assert resolve_litellm_model("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("openai/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    # LiteLLM prices grok-4.5 identically under xai/, perplexity/xai/ and
    # openrouter/x-ai/, so the resolver may return any of them; the contract
    # is a provider-qualified name LiteLLM can price.
    grok = resolve_litellm_model("grok-4.5")
    assert grok is not None
    assert grok.endswith("/grok-4.5")
    assert grok in litellm.model_cost
    # MiniMax-M3 is sold by several LiteLLM providers at different prices, so
    # the resolver must not guess from its bare name. A provider-qualified
    # model remains deterministic.
    assert resolve_litellm_model("minimax/MiniMax-M3") == "minimax/MiniMax-M3"


def test_resolver_returns_none_for_unresolvable_model() -> None:
    resolve_litellm_model.cache_clear()
    assert resolve_litellm_model("provider/not-a-real-model") is None


def test_ledger_uses_estimate_when_routed_provider_reports_no_cost() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("litellm.completion_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.42


def test_ledger_prefers_observed_cost_over_estimate() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("litellm.completion_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")
    ledger.record_observed_cost(0.17)

    assert ledger.total_cost == 0.17


def test_hydrated_estimate_continues_accumulating_new_estimates() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.hydrate({"cost": 0.42})

    with patch("litellm.completion_cost", return_value=0.17):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.59


def _usage(tokens: int) -> Usage:
    return Usage(requests=1, input_tokens=tokens, output_tokens=0, total_tokens=tokens)


def _estimates_per_thousand_tokens():
    """Deterministic estimates: 400 tokens cost 0.4, 700 tokens cost 0.7."""
    return patch(
        "strix.report.usage._estimate_litellm_cost",
        side_effect=lambda usage, _model: usage.input_tokens / 1000,
    )


def test_ledger_pairs_observed_and_estimate_by_response_id() -> None:
    ledger = LLMUsageLedger()
    with _estimates_per_thousand_tokens():
        ledger.record(agent_id="a", usage=_usage(400), model="m", response_id="charged")
        ledger.record(agent_id="a", usage=_usage(700), model="m", response_id="unpriced")
    ledger.record_observed_cost(0.2, response_id="charged")

    # The charge replaces its own response's estimate, and the response with no
    # charge keeps its estimate.
    assert ledger.total_cost == pytest.approx(0.9)

    # Arrival order does not matter: a charge can land before the usage hook.
    ledger.record_observed_cost(0.3, response_id="early")
    with _estimates_per_thousand_tokens():
        ledger.record(agent_id="a", usage=_usage(600), model="m", response_id="early")
    assert ledger.total_cost == pytest.approx(1.2)


def test_ledger_keyed_charge_replaces_unkeyed_estimate_when_ids_cannot_pair() -> None:
    # The provider callback named the response, the SDK hook had no id for it.
    # They may be the same response, so the upstream contract applies: the
    # charge replaces the estimate instead of adding to it.
    ledger = LLMUsageLedger()
    with _estimates_per_thousand_tokens():
        ledger.record(agent_id="a", usage=_usage(500), model="m")
    ledger.record_observed_cost(0.17, response_id="gen-1")

    assert ledger.total_cost == pytest.approx(0.17)


def test_ledger_keeps_unkeyed_estimate_until_a_charge_is_left_unmatched() -> None:
    ledger = LLMUsageLedger()
    with _estimates_per_thousand_tokens():
        ledger.record(agent_id="a", usage=_usage(400), model="m", response_id="paired")
        ledger.record(agent_id="a", usage=_usage(500), model="m")
    ledger.record_observed_cost(0.2, response_id="paired")

    # Every charge is matched by id, so nothing replaces the id-less estimate.
    assert ledger.total_cost == pytest.approx(0.7)

    ledger.record_observed_cost(0.3)

    assert ledger.total_cost == pytest.approx(0.5)


def test_hydrated_cost_is_kept_once_when_a_charge_replaces_new_estimates() -> None:
    ledger = LLMUsageLedger()
    ledger.hydrate({"cost": 0.42})
    with _estimates_per_thousand_tokens():
        ledger.record(agent_id="a", usage=_usage(1000), model="m")
    ledger.record_observed_cost(0.17)

    # 0.42 persisted + 0.17 charged, not + the 1.0 estimate.
    assert ledger.total_cost == pytest.approx(0.59)

    with _estimates_per_thousand_tokens():
        ledger.record(agent_id="a", usage=_usage(300), model="m", response_id="after-resume")
    ledger.record_observed_cost(0.05, response_id="after-resume")

    assert ledger.total_cost == pytest.approx(0.64)


def test_zero_cost_disables_both_observed_and_estimated_costs() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.zero_cost = True

    with patch("litellm.completion_cost", return_value=0.42) as estimate:
        ledger.record(agent_id="a", usage=usage, model="deepseek-v4-flash")
        ledger.record_observed_cost(1.0)

    estimate.assert_not_called()
    assert ledger.total_cost == 0.0


def test_resolver_uses_provider_when_bare_entry_has_one() -> None:
    original = litellm.model_cost
    litellm.model_cost = {
        "example": {
            "litellm_provider": "example-provider",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        }
    }
    try:
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("example") == "example-provider/example"
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()


def test_resolver_does_not_guess_between_differently_priced_providers() -> None:
    original = litellm.model_cost
    litellm.model_cost = {
        "provider-a/example": {
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        },
        "provider-b/example": {
            "input_cost_per_token": 3.0,
            "output_cost_per_token": 4.0,
        },
    }
    try:
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("example") is None
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()
