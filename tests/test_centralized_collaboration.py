import importlib
import sys
from types import SimpleNamespace

import pytest

from comlrl.trainers.preference.collaboration import CentralizedCollaboration


@pytest.mark.parametrize(
    "section,trainer_name",
    [
        ("madpo", "MADPOTrainer"),
        ("marlhf", "MARLHFTrainer"),
        ("madpo_iter", "MADPOIterTrainer"),
        ("marlhf_iter", "MARLHFIterTrainer"),
    ],
)
@pytest.mark.parametrize("mode", ["centralized", "decentralized"])
def test_entrypoint_passes_bfcl_adapter(section, trainer_name, mode, monkeypatch):
    entrypoint = importlib.import_module(f"native_parallel.train.train_{section}")
    config = entrypoint.Config(str(entrypoint.DEFAULT_CONFIG))
    assert config.get(f"{section}.collaboration_mode") == "decentralized"
    components = SimpleNamespace(
        model_name="local-test-model",
        agent_names=None,
        num_agents=2,
        model_config=config.get_agent_model_config(),
        tokenizers=[object()],
        train_dataset=[],
        eval_dataset=[],
        reward_func=lambda left, right: [1.0],
        reward_processor=None,
        formatters=[lambda _: "left tools", lambda _: "right tools"],
        output_dir="unused",
        eval_logger=None,
        eval_aggregator=None,
        metrics_callback=None,
    )
    observed = {}

    def create_trainer(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(
        entrypoint, "prepare_native_components", lambda *a, **kw: components
    )
    monkeypatch.setattr(entrypoint, "build_wandb_config", lambda *a, **kw: {})
    monkeypatch.setattr(entrypoint, "save_final_agents_if_requested", lambda *a: None)
    monkeypatch.setattr(entrypoint, trainer_name, create_trainer)
    monkeypatch.setattr(
        sys, "argv", ["train", "--override", f"{section}.collaboration_mode={mode}"]
    )
    entrypoint.main()
    assert observed["num_agents"] == 2
    assert observed["args"].collaboration_mode == mode
    if mode == "centralized":
        runtime = CentralizedCollaboration(
            observed["centralized_comparator_adapter"],
            observed["formatters"],
            observed["reward_func"],
            2,
        )
        assert "function-calling agents" in runtime.build_prompt({})
        response = '<agent_0>weather(city="Boston")</agent_0><agent_1>[]</agent_1>'
        assert runtime.split(response, {}) == ['weather(city="Boston")', "[]"]
        if section.endswith("_iter"):
            assert observed["args"].comparator_generation_mode == "centralized"
