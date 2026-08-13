from native_parallel.centralized_comparator import BFCLCentralizedComparatorAdapter


def test_bfcl_prompt_and_parser_preserve_separate_tool_calls():
    adapter = BFCLCentralizedComparatorAdapter()
    prompt = adapter.build_prompt({}, ["first assignment", "second assignment"])
    assert "function-calling agents" in prompt
    assert "<agent_0>" in prompt
    assert "Auxiliary" not in prompt

    outputs = adapter.parse_completion(
        '<agent_0>weather(city="Boston")</agent_0>' "<agent_1>[]</agent_1>",
        {},
        2,
    )
    assert outputs == ['weather(city="Boston")', "[]"]


def test_bfcl_sequential_prompt_supports_multiple_prior_agents():
    adapter = BFCLCentralizedComparatorAdapter()
    prompt = adapter.build_sequential_prompt(
        {},
        ["first assignment", "second assignment", "third assignment"],
        2,
        ['weather(city="Boston")', 'time(city="Boston")'],
    )

    assert "Final Agent 0 calls" in prompt
    assert 'weather(city="Boston")' in prompt
    assert "Final Agent 1 calls" in prompt
    assert "<agent_2>" in prompt
    assert (
        adapter.parse_sequential_completion(
            '<agent_2>calendar(date="tomorrow")</agent_2>',
            {},
            2,
        )
        == 'calendar(date="tomorrow")'
    )
