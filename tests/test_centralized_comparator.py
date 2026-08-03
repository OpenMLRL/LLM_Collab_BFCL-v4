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
