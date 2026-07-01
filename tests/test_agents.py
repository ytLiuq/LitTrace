from littrace.agents import agent_runtime_statuses, crew_role_specs


def test_agent_roles_include_planner_writer_and_eval():
    role_names = {role.name for role in crew_role_specs()}
    status_names = {status.name for status in agent_runtime_statuses()}

    expected = {"Research Planner", "Research Writer", "Eval Auditor", "FullText Resolver"}
    assert expected <= role_names
    assert expected <= status_names
