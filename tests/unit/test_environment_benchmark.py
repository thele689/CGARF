from src.environment.benchmark import BenchmarkRunResult, CGARFDockerBenchmarkEnv


def test_run_unit_tests_delegates_to_run(tmp_path, monkeypatch):
    env = CGARFDockerBenchmarkEnv(workspace_root=tmp_path, image_name="cgarf:test")
    captured = {}

    def fake_run(command, timeout=0, output_log=False, check=True):
        captured["command"] = command
        captured["timeout"] = timeout
        captured["output_log"] = output_log
        captured["check"] = check
        return BenchmarkRunResult(command=command, output="", exit_code=0)

    monkeypatch.setattr(env, "run", fake_run)
    result = env.run_unit_tests(timeout=321, output_log=True)

    assert captured["command"] == "pytest tests/unit -q"
    assert captured["timeout"] == 321
    assert captured["output_log"] is True
    assert captured["check"] is True
    assert result.ok is True


def test_smoke_install_delegates_to_run(tmp_path, monkeypatch):
    env = CGARFDockerBenchmarkEnv(workspace_root=tmp_path, image_name="cgarf:test")
    captured = {}

    def fake_run(command, timeout=0, output_log=False, check=True):
        captured["command"] = command
        captured["timeout"] = timeout
        captured["output_log"] = output_log
        captured["check"] = check
        return BenchmarkRunResult(command=command, output="CGARF import OK\n", exit_code=0)

    monkeypatch.setattr(env, "run", fake_run)
    result = env.smoke_install(timeout=111, output_log=False)

    assert "CGARF import OK" in result.output
    assert "src.environment.benchmark" in captured["command"]
    assert captured["timeout"] == 111
