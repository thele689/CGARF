from evaluation.run import build_parser, includes_stage, normalize_stage


def test_normalize_stage_aliases():
    assert normalize_stage("cg_mad") == "trace_analysis"
    assert normalize_stage("srcd") == "search"


def test_stage_inclusion_order():
    assert includes_stage("search", "trace_analysis")
    assert includes_stage("search", "reflection")
    assert not includes_stage("trace_analysis", "search")


def test_parser_accepts_public_example_arguments():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--final_stage",
            "trace_analysis",
            "--instance_ids",
            "astropy__astropy-12907",
            "astropy__astropy-6938",
        ]
    )

    assert args.final_stage == "trace_analysis"
    assert args.instance_ids == ["astropy__astropy-12907", "astropy__astropy-6938"]
