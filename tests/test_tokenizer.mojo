from std.testing import assert_equal, assert_true, TestSuite
from tokenizer.validate_tokenizers_match_hf_control import run_tokenizer_validation


def test_tokenizer_checkpoint_validation() raises:
    var result = run_tokenizer_validation()
    assert_true(result[0] > 0, msg="tokenizer validation did not run any checks")
    assert_equal(result[1], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
