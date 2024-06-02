import json
import logging
import os
import tempfile

import pytest
from translator import perform_translator
from vm import computer


@pytest.mark.golden_test("golden/*.yml")
def test_bar(golden, caplog):
    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "source.asm")
        target_file = os.path.join(tmpdir, "source.json")
        input_token = [ord(i) for i in golden["stdin"].rstrip("\n")]

        with open(source_file, "w", encoding="utf-8") as file:
            file.write(golden["source_code"])
            perform_translator(golden["source_code"], target_file)
            print("=" * 5)
            code_dict = json.load(open(target_file, encoding="utf-8"))
            computer(code_dict, input_token)

        with open(target_file, encoding="utf-8") as file:
            human_readable = file.read()

        assert human_readable.rstrip("\n") == golden.out["out_code_readable"].rstrip("\n")
        open("file_log.txt", "w").write(caplog.text)
        assert caplog.text.rstrip("\n") == golden.out["out_log"].rstrip("\n")
