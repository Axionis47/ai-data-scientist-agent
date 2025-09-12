from pathlib import Path
from app.eda.eda import detect_delimiter

def test_detect_delimiter_semicolon():
    sample = "a;b;c\n1;2;3\n4;5;6\n"
    assert detect_delimiter(sample) == ";"

def test_detect_delimiter_pipe():
    sample = "a|b|c\n1|2|3\n4|5|6\n"
    assert detect_delimiter(sample) == "|"

