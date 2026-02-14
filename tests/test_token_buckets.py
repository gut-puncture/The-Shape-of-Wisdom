import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from sow.token_buckets.option_buckets import build_buckets_from_tokenizer, validate_bucket_obj  # noqa: E402


class DummyTokenizer:
    def __init__(self, encode_map, decode_map):
        self._encode_map = encode_map
        self._decode_map = decode_map

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return list(self._encode_map.get(text, []))

    def decode(self, ids, **kwargs):
        _ = kwargs
        return "".join(self._decode_map[i] for i in ids)


class TestTokenBuckets(unittest.TestCase):
    def test_build_buckets_from_tokenizer(self) -> None:
        # Map some variants to ids. Include a multi-token encoding for "(A)".
        encode_map = {
            "A": [1],
            " A": [2],
            "\nA": [3],
            "(A)": [10, 1, 11],
            "A.": [4],
            "A:": [5],
            "B": [6],
            " B": [7],
            "\nB": [8],
            "(B)": [10, 6, 11],
            "B.": [9],
            "B:": [12],
            "C": [13],
            " D": [14],
        }
        decode_map = {
            1: "A",
            2: " A",
            3: "\nA",
            4: "A.",
            5: "A:",
            6: "B",
            7: " B",
            8: "\nB",
            9: "B.",
            10: "(",
            11: ")",
            12: "B:",
            13: "C",
            14: " D",
        }
        tok = DummyTokenizer(encode_map=encode_map, decode_map=decode_map)

        obj = build_buckets_from_tokenizer(tok)

        # Basic structure + validation.
        self.assertIn("buckets", obj)
        validate_bucket_obj({"buckets": obj["buckets"], "overlaps": obj["overlaps"]})

        self.assertIn(1, obj["buckets"]["A"])
        self.assertIn(2, obj["buckets"]["A"])
        self.assertIn(3, obj["buckets"]["A"])
        self.assertIn(4, obj["buckets"]["A"])
        self.assertIn(5, obj["buckets"]["A"])

        self.assertIn(6, obj["buckets"]["B"])
        self.assertIn(7, obj["buckets"]["B"])
        self.assertIn(8, obj["buckets"]["B"])
        self.assertIn(9, obj["buckets"]["B"])
        self.assertIn(12, obj["buckets"]["B"])

        # C bucket should at least contain 13; D should at least contain 14.
        self.assertIn(13, obj["buckets"]["C"])
        self.assertIn(14, obj["buckets"]["D"])

