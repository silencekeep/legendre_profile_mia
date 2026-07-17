from legendre_mia.utils.validation import verify_source_manifest


def test_vendored_source_hashes() -> None:
    assert verify_source_manifest()["ok"]
