from __future__ import annotations

import pandas as pd

from ufc_winprob.cleaning.entity_resolution import canonicalize_name, merge_aliases


def test_canonicalize_handles_case_and_diacritics() -> None:
    assert canonicalize_name("JoSe AlDo") == "José Aldo"
    assert canonicalize_name("joanna jedrzejczyk") == "Joanna Jędrzejczyk"


def test_canonicalize_strips_punctuation() -> None:
    assert canonicalize_name("Jon 'Bones' Jones") == "Jon Jones"


def test_merge_aliases_preserves_canonical_ids() -> None:
    frame = pd.DataFrame({"fighter": ["Stipe Miočić", "stipe miocic"]})
    merged = merge_aliases(frame)
    ids = merged["canonical_id"].unique()
    assert len(ids) == 1
    assert ids[0] == merged.iloc[0]["canonical_id"]
