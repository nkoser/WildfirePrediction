from wildfire.paper.importance import GROUPS, SEASON, variables
from wildfire.paper.prepare import CHANNEL_NAMES

PARTITION = [g for g in GROUPS if g not in ("G_all_modis", "G_all_but_fire")]


def test_groups_partition_the_channels():
    chs = [c for g in PARTITION for c in GROUPS[g]]
    assert sorted(chs) == list(range(len(CHANNEL_NAMES)))
    assert GROUPS["G_all_modis"] == [c for g in ("G_vegetation", "G_reflectance", "G_lst_emissivity",
                                                 "G_brightness_temp") for c in GROUPS[g]]
    assert 0 not in GROUPS["G_all_but_fire"] and len(GROUPS["G_all_but_fire"]) == 54


def test_variables_per_model():
    v = variables("sts")
    assert v["FWI"] == [CHANNEL_NAMES.index("FWI")] and v[SEASON] == []
    assert SEASON not in variables("pxst") and len(variables("pxst")) == 55 + len(GROUPS)
