import numpy as np

from wildfire.paper import extra


def _fake(tmp_path, n=5):
    d = tmp_path / "derived_data" / "paper_extra"
    d.mkdir(parents=True)
    X = np.zeros((n, 61, 55, 13, 13), np.uint8)
    X[0, -1, 0, 6, 6] = 1                                     # Stichprobe 0: Feuer an D
    np.save(d / "X.npy", X)
    np.save(d / "y.npy", np.array([1, 0, 0, 0, 0], np.int8))
    np.save(d / "y_next_patch.npy", np.array([1, 0, 0, 1, 0], np.int8))
    fn = np.zeros((n, 13, 13), np.uint8)
    fn[0, 6, 6] = fn[3, 2, 2] = 1
    np.save(d / "fire_next.npy", fn)
    np.savez(d / "meta.npz", kind=np.array([0, 1, 1, 2, 2], np.int8))
    return tmp_path


def test_load_selects_by_kind_and_target(tmp_path):
    exp = _fake(tmp_path)
    r = extra.load(exp, "cache", "random")
    assert r["n"] == 3 and r["y"].tolist() == [1, 0, 0] and r["fire_map"][0, 6, 6]
    h = extra.load(exp, "next_patch", "hnm")
    assert h["n"] == 3 and h["y"].tolist() == [1, 1, 0] and h["fire_map"][1, 2, 2]
    assert extra.load(exp, "cache", "both")["n"] == 5


def test_append_keeps_cache_rows_and_marks_extra_as_train(tmp_path):
    ex = extra.load(_fake(tmp_path), "cache", "both")
    X = np.ones((4, 61, 55, 13, 13), np.uint8)
    y = np.array([1, 0, 1, 0])
    split = np.array(["train", "val", "test", "test"])
    X2, y2, s2 = extra.append(X, y, split, ex)
    assert X2.shape[0] == 9 and (X2[:4] == 1).all()
    assert s2[:4].tolist() == split.tolist() and set(s2[4:]) == {"train"}
    assert y2[4:].tolist() == ex["y"].tolist()
    assert extra.tag("none") == "" and extra.tag("hnm") == "_xhnm"
