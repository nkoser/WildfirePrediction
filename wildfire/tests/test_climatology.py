import datetime as dt

import numpy as np

from wildfire.paper.climatology import clim_features, day_reference
from wildfire.paper.surv_prepare import DAY_BITS


def _season_index(years):
    dates = []
    for y in years:
        a = dt.date(y, 4, 1)
        dates += [(a + dt.timedelta(days=k)).isoformat() for k in range(183)]
    return {d: i for i, d in enumerate(dates)}


def test_clim_features_counts_window_days_and_seasons_without_own_year():
    H, W = 40, 40
    t_of = _season_index([2001, 2002, 2003])
    fires = [(20, 22, "2001-06-10"), (20, 22, "2001-06-11"),   # im Fenster um 16.6. (2 Tage)
             (14, 20, "2001-06-12"),                            # Zeile 14 = Rand des Ausschnitts (20 - 6)
             (20, 20, "2001-07-30"),                            # ausserhalb des Fensters, aber in der Saison
             (5, 5, "2003-06-16"),                              # ausserhalb des Ausschnitts
             (20, 20, "2002-06-16")]                            # eigenes Jahr: zaehlt nicht
    keys = np.sort(np.array([(r * W + c) * DAY_BITS + t_of[d] for r, c, d in fires], np.int64))
    meta = np.array([[20020615, 20, 20]])
    day, season, ny = clim_features(keys, t_of, meta, H, W, years=range(2001, 2004))
    assert ny[0] == 2                                           # 2001 und 2003
    assert np.isclose(day[0], (3 / 15 + 0 / 15) / 2)           # 10., 11. und 12.6. im Jahr 2001
    assert np.isclose(season[0], 0.5)
    assert np.isclose(day_reference(keys, t_of, *meta[0], W, years=range(2001, 2004)), day[0])


def test_clim_window_is_clipped_to_the_season():
    H, W = 40, 40
    t_of = _season_index([2001, 2002])
    keys = np.sort(np.array([(20 * W + 20) * DAY_BITS + t_of["2001-04-02"]], np.int64))
    meta = np.array([[20020401, 20, 20]])                      # Zieltag 2.4.: Fenster 1.4. bis 9.4. (9 Tage)
    day, season, ny = clim_features(keys, t_of, meta, H, W, years=range(2001, 2003))
    assert ny[0] == 1 and np.isclose(day[0], 1 / 9) and season[0] == 1
    assert np.isclose(day_reference(keys, t_of, *meta[0], W, years=range(2001, 2003)), day[0])
