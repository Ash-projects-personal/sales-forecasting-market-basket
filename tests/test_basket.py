"""
Tests for the Apriori market-basket analysis.

We don't rerun the full 10k-transaction generator (slow); instead we hand-craft
a deterministic basket where the support / confidence / lift values are
arithmetically obvious, then assert the mlxtend pipeline reproduces them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import apriori, association_rules


def _hand_built_basket() -> pd.DataFrame:
    """4 transactions where {A, B} appears in 3 of 4 → support 0.75, conf(A→B) = 1.0."""
    rows = [
        {"transaction_id": 1, "sku": "A"},
        {"transaction_id": 1, "sku": "B"},
        {"transaction_id": 1, "sku": "C"},
        {"transaction_id": 2, "sku": "A"},
        {"transaction_id": 2, "sku": "B"},
        {"transaction_id": 3, "sku": "A"},
        {"transaction_id": 3, "sku": "B"},
        {"transaction_id": 3, "sku": "D"},
        {"transaction_id": 4, "sku": "C"},
        {"transaction_id": 4, "sku": "D"},
    ]
    return pd.DataFrame(rows)


def _to_onehot_basket(df_trans: pd.DataFrame) -> pd.DataFrame:
    """Mirror the basket pivot the production module performs in run_market_basket()."""
    basket = (
        df_trans.groupby(["transaction_id", "sku"])["sku"]
        .count()
        .unstack()
        .fillna(0)
    )
    return basket.map(lambda v: 1 if v > 0 else 0).astype(bool)


def test_apriori_support_matches_hand_calculation():
    df = _hand_built_basket()
    basket = _to_onehot_basket(df)

    # A appears in 3/4; B in 3/4; {A, B} in 3/4; C in 2/4; D in 2/4.
    itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    item_lookup = {frozenset(items): sup for items, sup in zip(itemsets["itemsets"], itemsets["support"])}

    assert pytest.approx(item_lookup[frozenset({"A"})], rel=1e-6) == 0.75
    assert pytest.approx(item_lookup[frozenset({"B"})], rel=1e-6) == 0.75
    assert pytest.approx(item_lookup[frozenset({"C"})], rel=1e-6) == 0.50
    assert pytest.approx(item_lookup[frozenset({"D"})], rel=1e-6) == 0.50
    assert pytest.approx(item_lookup[frozenset({"A", "B"})], rel=1e-6) == 0.75


def test_apriori_confidence_for_known_rule():
    df = _hand_built_basket()
    basket = _to_onehot_basket(df)
    itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.0)

    # confidence(A → B) = P(A ∩ B) / P(A) = 0.75 / 0.75 = 1.0
    a_to_b = rules[
        (rules["antecedents"] == frozenset({"A"}))
        & (rules["consequents"] == frozenset({"B"}))
    ]
    assert len(a_to_b) == 1
    assert pytest.approx(float(a_to_b["confidence"].iloc[0]), rel=1e-6) == 1.0

    # lift(A → B) = conf / support(B) = 1.0 / 0.75 ≈ 1.3333
    assert pytest.approx(float(a_to_b["lift"].iloc[0]), rel=1e-4) == 1.0 / 0.75


def test_apriori_lift_above_one_for_co_occurring_pair():
    df = _hand_built_basket()
    basket = _to_onehot_basket(df)
    itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)

    # Both directions of A↔B must be present with lift > 1.0
    pair_rules = rules[
        rules.apply(
            lambda row: set(row["antecedents"]) | set(row["consequents"]) == {"A", "B"},
            axis=1,
        )
    ]
    assert len(pair_rules) >= 1
    assert (pair_rules["lift"] > 1.0).all()


def test_basket_pivot_is_boolean_matrix():
    """The one-hot pivot the module relies on must be 0/1 (truthy) only."""
    df = _hand_built_basket()
    basket = _to_onehot_basket(df)
    # Every cell is either True or False (bool dtype).
    assert (basket.dtypes == bool).all()
    # Row counts match the transaction count.
    assert basket.shape[0] == df["transaction_id"].nunique()
