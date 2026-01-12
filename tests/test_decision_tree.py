import pytest
import math
from algorithms.DecisionTree import DecisionTree
from algorithms.Node import Node


@pytest.fixture
def sample_data():
    return [
        (1, "CTCCGAAGTAGGATT"),
        (1, "TCAGAAGGTGAGGGC"),
        (1, "TTGGAAGGTTCGCAG"),
        (0, "ACTGCTAGCTAGCTA"),
        (0, "GGGGCCCCAAAATTT"),
        (0, "TATAATATATATATA"),
    ]


@pytest.fixture
def tree_instance(sample_data):
    return DecisionTree(sample_data, max_depth=2)


def test_count_positives(tree_instance, sample_data):
    assert tree_instance.get_positive_count(sample_data) == sum(1 for label, _ in sample_data if label == 1)


def test_majority_class(tree_instance, sample_data):
    majority_class = (
        1 
        if sum(1 for label, _ in sample_data if label == 1) >= sum(1 for label, _ in sample_data if label == 0)
        else 0
    )
    assert tree_instance.get_majority_class(sample_data) == majority_class

    mixed_data = [(1, "A"), (0, "B"), (0, "C")]
    assert tree_instance.get_majority_class(mixed_data) == 0


def test_entropy_empty(tree_instance):
    assert tree_instance.entropy([]) == 0

def test_entropy_pure(tree_instance):
    only_positive = [(1, "ACGT"), (1, "TGCA"), (1, "AAAA"), (1, "CCCC")]
    only_negative = [(0, "ACGT"), (0, "TGCA"), (0, "AAAA")]
    just_one_element = [(1, "ACGT")]

    assert tree_instance.entropy(only_positive) == 0
    assert tree_instance.entropy(only_negative) == 0
    assert tree_instance.entropy(just_one_element) == 0


def test_entropy_mixed(tree_instance):
    half_each = [(1, "A"), (0, "B"), (0, "C"), (1, "D")]
    mixed_data_1 = [(1, "ACGT"), (1, "TGCA"), (1, "GGGG"), (0, "AAAA")]
    mixed_data_2 = [(0, "ACGT"), (0, "TGCA"), (0, "GGGG"), (0, "TTTT"), (1, "AAAA")]
    mixed_data_3 = [(1, "ACGT"), (1, "TGCA"), (1, "GGGG"), (1, "CCCC"), (1, "TTTT"), (1, "AAAA"), (0, "AGTC"), (0, "CGTA"), (0, "GCAT"), (0, "TACG")]
    
    # pc_frac = positives / len(data_set)
    # (pc_frac * math.log(pc_frac, 2) + (1 - pc_frac) * math.log(1 - pc_frac, 2)) * (-1)
    assert tree_instance.entropy(half_each) == 1.0
    assert tree_instance.entropy(mixed_data_1) == pytest.approx(
        (3 / 4 * math.log(3 / 4, 2) + (1 - 3 / 4) * math.log(1 / 4, 2)) * (-1)
    )
    assert tree_instance.entropy(mixed_data_2) == pytest.approx(
        (1 / 5 * math.log(1 / 5, 2) + (4 / 5) * math.log(4 / 5, 2)) * (-1)
    )
    assert tree_instance.entropy(mixed_data_3) == pytest.approx(
        (6 / 10 * math.log(6 / 10, 2) + (4 / 10) * math.log(4 / 10, 2)) * (-1)
    )

def test_generate_attribute_sets(tree_instance):
    attribute_sets = tree_instance.generate_attribute_sets()
    print(attribute_sets)

    assert {"C"} in attribute_sets
    assert {"G"} in attribute_sets
    assert {"T"} in attribute_sets
    assert {"A", "C"} in attribute_sets
    assert {"T", "G"} in attribute_sets
    assert {"A", "T"} in attribute_sets
    assert {"C", "G"} in attribute_sets


@pytest.mark.parametrize("parent, left, right, expected_gain", [
        ([], [], [], 0),
        ( [(1, "ACGT"), (1, "ACGA"), (0, "TCGT"), (0, "TCGA")], [(1, "ACGT"), (1, "ACGA")], [(0, "TCGT"), (0, "TCGA")], 1.0 ),
        ( [(1, "ACGT"), (1, "TCGA"), (0, "GGGG"), (0, "AAAA")], [(1, "ACGT"), (0, "GGGG")], [(1, "TCGA"), (0, "AAAA")], 0.0 ),
        ( [ (1, "ACGT"), (1, "ACGA"), (1, "ACCC"), (0, "TCGT"), (0, "TCGA"), (0, "TCCC") ], [(1, "ACGT"), (1, "ACGA"), (1, "ACCC"), (0, "TCGT")], [(0, "TCGA"), (0, "TCCC")], 0.459 ),
        ( [(1, "AAAA"), (1, "AAAT"), (1, "AAAC"), (1, "AAAG"), (0, "TTTT"), (0, "TTTA")], [(1, "AAAA"), (1, "AAAT"), (1, "AAAC"), (1, "AAAG")], [(0, "TTTT"), (0, "TTTA")], 0.918 ),
        ( [(1, "ACGT"), (0, "TCGA"), (0, "GGGG"), (0, "AAAA"), (0, "CCCC")], [(1, "ACGT"), (0, "TCGA")], [(0, "GGGG"), (0, "AAAA"), (0, "CCCC")], 0.322 ),
        ( [(1, "ACGT"), (0, "TCGA")], [(1, "ACGT")], [(0, "TCGA")], 1.0 ),
        ( [(1, "AAAA"), (1, "AAAT"), (1, "AAAC"), (1, "AAAG"), (0, "TTTT"), (0, "TTTA")], [(1, "AAAA"), (1, "AAAT"), (1, "AAAC"), (1, "AAAG")], [(0, "TTTT"), (0, "TTTA")], 0.918 ),
        ( [(1, "ACGT"), (1, "ACGA"), (0, "GGGG"), (0, "AAAA")], [(1, "ACGT"), (1, "ACGA"), (0, "GGGG"), (0, "AAAA")], [], 0.0 ),
        ( [(1, "ACGT"), (0, "TCGA"), (0, "GGGG"), (0, "AAAA"), (0, "CCCC")], [(1, "ACGT"), (0, "TCGA")], [(0, "GGGG"), (0, "AAAA"), (0, "CCCC")], 0.322 ),
        ( [(1, "ACGT"), (0, "TCGA")], [(1, "ACGT")], [(0, "TCGA")], 1.0 ),
        ([(1, "ACGT"), (0, "TCGA")], [(1, "ACGT")], [(0, "TCGA")], 1.0),
],)
def test_info_gain(tree_instance, parent, left, right, expected_gain):
    gain = tree_instance.information_gain(parent, left, right)
    assert pytest.approx(gain, 0.01) == expected_gain


@pytest.mark.parametrize("data, expected_best_gain, expected_best_split", [
    ( [(1, "ACGT"), (1, "ACGT"), (0, "TTTT"), (0, "TTTT")], 1.0, (0, {'A'}, [(1, "ACGT"), (1, "ACGT")], [(0, "TTTT"), (0, "TTTT")]) ),
    ( [(1, "AAAA"), (1, "AAAA"), (1, "AAAA"), (1, "AAAA")], 0, None ),
    ( [(1, "ACGT"), (0, "ACGG"), (1, "ACTT"), (0, "ACCT")], 0.31, (2, {'C'}, [(0, "ACCT")], [(1, "ACGT"), (0, "ACGG"), (1, "ACTT")])),
    ( [(1, "GGGG"), (0, "AAAA"), (1, "GGAA"), (0, "AATT")], 1.0, (0, {'A'}, [(0, "AAAA"), (0, "AATT")], [(1, "GGGG"), (1, "GGAA")]) ),
    ( [(1, "ATCG"), (1, "GTCG"), (0, "AACG"), (0, "GACG"), (1, "ATTG"), (0, "AATG")], 1.0, (1, {'A'}, [(0, "AACG"), (0, "GACG"), (0, "AATG")], [(1, "ATCG"), (1, "GTCG"), (1, "ATTG")]) ),
    ( [(1, "AA"), (1, "AT"), (0, "GA"), (0, "GT"), (1, "CA"), (0, "TA")], 1.0, (0, {'A', 'C'}, [(1, "AA"), (1, "AT"), (1, "CA")], [(0, "GA"), (0, "GT"), (0, "TA")]) ),
    ( [(0, "AAAA"), (0, "CCCC"), (0, "GGGG"), (0, "TTTT")], 0, None ),
    ( [(1, "ACGT"), (0, "TGCA"), (1, "ACGA"), (0, "TGCT"), (1, "CCGT"), (0, "TACA")], 1.0, (0, {'T'}, [(0, "TGCA"), (0, "TGCT"), (0, "TACA")], [(1, "ACGT"), (1, "ACGA"), (1, "CCGT")]))
])
def test_choose_best_split(tree_instance, data, expected_best_gain, expected_best_split):
    best_gain, best_split = tree_instance.choose_best_split(data)

    assert pytest.approx(best_gain, 0.01) == expected_best_gain
    assert best_split == expected_best_split


@pytest.mark.parametrize("data, attr_position, group, expected_left, expected_right", [
    ([(1, "ACGT"), (0, "TGCA"), (1, "ACGA"), (0, "TGCT"), (1, "CCGT"), (0, "TACA")], 0, {"A"}, [(1, "ACGT"), (1, "ACGA")], [(0, "TGCA"), (0, "TGCT"), (1, "CCGT"), (0, "TACA")]),
    ([(1, "ACGT"), (0, "TGCA"), (1, "ACGA"), (0, "TGCT"), (1, "CCGT"), (0, "TACA")], 0, {"A", "C"}, [(1, "ACGT"), (1, "ACGA"), (1, "CCGT")], [(0, "TGCA"), (0, "TGCT"), (0, "TACA")]),
    ([], 0, {}, [], []),
])
def test_split_data(tree_instance, data, attr_position, group, expected_left, expected_right):
    left, right = tree_instance.split_data(data, attr_position, group)
    assert left == expected_left
    assert right == expected_right