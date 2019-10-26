import pytest
from topsis import topsis


@pytest.fixture
def ex1_criteria():
    return [
        {'name': "C1", 'maximize': True},
        {'name': "C2", 'maximize': False},
        {'name': "C3", 'maximize': True},
        {'name': "C4", 'maximize': True}
    ]


@pytest.fixture
def ex1_individual_1_criteria_weights():
    return [
        0.175,
        0.425,
        0.175,
        0.225
    ]


@pytest.fixture
def ex1_individual_1_alternative_scores():
    return [
        {'name': 'A1', 'scores': [7, 21, 4, 7]},
        {'name': 'A2', 'scores': [7, 24, 6, 4]},
        {'name': 'A3', 'scores': [14, 25, 3, 8]},
        {'name': 'A4', 'scores': [14, 26, 9, 3]},
        {'name': 'A5', 'scores': [21, 35, 4, 4]}
    ]


@pytest.fixture
def ex1_individual_1_expected_positive_separations():
    return [
        0.107805,
        0.117942,
        0.096981,
        0.105379,
        0.141766
    ]


@pytest.fixture
def ex1_individual_1_expected_negative_separations():
    return [
        0.124281,
        0.090785,
        0.122181,
        0.112779,
        0.083486
    ]


@pytest.fixture
def ex1_individual_1_expected_relative_closeness():
    return [
        0.535497,
        0.434945,
        0.557491,
        0.516961,
        0.370633
    ]


@pytest.fixture
def ex1_individual_1_expected_ranks():
    return [
        {'name': 'A3', 'rank': 1},
        {'name': 'A1', 'rank': 2},
        {'name': 'A4', 'rank': 3},
        {'name': 'A2', 'rank': 4},
        {'name': 'A5', 'rank': 5}
    ]


def test_topsis_results(
    ex1_criteria,
    ex1_individual_1_criteria_weights,
    ex1_individual_1_alternative_scores,
    ex1_individual_1_expected_positive_separations,
    ex1_individual_1_expected_negative_separations,
    ex1_individual_1_expected_relative_closeness,
    ex1_individual_1_expected_ranks
):
    result = topsis(criteria=ex1_criteria,
                    weights=ex1_individual_1_criteria_weights,
                    alternatives=ex1_individual_1_alternative_scores)

    assert 'positive_separations' in result
    assert 'negative_separations' in result
    assert 'relative_closeness' in result
    assert 'ranks' in result

    positive_separations_rounded = [
        round(separation, 6) for separation in result['positive_separations']
    ]

    negative_separations_rounded = [
        round(separation, 6) for separation in result['negative_separations']
    ]

    relative_closeness_rounded = [
        round(value, 6) for value in result['relative_closeness']
    ]

    assert positive_separations_rounded == ex1_individual_1_expected_positive_separations
    assert negative_separations_rounded == ex1_individual_1_expected_negative_separations
    assert relative_closeness_rounded == ex1_individual_1_expected_relative_closeness
    assert result['ranks'] == ex1_individual_1_expected_ranks
