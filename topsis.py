import numpy as np
import math


def topsis(criteria=[], alternatives=[], weights=[]):
    assert type(alternatives) is list, "alternatives should be a list"
    assert type(criteria) is list, "criteria should be a list"
    assert type(weights) is list, "weights should be a list"

    num_criteria = len(criteria)
    num_weights = len(weights)

    alternatives_matrix = [
        alternative['scores'] for alternative in alternatives
    ]

    assert num_criteria == num_weights, "there should be a weight for each criterion"
    assert (len(alternative_scores) == num_criteria
            for alternative_scores in alternatives_matrix), "each alternative must have a score for each criterion"

    # Step 1: Construct the decision matrix and determine the weight of criteria.

    # Let X be a decision matrix, where each element x(i, j) is a real number
    X = np.array(alternatives_matrix)
    assert (x is int or float
            for x in np.nditer(X)), "scores must be real numbers"

    # Let W be a weight vector, where each element w(i, j) is a real number,
    # and the sum of all w(i, j) is equal to 1.
    W = np.array(weights)
    assert (w is int or float
            for w in np.nditer(W)), "weights must be real numbers"
    assert math.isclose(W.sum(), 1), "the sum of all weights must equal 1"

    # Step 2: Calculate the normalized decision matrix.
    # Use the Frobenius formula

    N = X / np.linalg.norm(X, axis=0)

    # Step 3: Calculate the weighted normalized decision matrix.

    V = W * N

    # Step 4: Determine the positive ideal and negative ideal solutions.

    VMin = V.min(axis=0)
    VMax = V.max(axis=0)

    APos = np.where([criteria['maximize'] for criteria in criteria],
                    VMax,
                    VMin)

    AMin = np.where([criteria['maximize'] for criteria in criteria],
                    VMin,
                    VMax)

    # Step 5: Calculate the separation measures from the positive ideal solution
    # and the negative ideal solution.

    DPos = np.sqrt(np.sum((V - APos)**2, axis=1))
    DMin = np.sqrt(np.sum((V - AMin)**2, axis=1))

    # Step 6: Calculate relative closeness to the positive ideal solution.

    R = DMin / (DMin + DPos)

    # Step 7: Rank the preference order / select the alternative closest to 1.

    Rr = list(
        {'name': alternatives[index]['name'], 'closeness': closeness}
        for (index, closeness) in enumerate(R.tolist())
    )

    SRr = list(
        {'name': alternative['name'], 'rank': index + 1}
        for (index, alternative) in
        enumerate(sorted(Rr, key=lambda v: v['closeness'], reverse=True))
    )

    return {
        'positive_separations': DPos.tolist(),
        'negative_separations': DMin.tolist(),
        'relative_closeness': R.tolist(),
        'ranks': SRr
    }
