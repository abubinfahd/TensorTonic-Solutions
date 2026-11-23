from collections import Counter


def gini_impurity(y_left: list, y_right: list) -> float:
  """
    Returns the impurity as a float.
    """
  n_left = len(y_left)
  n_right = len(y_right)
  n_total = n_left + n_right

  if n_total == 0:
    return 0.0

  def node_gini(y: list, n: int) -> float:
    if n == 0:
      return 0.0
    counts = Counter(y).values()
    return 1.0 - sum((c / n) ** 2 for c in counts)

  return (n_left / n_total) * node_gini(y_left, n_left) + (
      n_right / n_total
  ) * node_gini(y_right, n_right)