"""
Module that provides classes for tree creation and handling.

Trees are powerful structures to sort a huge amount of data and to speed up
performing query requests on them significantly.
"""

from collections import Iterable

import pandas as pd
import numba
import numpy as np

from sklearn.neighbors import BallTree, KDTree

__all__ = [
    "IntervalTree",
    "RangeTree",
]


class IntervalTreeNode:
    """Helper class for IntervalTree.

    """
    def __init__(self, center_point, center, left, right):
        self.center_point = center_point
        self.center = np.asarray(center)
        self.left = left
        self.right = right


class IntervalTree:
    """Tree to implement fast 1-dimensional interval searches.

    Based on the description in Wikipedia
    (https://en.wikipedia.org/wiki/Interval_tree#Centered_interval_tree)
    and the GitHub repository by tylerkahn
    (https://github.com/tylerkahn/intervaltree-python).

    Examples:
        Check 1000 intervals on 1000 other intervals:

    .. code-block:: python

        import numpy as np
        from typhon.trees import IntervalTree

        intervals = np.asarray([np.arange(1000)-0.5, np.arange(1000)+0.5]).T
        tree = IntervalTree(intervals)
        query_intervals = [[i-1, i+1] for i in range(1000)]
        results = tree.query(query_intervals)

    """
    def __init__(self, intervals):
        """Creates an IntervalTree object.

        Args:
            intervals: A list or numpy.array containing the intervals (list of
                two numbers).
        """
        if not isinstance(intervals, np.ndarray):
            intervals = np.asarray(intervals)

        # Check the intervals whether they are valid:
        self.left = np.min(intervals)
        self.right = np.max(intervals)

        # We want to return the indices of the intervals instead of their
        # actual bounds. But the original indices will be lost due resorting.
        # Hence, we add the original indices to the intervals themselves.
        indices = np.arange(intervals.shape[0]).reshape(intervals.shape[0], 1)
        indexed_intervals = np.hstack([intervals, indices])
        self.root = self._build_tree(np.sort(indexed_intervals, axis=0))

    def __contains__(self, item):
        if isinstance(item, (tuple, list)):
            return bool(self._query(item, self.root, check_extreme=True))
        else:
            return bool(self._query_point(item, self.root, check_extreme=True))

    # @numba.jit
    def _build_tree(self, intervals):
        if not intervals.any():
            return None

        center_point = self._get_center(intervals)

        # Sort the intervals into bins
        center = intervals[(intervals[:, 0] <= center_point)
                           & (intervals[:, 1] >= center_point)]
        left = intervals[intervals[:, 1] < center_point]
        right = intervals[intervals[:, 0] > center_point]

        return IntervalTreeNode(
            center_point, center,
            self._build_tree(left), self._build_tree(right)
        )

    @staticmethod
    def _get_center(intervals):
        return intervals[int(intervals.shape[0]/2), 0]

    @staticmethod
    def interval_overlaps(interval1, interval2):
        """Checks whether two intervals overlap each other.

        Args:
            interval1: A tuple of two numbers: the lower and higher bound of
                the first interval.
            interval2: A tuple of two numbers: the lower and higher bound of
                the second interval.

        Returns:
            True if the intervals overlap.
        """
        return interval1[0] <= interval2[1] and interval1[1] >= interval2[0]

    @staticmethod
    def interval_contains(interval, point):
        """Checks whether a point lies in a interval.

        Args:
            interval: A tuple of two numbers: the lower and higher bound of the
                first interval.
            point: The point (just a number)

        Returns:
            True if point lies in the interval.
        """
        return interval[0] <= point <= interval[1]

    # @numba.jit
    def query(self, intervals):
        """Find all overlaps between this tree and a list of intervals.

        Args:
            intervals: A list of intervals. Each interval is a tuple/list of
                two elements: its lower and higher boundary.

        Returns:
            List of lists which contain the overlapping intervals of this tree
            for each element in `intervals`.
        """
        return [self._query(interval, self.root, check_extreme=True)
                for interval in intervals]

    # @numba.jit
    def _query(self, query_interval, node, check_extreme=False):
        # Check this special case: the bounds of the query interval lie outside
        # of the bounds of this tree:
        if (check_extreme
                and IntervalTree.interval_contains(query_interval, self.left)
                and IntervalTree.interval_contains(query_interval, self.right)):
            return []  # TODO: Return all intervals

        # Let's start with the centered intervals
        intervals = [int(interval[2]) for interval in node.center
                     if IntervalTree.interval_overlaps(interval, query_interval)]

        if query_interval[0] <= node.center_point and node.left is not None:
            intervals.extend(self._query(query_interval, node.left))

        if query_interval[1] >= node.center_point and node.right is not None:
            intervals.extend(self._query(query_interval, node.right))

        return intervals

    # @numba.jit
    def query_points(self, points):
        """Find all intervals of this tree which contain one of those points.

        Args:
            points: A list of points.

        Returns:
            List of lists which contain the enclosing intervals of this tree
            for each element in `points`.
        """
        return [self._query_point(point, self.root, check_extreme=True)
                for point in points]

    # @numba.jit
    def _query_point(self, point, node, check_extreme=False):
        # Check this special case: the query point lies outside of the bounds
        # of this tree:
        if check_extreme \
                and not IntervalTree.interval_contains(
                    (self.left, self.right), point):
            return []

        # Let's start with the centered intervals
        intervals = [int(interval[2])
                     for interval in node.center
                     if IntervalTree.interval_contains(interval, point)]

        if point < node.center_point and node.left is not None:
            intervals.extend(self._query_point(point, node))

        if point > node.center_point and node.right is not None:
            intervals.extend(self._query_point(point, node))

        return intervals


class RangeTree:
    def __init__(self, points, shuffle=True, tree_class=None):
        # KD- or ball trees have a very poor building performance for sorted
        # data (such as from SEVIRI) as discussed in this issue:
        # https://github.com/scikit-learn/scikit-learn/issues/7687
        # Hence, we shuffle the data points before inserting them.
        if shuffle:
            self.shuffler = np.arange(points.shape[0])
            np.random.shuffle(self.shuffler)

            points = points[self.shuffler]
        else:
            # The user does not want to shuffle
            self.shuffler = None

        if tree_class is None or tree_class == "Ball":
            tree_class = BallTree
        elif tree_class == "KD":
            tree_class = KDTree

        tree_points = np.column_stack([points, np.zeros_like(points)])
        self.tree = tree_class(tree_points)

    def query_radius(self, points, r):
        query_points = np.column_stack([points, np.zeros_like(points)])

        jagged_pairs = self.tree.query_radius(query_points, r)

        # Build the list of pairs:
        pairs = np.array([
            [build_point, query_point]
            for query_point, build_points in enumerate(jagged_pairs)
            for build_point in build_points
        ]).T

        if self.shuffler is None:
            return pairs
        else:

            # We shuffled the build points in the beginning, so the current
            # indices in the second row (the collocation indices from the build
            # points) are not correct
            pairs[0, :] = self.shuffler[pairs[0, :]]

            return pairs
