"""Synthetic geometry checks; these are not competition scores."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features = load('path_features_test', ROOT / 'benchmarks' / 'round2_robot_features.py')
solver = load('robot_solver_test', ROOT / 'solutions' / 'day1_robot_chasing.py')
sys.path.insert(0, str(ROOT / 'benchmarks'))
candidate = load('robot_candidate_test', ROOT / 'benchmarks' / 'round2_robot_candidate.py')


def observation():
    image = [[[2, 0] if r in (0, 7) or c in (0, 7) else [1, 0]
              for c in range(8)] for r in range(8)]
    image[3][3] = [10, 0]
    image[3][5] = [7, 2]
    return {'robot_id': 0, 'image': image, 'direction': 3,
            'mission': 'put the red ball next to the blue box', 'carrying': [6, 0]}


class PathFeaturesTest(unittest.TestCase):
    def test_standalone_matches_selected_candidate_and_preserves_original(self):
        import numpy as np
        for direction in range(4):
            obs = observation()
            obs['direction'] = direction
            mission = solver.parse_mission(obs['mission'])
            local = candidate.semantic_grid(obs, mission, True).reshape(10, 13, 13)[:, 4:9, 4:9]
            reference = np.concatenate((solver.feature_vector(obs), local.reshape(-1),
                                        np.asarray(features.path_features(obs, mission, False), dtype=np.float32)))
            actual = solver.local_feature_vector(obs)
            self.assertEqual(len(actual), 1622)
            np.testing.assert_array_equal(actual[:1328], solver.feature_vector(obs))
            np.testing.assert_array_equal(actual, reference)
            self.assertEqual(solver.interaction_features(obs, mission), features.path_features(obs, mission, False))

    def test_local_view_at_border_preserves_wall_padding(self):
        import numpy as np
        obs = observation()
        obs['image'][3][3] = [1, 0]
        obs['image'][1][1] = [10, 0]
        mission = solver.parse_mission(obs['mission'])
        reference = candidate.semantic_grid(obs, mission, True).reshape(10, 13, 13)[:, 4:9, 4:9]
        actual = solver.local_feature_vector(obs)[1328:1578].reshape(10, 5, 5)
        np.testing.assert_array_equal(actual, reference)
        self.assertEqual(actual[2, 0, 0], 1)

    def test_static_features_equal_path_prefix(self):
        obs = observation()
        mission = solver.parse_mission(obs['mission'])
        self.assertEqual(features.path_features(obs, mission, include_paths=False),
                         features.path_features(obs, mission)[:44])

    def test_semantic_grid_centers_robot_but_keeps_absolute_axes(self):
        obs = observation()
        obs['image'][2][3] = [6, 0]
        obs['image'][5][4] = [11, 4]
        grid = candidate.semantic_grid(obs, solver.parse_mission(obs['mission']), True).reshape(10, 13, 13)
        self.assertEqual(grid[0, 5, 6], 1)  # source one cell up
        self.assertEqual(grid[1, 6, 8], 1)  # destination two cells right
        self.assertEqual(grid[8, 8, 7], 1)  # yellow token
        obs['direction'] = 0
        other = candidate.semantic_grid(obs, solver.parse_mission(obs['mission']), True)
        self.assertEqual(grid.reshape(-1).tolist(), other.tolist())

    def test_bfs_routes_around_blocked_cell(self):
        walkable = [False] * 64
        for row, col in [(3, 1), (2, 1), (2, 2), (2, 3), (3, 3)]:
            walkable[row * 8 + col] = True
        distances = features.distances_to({3 * 8 + 3}, walkable)
        self.assertEqual(distances[3 * 8 + 1], 4)
        self.assertEqual(distances[3 * 8 + 2], 64)

    def test_drop_condition_requires_adjacent_empty_front(self):
        obs = observation()
        x = features.path_features(obs, solver.parse_mission(obs['mission']))
        self.assertEqual(len(x), 228)
        self.assertEqual(x[4 + 3 * 6 + 3], 1)  # right drop condition
        self.assertEqual(x[4 + 3 * 6 + 5], 1)  # currently facing right
        obs['direction'] = 0
        y = features.path_features(obs, solver.parse_mission(obs['mission']))
        self.assertEqual(y[4 + 3 * 6 + 3], 1)
        self.assertEqual(y[4 + 3 * 6 + 5], 0)
        obs['image'][3][4] = [5, 1]
        z = features.path_features(obs, solver.parse_mission(obs['mission']))
        self.assertEqual(z[4 + 3 * 6 + 3], 0)

    def test_wrong_carrying_object_is_not_drop_completion(self):
        obs = observation()
        obs['carrying'] = [6, 1]
        x = features.path_features(obs, solver.parse_mission(obs['mission']))
        self.assertEqual(x[4 + 3 * 6 + 3], 0)
        self.assertEqual(x[3], 1)

    def test_blocked_target_move_keeps_pickup_orientation_signal(self):
        obs = observation()
        obs.update(mission='pick up the red ball', carrying=None, direction=0)
        obs['image'][2][3] = [6, 0]
        x = features.path_features(obs, solver.parse_mission(obs['mission']))
        self.assertEqual(x[4 + 2], 1)
        self.assertEqual(x[4 + 4], 1)
        self.assertEqual(x[4], 0)  # target is an object, not an empty move cell

    def test_token_passability_is_represented_both_ways(self):
        obs = observation()
        obs['image'] = [[[2, 0] for _ in range(8)] for _ in range(8)]
        obs['image'][3][1] = [10, 0]
        obs['image'][3][2] = [11, 0]
        obs['image'][3][3] = [1, 0]
        obs['image'][3][4] = [6, 0]
        obs.update(mission='approach the red ball', carrying=None)
        x = features.path_features(obs, solver.parse_mission(obs['mission']))
        # 44 static features, 4 goal families * 23 values per token assumption.
        self.assertEqual(x[44 + 1], 0)
        self.assertEqual(x[44 + 4 * 23 + 1], 1)
        self.assertAlmostEqual(x[44 + 4 * 23], 2 / 12)


if __name__ == '__main__':
    unittest.main()
