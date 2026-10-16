"""Training-only candidate runner. Extra feature computation is inside timing."""
import argparse
import importlib.util
from pathlib import Path
import sys

from round2_robot_features import path_features


def semantic_grid(observation, mission, centered):
    """Mission target masks and colored tokens in a fixed, simple coordinate grid."""
    import numpy as np
    size = 13 if centered else 8
    planes = np.zeros((10, size, size), dtype=np.float32)
    planes[2] = 1  # Outside the observed room is not traversable.
    pairs = mission['pairs'] + [(-1, -1)] * (2 - len(mission['pairs']))
    agent = next((r, c) for r, row in enumerate(observation['image'])
                 for c, cell in enumerate(row) if cell[0] == 10)
    for r, row in enumerate(observation['image']):
        for c, cell in enumerate(row):
            rr, cc = (r - agent[0] + 6, c - agent[1] + 6) if centered else (r, c)
            obj, color = cell
            planes[0, rr, cc] = float(tuple(cell) == pairs[0])
            planes[1, rr, cc] = float(tuple(cell) == pairs[1])
            planes[2, rr, cc] = float(obj == 2)
            planes[3, rr, cc] = float(obj in (5, 6, 7))
            if obj == 11:
                planes[4 + color, rr, cc] = 1
    return planes.reshape(-1)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source', default=str(Path(__file__).resolve().parents[1] / 'solutions' / 'day1_robot_chasing.py'))
    parser.add_argument('--variant', choices=['paths-full-mlp', 'paths-compact-mlp', 'paths-hist',
                                              'semantic-grid-mlp', 'relative-grid-mlp', 'local-grid-mlp',
                                              'target-grid-mlp'], required=True)
    known, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location('candidate_robot', known.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.feature_vector

    def feature_vector(observation):
        import numpy as np
        base = original(observation)
        mission = module.parse_mission(observation['mission'])
        if known.variant in ('semantic-grid-mlp', 'relative-grid-mlp'):
            return np.concatenate((base[1152:], semantic_grid(
                observation, mission, centered=known.variant == 'relative-grid-mlp')))
        if known.variant == 'local-grid-mlp':
            local = semantic_grid(observation, mission, centered=True).reshape(10, 13, 13)[:, 4:9, 4:9]
            interactions = path_features(observation, mission, include_paths=False)
            return np.concatenate((base, local.reshape(-1), np.asarray(interactions, dtype=np.float32)))
        if known.variant == 'target-grid-mlp':
            target_masks = semantic_grid(observation, mission, centered=False).reshape(10, 8, 8)[:2]
            pairs = mission['pairs'] + [(-1, -1)] * (2 - len(mission['pairs']))
            categories = [float(obj == value) for obj, color in pairs for value in (5, 6, 7)]
            categories += [float(color == value) for obj, color in pairs for value in range(6)]
            interactions = path_features(observation, mission, include_paths=False)
            return np.concatenate((base, target_masks.reshape(-1),
                                   np.asarray(categories + interactions, dtype=np.float32)))
        extra = path_features(observation, mission)
        if known.variant == 'paths-compact-mlp':
            base = base[1152:]
        return np.concatenate((base, np.asarray(extra, dtype=np.float32)))

    module.feature_vector = feature_vector
    model = 'hist' if known.variant == 'paths-hist' else 'mlp'
    print(f'candidate={known.variant}; model={model}', flush=True)
    # New releases add their own optional augmentation. The candidate wrapper
    # already supplies its complete feature vector; avoid applying it twice.
    if any(action.dest == 'features' for action in module.make_parser()._actions):
        remaining += ['--features', 'base']
    module.main(remaining + ['--model', model])


if __name__ == '__main__':
    main()
