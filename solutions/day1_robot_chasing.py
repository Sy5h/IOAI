"""IOAI 2026 Day 1: Robot Chasing, per-robot small classifiers.

Uses only training labels, deterministic instruction parsing and observable grid
features. No path-planner assumptions, pretrained model or test-label reads.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
import zlib


COLORS = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "gray": 5}
OBJECTS = {"key": 5, "ball": 6, "box": 7}
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = tuple(tuple((r + dr) * 8 + c + dc if 0 <= r + dr < 8 and 0 <= c + dc < 8 else -1
                        for dr, dc in DELTAS) for r in range(8) for c in range(8))
PAIR_RE = re.compile(r"\b(red|green|blue|purple|yellow|gr[ae]y)\s+(?:colou?red\s+)?(key|ball|box)\b")


def parse_mission(mission):
    text = re.sub(r"[-_]", " ", str(mission).lower())
    pairs = [(OBJECTS[obj], COLORS[color]) for color, obj in PAIR_RE.findall(text)]
    if len(pairs) >= 2:
        family = 2  # put source next to destination
    elif (re.search(r"\b(pick|pickup|grab|collect|take|fetch|lift|retrieve|acquire|obtain)\b", text)
          or (re.search(r"\bget\b", text)
              and not re.search(r"\bget\s+(close|closer|near|to|next)\b", text))):
        family = 1
    else:
        family = 0  # approach; vocabulary hash remains available for unknown verbs
    return {"family": family, "pairs": pairs[:2], "text": text}


def robot_position(image):
    positions = [(r, c) for r, row in enumerate(image) for c, cell in enumerate(row) if cell[0] == 10]
    if len(positions) != 1:
        raise ValueError("an observation must contain exactly one robot (object ID 10)")
    return positions[0]


def front_cell(observation, direction=None):
    """Facing uses absolute directions; pickup/drop affect this adjacent cell."""
    r, c = robot_position(observation["image"])
    direction = observation["direction"] if direction is None else direction
    if direction not in range(4):
        raise ValueError("direction must be 0..3")
    dr, dc = DELTAS[direction]
    rr, cc = r + dr, c + dc
    if not (0 <= rr < 8 and 0 <= cc < 8):
        return (rr, cc), (2, 0)
    return (rr, cc), tuple(observation["image"][rr][cc])


def movement_observation(observation, action):
    """Describe an attempted move without declaring blocked actions invalid.

    A blocked move still changes facing to action. The returned cell data lets
    the classifier learn which such orientation changes the robot actually uses.
    """
    if action not in range(4):
        raise ValueError("movement action must be 0..3")
    coordinate, cell = front_cell(observation, action)
    return {"attempted_cell": coordinate, "object": cell[0], "color": cell[1],
            "new_direction": action}


def feature_vector(observation):
    import numpy as np

    grid = np.asarray(observation["image"], dtype=np.int16)
    if grid.shape != (8, 8, 2):
        raise ValueError("image must have shape 8x8x2")
    object_ids, color_ids = grid[:, :, 0], grid[:, :, 1]
    if object_ids.min() < 0 or object_ids.max() > 11 or color_ids.min() < 0 or color_ids.max() > 5:
        raise ValueError("unexpected object or color ID")
    r, c = robot_position(observation["image"])
    direction = int(observation["direction"])
    if direction not in range(4):
        raise ValueError("direction must be 0..3")
    mission = parse_mission(observation["mission"])
    carrying = observation.get("carrying")
    carrying = (-1, -1) if carrying is None else tuple(carrying)
    target_pairs = mission["pairs"] + [(-1, -1)] * (2 - len(mission["pairs"]))

    # Full spatial information stays available, including distractors and tokens.
    obj_features = np.eye(12, dtype=np.float32)[object_ids].reshape(-1)
    color_features = np.eye(6, dtype=np.float32)[color_ids]
    color_features[(object_ids == 1) | (object_ids == 2)] = 0
    values = [r / 7, c / 7, float(carrying[0] >= 0)]
    values += [float(direction == k) for k in range(4)]
    values += [float(mission["family"] == k) for k in range(3)]
    values += [float(carrying[0] == k) for k in (5, 6, 7, 11)]
    values += [float(carrying[1] == k) for k in range(6)]

    for obj, color in target_pairs:
        coordinates = np.argwhere((object_ids == obj) & (color_ids == color))
        values += [obj / 11, color / 5, len(coordinates) / 6]
        if len(coordinates):
            offsets = coordinates - np.array([r, c])
            distances = np.abs(offsets).sum(axis=1)
            nearest = int(distances.argmin())
            rr, cc = coordinates[nearest]
            dr, dc = offsets[nearest]
            values += [rr / 7, cc / 7, dr / 6, dc / 6, distances[nearest] / 12,
                       float(distances[nearest] == 1), float(dr == 0), float(dc == 0)]
        else:
            values += [-1, -1, 0, 0, 1, 0, 0, 0]
        values.append(float(tuple(carrying) == (obj, color) and obj >= 0))

    # Per-color/type counts and nearest offsets expose possible robot preferences.
    for obj in (5, 6, 7, 11):
        for color in range(6):
            coordinates = np.argwhere((object_ids == obj) & (color_ids == color))
            if len(coordinates):
                offsets = coordinates - np.array([r, c])
                nearest = offsets[np.abs(offsets).sum(axis=1).argmin()]
                values += [len(coordinates) / 6, nearest[0] / 6, nearest[1] / 6]
            else:
                values += [0, 0, 0]

    for action in range(4):
        move = movement_observation(observation, action)
        obj, color = move["object"], move["color"]
        values += [obj / 11, color / 5, float(obj == 1), float(action == direction)]
        values += [float((obj, color) == pair) for pair in target_pairs]
    _, front = front_cell(observation)
    source, destination = target_pairs
    source_in_front = source[0] >= 0 and front == source
    carrying_source = source[0] >= 0 and tuple(carrying) == source
    values += [float(source_in_front), float(carrying_source),
               float(mission["family"] == 2 and not carrying_source),
               float(mission["family"] == 2 and carrying_source)]

    # Small deterministic bag of words helps when the simple verb parser misses a
    # phrase. Hashing is stable across processes and does not require training text.
    word_features = np.zeros(32, dtype=np.float32)
    for word in re.findall(r"[a-z]+", mission["text"]):
        word_features[zlib.crc32(word.encode("utf-8")) % 32] += 0.2
    return np.concatenate((obj_features, color_features.reshape(-1),
                           np.asarray(values, dtype=np.float32), word_features))


def interaction_features(observation, mission):
    """Observable pickup/drop conditions and orientation-relative offsets.

    These are classifier inputs. Even blocked movement actions remain available,
    since attempting movement also changes the robot's facing direction.
    """
    cells = [tuple(cell) for row in observation['image'] for cell in row]
    objects = [cell[0] for cell in cells]
    position = objects.index(10)
    r, c = divmod(position, 8)
    direction = int(observation['direction'])
    carrying = tuple(observation.get('carrying') or (-1, -1))
    pairs = mission['pairs'] + [(-1, -1)] * (2 - len(mission['pairs']))
    source, destination = pairs
    locations = [[i for i, cell in enumerate(cells) if cell == pair] for pair in pairs]
    carrying_source = source[0] >= 0 and carrying == source
    is_put = mission['family'] == 2
    active = locations[1] if is_put and carrying_source else locations[0]
    adjacent_destination = {neighbor for i in locations[1] for neighbor in NEIGHBORS[i] if neighbor >= 0}
    values = [float(carrying_source), float(is_put and carrying_source),
              float(is_put and not carrying_source), float(carrying[0] >= 0 and not carrying_source)]
    for action, neighbor in enumerate(NEIGHBORS[position]):
        cell = cells[neighbor] if neighbor >= 0 else (2, 0)
        empty = cell[0] == 1
        pick_source = source[0] >= 0 and cell == source and carrying[0] < 0
        drop_near_destination = empty and neighbor in adjacent_destination and carrying_source and is_put
        values += [float(empty), float(cell[0] == 11), float(pick_source),
                   float(drop_near_destination), float(action == direction and pick_source),
                   float(action == direction and drop_near_destination)]
    forward = DELTAS[direction]
    right = (forward[1], -forward[0])
    for targets in (locations[0], locations[1], active):
        offsets = [(divmod(i, 8)[0] - r, divmod(i, 8)[1] - c) for i in targets]
        if offsets:
            dr, dc = min(offsets, key=lambda pair: abs(pair[0]) + abs(pair[1]))
            values += [dr / 6, dc / 6, (dr * forward[0] + dc * forward[1]) / 6,
                       (dr * right[0] + dc * right[1]) / 6,
                       float(sum(abs(a) + abs(b) == abs(dr) + abs(dc) for a, b in offsets)) / 6]
        else:
            values += [0, 0, 0, 0, 0]
    values.append(min((abs(a // 8 - b // 8) + abs(a % 8 - b % 8)
                       for a in locations[0] for b in locations[1]), default=12) / 12)
    return values


def local_feature_vector(observation):
    """Preserve the full original grid; append a centered 5x5 semantic view."""
    import numpy as np

    base = feature_vector(observation)
    mission = parse_mission(observation['mission'])
    pairs = mission['pairs'] + [(-1, -1)] * (2 - len(mission['pairs']))
    r, c = robot_position(observation['image'])
    planes = np.zeros((10, 5, 5), dtype=np.float32)
    planes[2] = 1  # Outside the observed room is represented as wall.
    for rr in range(max(0, r - 2), min(8, r + 3)):
        for cc in range(max(0, c - 2), min(8, c + 3)):
            cell = tuple(observation['image'][rr][cc])
            obj, color = cell
            row, col = rr - r + 2, cc - c + 2
            planes[0, row, col] = float(cell == pairs[0])
            planes[1, row, col] = float(cell == pairs[1])
            planes[2, row, col] = float(obj == 2)
            planes[3, row, col] = float(obj in (5, 6, 7))
            if obj == 11:
                planes[4 + color, row, col] = 1
    return np.concatenate((base, planes.reshape(-1),
                           np.asarray(interaction_features(observation, mission), dtype=np.float32)))


def model_features(observation, args):
    return (local_feature_vector(observation) if getattr(args, 'features', 'base') == 'local'
            else feature_vector(observation))


def majority_actions(observations, labels):
    if len(observations) != len(labels) or not labels:
        raise ValueError("training observations and labels must have equal, nonzero length")
    if any(type(y) is not int or y not in range(6) for y in labels):
        raise ValueError("labels must be integer actions 0..5")
    counters = {robot: Counter() for robot in range(6)}
    for observation, label in zip(observations, labels):
        robot = observation["robot_id"]
        if robot not in counters:
            raise ValueError("robot_id must be 0..5")
        counters[robot][label] += 1
    global_default = Counter(labels).most_common(1)[0][0]
    return {robot: counts.most_common(1)[0][0] if counts else global_default
            for robot, counts in counters.items()}


def write_predictions(predictions, output):
    if any(type(p) is not int or p not in range(6) for p in predictions):
        raise ValueError("predictions must be integer actions 0..5")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions), encoding="utf-8")


def train_predict(train, labels, test, defaults, args, stop_at):
    import numpy as np
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(args.cpu_threads, os.cpu_count() or 1)))
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = [int(defaults[o["robot_id"]]) for o in test]
    feature_count = len(model_features(train[0], args))
    states = []
    # Prepare one robot at a time; feature time is included in the total budget.
    for robot in range(6):
        if time.monotonic() >= stop_at:
            break
        indices = [i for i, o in enumerate(train) if o["robot_id"] == robot]
        if len(indices) < 10:
            continue
        rng.shuffle(indices)
        validation_size = max(1, int(len(indices) * args.validation_fraction))
        x = np.empty((len(indices), feature_count), dtype=np.float32)
        for row, index in enumerate(indices):
            x[row] = model_features(train[index], args)
            if row % 256 == 0 and time.monotonic() >= stop_at:
                return predictions
        y = np.asarray([labels[i] for i in indices], dtype=np.int64)
        tx = torch.from_numpy(x[validation_size:]).to(device)
        ty = torch.from_numpy(y[validation_size:]).to(device)
        vx = torch.from_numpy(x[:validation_size]).to(device)
        vy = torch.from_numpy(y[:validation_size]).to(device)
        model = nn.Sequential(nn.Linear(feature_count, args.hidden), nn.ReLU(),
                              nn.Dropout(0.05), nn.Linear(args.hidden, 64), nn.ReLU(),
                              nn.Linear(64, 6)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        baseline = float((y[:validation_size] == defaults[robot]).mean())
        states.append({"robot": robot, "model": model, "optimizer": optimizer,
                       "tx": tx, "ty": ty, "vx": vx, "vy": vy,
                       "best_score": baseline, "best": None, "stale": 0, "epochs": 0})
    # Round-robin epochs ensure a slow robot does not consume everyone else's time.
    for epoch in range(args.epochs):
        did_work = False
        for state in states:
            if time.monotonic() >= stop_at:
                break
            if state["stale"] >= args.patience:
                continue
            did_work = True
            model, optimizer = state["model"], state["optimizer"]
            model.train()
            order = torch.randperm(len(state["ty"]), device=device)
            complete = True
            for start in range(0, len(order), args.batch_size):
                if time.monotonic() >= stop_at:
                    complete = False
                    break
                idx = order[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(state["tx"][idx]), state["ty"][idx])
                loss.backward()
                optimizer.step()
            if not complete:
                break
            state["epochs"] += 1
            model.eval()
            with torch.inference_mode():
                correct = sum(int((model(state["vx"][s:s + args.batch_size]).argmax(dim=1)
                                   == state["vy"][s:s + args.batch_size]).sum().item())
                              for s in range(0, len(state["vy"]), args.batch_size))
            accuracy = correct / len(state["vy"])
            if accuracy > state["best_score"] + 1e-6:
                state["best_score"] = accuracy
                state["best"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                state["stale"] = 0
            else:
                state["stale"] += 1
        if not did_work or time.monotonic() >= stop_at:
            break
    # Inference reserve is outside stop_at; untouched robots keep majority output.
    for state in states:
        print(f"robot {state['robot']}: epochs={state['epochs']}, "
              f"validation={state['best_score']:.4f}, learned={state['best'] is not None}", flush=True)
        if state["best"] is None:
            continue
        model = state["model"]
        model.load_state_dict(state["best"])
        model.eval()
        indices = [i for i, o in enumerate(test) if o["robot_id"] == state["robot"]]
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            x = torch.from_numpy(np.stack([model_features(test[i], args) for i in batch])).to(device)
            with torch.inference_mode():
                labels_out = model(x).argmax(dim=1).cpu().tolist()
            for index, action in zip(batch, labels_out):
                predictions[index] = int(action)
    return predictions


def hist_train_predict(train, labels, test, defaults, args, stop_at):
    """CPU-friendly boosted trees, selected using a training-only holdout.

    Uses the 176 engineered/word features after the 1152 full-grid one-hot
    features. The MLP option retains the complete grid representation.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    predictions = [int(defaults[o["robot_id"]]) for o in test]
    for robot in range(6):
        if time.monotonic() >= stop_at:
            print("training budget exhausted; untrained robots keep majority actions", flush=True)
            break
        train_indices = [i for i, o in enumerate(train) if o["robot_id"] == robot]
        test_indices = [i for i, o in enumerate(test) if o["robot_id"] == robot]
        if not train_indices or not test_indices:
            continue
        features = []
        for row, i in enumerate(train_indices):
            features.append(feature_vector(train[i])[1152:])
            if row % 256 == 0 and time.monotonic() >= stop_at:
                print("training budget exhausted during features; partial predictions retained", flush=True)
                return predictions
        x = np.stack(features)
        y = np.asarray([labels[i] for i in train_indices])
        xt = np.stack([feature_vector(test[i])[1152:] for i in test_indices])
        model = HistGradientBoostingClassifier(
            max_iter=args.hist_iterations, learning_rate=0.1,
            max_leaf_nodes=31, l2_regularization=1,
            early_stopping=len(train_indices) >= 100,
            validation_fraction=args.validation_fraction, n_iter_no_change=15,
            random_state=args.seed + robot)
        with threadpool_limits(limits=args.cpu_threads):
            model.fit(x, y)
            predicted = model.predict(xt)
        for index, action in zip(test_indices, predicted):
            predictions[index] = int(action)
        write_predictions(predictions, args.output)
        print(f"robot {robot}: hist iterations={model.n_iter_}, features={x.shape[1]}", flush=True)
    return predictions


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--budget", type=float, default=270, help="soft total seconds; statement limit is 300s")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--model", choices=("hist", "mlp"), default="mlp")
    parser.add_argument("--features", choices=("base", "local"), default="local",
                        help="MLP features: original base or full grid plus local view; hist keeps its original compact features")
    parser.add_argument("--hist-iterations", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if min(args.hidden, args.batch_size, args.patience, args.cpu_threads, args.hist_iterations) < 1:
        raise ValueError("model sizes, patience and CPU threads must be positive")
    if not 0 < args.validation_fraction < 0.5 or args.learning_rate <= 0:
        raise ValueError("invalid validation fraction or learning rate")
    started = time.monotonic()
    root = Path(args.dataset_root)
    train = json.loads((root / "train" / "observations.json").read_text(encoding="utf-8"))
    labels = json.loads((root / "train" / "labels.json").read_text(encoding="utf-8"))
    test = json.loads((root / "test_public" / "observations.json").read_text(encoding="utf-8"))
    defaults = majority_actions(train, labels)
    predictions = [int(defaults[o["robot_id"]]) for o in test]
    write_predictions(predictions, args.output)
    if args.epochs > 0 and time.monotonic() < started + args.budget - 20:
        try:
            predictor = hist_train_predict if args.model == "hist" else train_predict
            predictions = predictor(train, labels, test, defaults, args, started + args.budget - 20)
        except Exception as exc:
            # A later robot can fail after earlier robots were already saved.
            # Preserve that progress instead of overwriting it with all-majority.
            saved = json.loads(Path(args.output).read_text(encoding="utf-8"))
            if len(saved) == len(test) and all(type(x) is int and 0 <= x < 6 for x in saved):
                predictions = saved
            print(f"training failed; saved partial predictions/fallback retained: {exc}", file=sys.stderr)
    write_predictions(predictions, args.output)
    print(f"wrote {len(predictions)} predictions in {time.monotonic() - started:.1f}s")
    return predictions


if __name__ == "__main__":
    main()
