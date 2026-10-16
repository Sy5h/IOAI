"""Observable mission-stage and path features; no action is forced by planning."""
from collections import deque

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = tuple(tuple((r + dr) * 8 + c + dc if 0 <= r + dr < 8 and 0 <= c + dc < 8 else -1
                        for dr, dc in DELTAS) for r in range(8) for c in range(8))


def distances_to(goals, walkable):
    distances = [64] * 64
    queue = deque()
    for goal in goals:
        if walkable[goal] and distances[goal]:
            distances[goal] = 0
            queue.append(goal)
    while queue:
        cell = queue.popleft()
        for other in NEIGHBORS[cell]:
            if other >= 0 and walkable[other] and distances[other] == 64:
                distances[other] = distances[cell] + 1
                queue.append(other)
    return distances


def path_features(observation, parsed_mission, include_paths=True):
    """Keep absolute actions and both possible token-passability assumptions.

    Position distances describe geometry, not the robot's unknown policy. Blocked
    moves retain their orientation/interaction features and are never masked.
    """
    cells = [tuple(cell) for row in observation['image'] for cell in row]
    objects = [cell[0] for cell in cells]
    position = objects.index(10)
    r, c = divmod(position, 8)
    direction = int(observation['direction'])
    carrying = tuple(observation.get('carrying') or (-1, -1))
    pairs = parsed_mission['pairs'] + [(-1, -1)] * (2 - len(parsed_mission['pairs']))
    source, destination = pairs
    locations = [[i for i, cell in enumerate(cells) if cell == pair] for pair in pairs]
    carrying_source = source[0] >= 0 and carrying == source
    is_put = parsed_mission['family'] == 2
    active = locations[1] if is_put and carrying_source else locations[0]
    adjacent_destination = {neighbor for i in locations[1] for neighbor in NEIGHBORS[i] if neighbor >= 0}
    values = [float(carrying_source), float(is_put and carrying_source),
              float(is_put and not carrying_source), float(carrying[0] >= 0 and not carrying_source)]
    # Current and post-turn interaction conditions, with absolute action order.
    for action, neighbor in enumerate(NEIGHBORS[position]):
        cell = cells[neighbor] if neighbor >= 0 else (2, 0)
        empty = cell[0] == 1
        pick_source = source[0] >= 0 and cell == source and carrying[0] < 0
        drop_near_destination = empty and neighbor in adjacent_destination and carrying_source and is_put
        values += [float(empty), float(cell[0] == 11), float(pick_source),
                   float(drop_near_destination), float(action == direction and pick_source),
                   float(action == direction and drop_near_destination)]
    # Relative-to-facing offsets supplement, rather than replace, absolute ones.
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
    if not include_paths:
        return values
    for allow_tokens in (False, True):
        walkable = [obj in ((1, 10, 11) if allow_tokens else (1, 10)) for obj in objects]
        source_goals = {n for i in locations[0] for n in NEIGHBORS[i] if n >= 0 and walkable[n]}
        destination_goals = {n for i in locations[1] for n in NEIGHBORS[i] if n >= 0 and walkable[n]}
        # A carrying robot must stand next to an empty drop cell adjacent to the
        # destination. A current robot cell can become empty after moving away.
        drop_cells = {i for i in adjacent_destination if objects[i] in (1, 10)}
        drop_goals = {n for i in drop_cells for n in NEIGHBORS[i] if n >= 0 and walkable[n]}
        active_goals = drop_goals if is_put and carrying_source else source_goals
        for goals in (source_goals, destination_goals, drop_goals, active_goals):
            distances = distances_to(goals, walkable)
            current = distances[position]
            values += [min(current, 12) / 12, float(current < 64), float(current == 0)]
            action_distances = []
            for neighbor in NEIGHBORS[position]:
                can_move = neighbor >= 0 and walkable[neighbor]
                successor = neighbor if can_move else position
                distance = distances[successor]
                action_distances.append(distance)
                values += [min(distance, 12) / 12, float(distance < current),
                           float(distance == 0), float(can_move)]
            best = min(action_distances)
            values += [float(distance == best and best < 64) for distance in action_distances]
    return values
