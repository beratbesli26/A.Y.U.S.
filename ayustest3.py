import math

import cv2
import networkx as nx
import numpy as np

FOTO_ADI = "depremfoto.png"
SONUC_ADI = "afet_rota_sonuclari.png"
RISK_ADI = "afet_risk_haritasi.png"
GRID_SIZE = 40
BLUR_VAL = 1
CANNY_LOW = 255
CANNY_HIGH = 255
EDGE_DILATION = 1
BLOCK_THRESHOLD = 0.10
RISK_THRESHOLD = 0.04
BUFFER_RADIUS = 2
SAFE_ZONE_RADIUS = 1
RISK_EDGE_WEIGHT = 18.0
CLEARANCE_EDGE_WEIGHT = 4.5
ALTERNATIVE_ROUTE_COUNT = 3
ALTERNATIVE_SEARCH_ATTEMPTS = 20
ALTERNATIVE_OVERLAP_LIMIT = 0.65
ACO_ANTS = 45
ACO_ITERATIONS = 30
ACO_ALPHA = 1.2
ACO_BETA = 3.0
ACO_GAMMA = 2.0
ACO_EVAPORATION = 0.35
ACO_DEPOSIT = 160.0
MAX_STEPS_FACTOR = 5
SHOW_WINDOWS = True
SAVE_OUTPUTS = True
ROUTE_COLORS = [
    (0, 255, 0),
    (255, 200, 0),
    (255, 0, 255),
]


def clamp_odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def load_image():
    image = cv2.imread(FOTO_ADI)
    if image is None:
        print(f"HATA: {FOTO_ADI} bulunamadi!")
        raise SystemExit(1)
    return cv2.resize(image, (1408, 768))


def build_risk_maps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_size = clamp_odd(BLUR_VAL)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    if EDGE_DILATION > 1:
        kernel = np.ones((EDGE_DILATION, EDGE_DILATION), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    h, w = image.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    risk_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            roi = edges[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            risk_map[r, c] = float(np.mean(roi == 255))

    if BUFFER_RADIUS > 0:
        kernel_size = BUFFER_RADIUS * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_risk = cv2.dilate(risk_map, kernel, iterations=1)
        smoothed_risk = cv2.GaussianBlur(risk_map, (0, 0), sigmaX=1.1)
        buffered_risk = np.maximum(dilated_risk * 0.80, smoothed_risk)
    else:
        buffered_risk = risk_map.copy()

    blocked_mask = risk_map >= BLOCK_THRESHOLD
    return edges, cell_h, cell_w, risk_map, buffered_risk, blocked_mask


def compute_clearance_map(blocked_mask):
    free_mask = np.where(blocked_mask, 0, 255).astype(np.uint8)
    if not np.any(free_mask):
        return np.zeros_like(blocked_mask, dtype=np.float32)

    clearance = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    max_clearance = float(clearance.max())
    if max_clearance > 0:
        clearance = clearance / max_clearance
    return clearance.astype(np.float32)


def in_bounds(node):
    r, c = node
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def clear_safe_zone(blocked_mask, center, radius):
    if center is None or radius < 0:
        return

    r0, c0 = center
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            node = (r0 + dr, c0 + dc)
            if in_bounds(node):
                blocked_mask[node] = False


def find_corner_anchor(blocked_mask, risk_map, clearance_map, corner):
    depth = max(2, GRID_SIZE // 8)
    corner_span = max(4, GRID_SIZE // 3)
    candidates = []

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if blocked_mask[r, c]:
                continue

            if corner == "top_left":
                near_border = r < depth or c < depth
                within_corner = (r + c) <= corner_span
                corner_distance = r + c
            else:
                near_border = r >= GRID_SIZE - depth or c >= GRID_SIZE - depth
                within_corner = ((GRID_SIZE - 1 - r) + (GRID_SIZE - 1 - c)) <= corner_span
                corner_distance = (GRID_SIZE - 1 - r) + (GRID_SIZE - 1 - c)

            if not (near_border and within_corner):
                continue

            score = corner_distance + (risk_map[r, c] * GRID_SIZE * 3.0) - (clearance_map[r, c] * GRID_SIZE * 2.0)
            candidates.append((score, (r, c)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def add_weighted_edge(graph, u, v, step_cost):
    avg_risk = (graph.nodes[u]["risk"] + graph.nodes[v]["risk"]) * 0.5
    avg_clearance = (graph.nodes[u]["clearance"] + graph.nodes[v]["clearance"]) * 0.5
    edge_weight = step_cost * (
        1.0
        + (avg_risk * RISK_EDGE_WEIGHT)
        + ((1.0 - avg_clearance) * CLEARANCE_EDGE_WEIGHT)
    )
    graph.add_edge(u, v, weight=edge_weight)


def build_graph(blocked_mask, risk_map, clearance_map):
    graph = nx.Graph()

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not blocked_mask[r, c]:
                graph.add_node((r, c), risk=float(risk_map[r, c]), clearance=float(clearance_map[r, c]))

    orthogonal_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    diagonal_dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1))

    for r, c in list(graph.nodes):
        for dr, dc in orthogonal_dirs:
            neighbor = (r + dr, c + dc)
            if neighbor in graph and not graph.has_edge((r, c), neighbor):
                add_weighted_edge(graph, (r, c), neighbor, 1.0)

        for dr, dc in diagonal_dirs:
            neighbor = (r + dr, c + dc)
            side_a = (r + dr, c)
            side_b = (r, c + dc)
            if neighbor not in graph or graph.has_edge((r, c), neighbor):
                continue
            if side_a not in graph or side_b not in graph:
                continue
            add_weighted_edge(graph, (r, c), neighbor, math.sqrt(2.0))

    return graph


def rank_candidate_nodes(graph, target_node, risk_map, clearance_map, limit=25):
    if target_node is None:
        target_node = (0, 0)

    distance_limit = max(6, GRID_SIZE // 3)
    candidates = []
    tr, tc = target_node
    for node in graph.nodes:
        r, c = node
        distance = abs(r - tr) + abs(c - tc)
        if distance > distance_limit:
            continue
        score = distance + (risk_map[node] * GRID_SIZE * 4.0) - (clearance_map[node] * GRID_SIZE * 2.5)
        candidates.append((score, node))

    if not candidates:
        for node in graph.nodes:
            r, c = node
            distance = abs(r - tr) + abs(c - tc)
            score = distance + (risk_map[node] * GRID_SIZE * 4.0) - (clearance_map[node] * GRID_SIZE * 2.5)
            candidates.append((score, node))

    candidates.sort(key=lambda item: item[0])
    return candidates[:limit]


def choose_endpoints(graph, risk_map, clearance_map, start_target, end_target):
    components = list(nx.connected_components(graph))
    if not components:
        return None, None

    node_to_component = {}
    for index, component in enumerate(components):
        for node in component:
            node_to_component[node] = index

    start_candidates = rank_candidate_nodes(graph, start_target, risk_map, clearance_map)
    end_candidates = rank_candidate_nodes(graph, end_target, risk_map, clearance_map)

    best_pair = (None, None)
    best_score = float("inf")

    for start_score, start_node in start_candidates:
        start_component = node_to_component[start_node]
        for end_score, end_node in end_candidates:
            if start_node == end_node:
                continue
            if node_to_component[end_node] != start_component:
                continue
            try:
                corridor_cost = nx.shortest_path_length(graph, start_node, end_node, weight="weight")
            except nx.NetworkXNoPath:
                continue

            endpoint_clearance = graph.nodes[start_node]["clearance"] + graph.nodes[end_node]["clearance"]
            total_score = corridor_cost + (start_score * 3.5) + (end_score * 3.5) - (endpoint_clearance * 8.0)
            if total_score < best_score:
                best_score = total_score
                best_pair = (start_node, end_node)

    return best_pair


def edge_key(u, v):
    return tuple(sorted((u, v)))


def path_cost(graph, path):
    cost = 0.0
    for i in range(len(path) - 1):
        cost += graph.edges[path[i], path[i + 1]]["weight"]
    return cost


def path_length_cells(path):
    length = 0.0
    for i in range(len(path) - 1):
        delta_r = path[i + 1][0] - path[i][0]
        delta_c = path[i + 1][1] - path[i][1]
        length += math.hypot(delta_r, delta_c)
    return length


def run_aco(graph, start_node, end_node):
    pheromone = {edge_key(u, v): 1.0 for u, v in graph.edges}
    best_path = []
    best_cost = float("inf")
    max_steps = max(GRID_SIZE * MAX_STEPS_FACTOR, 10)

    for _ in range(ACO_ITERATIONS):
        successful_paths = []

        for _ in range(ACO_ANTS):
            current = start_node
            path = [current]
            visited = {current}

            while current != end_node and len(path) < max_steps:
                choices = []
                weights = []

                for neighbor in graph.neighbors(current):
                    if neighbor in visited:
                        continue

                    edge = edge_key(current, neighbor)
                    pher = pheromone[edge] ** ACO_ALPHA
                    cost_term = 1.0 / graph.edges[current, neighbor]["weight"]
                    goal_term = 1.0 / (1.0 + abs(neighbor[0] - end_node[0]) + abs(neighbor[1] - end_node[1]))
                    choices.append(neighbor)
                    weights.append(pher * (cost_term ** ACO_BETA) * (goal_term ** ACO_GAMMA))

                if not choices:
                    break

                weight_array = np.array(weights, dtype=np.float64)
                weight_sum = weight_array.sum()
                if weight_sum <= 0:
                    probabilities = np.full(len(choices), 1.0 / len(choices))
                else:
                    probabilities = weight_array / weight_sum

                next_index = np.random.choice(len(choices), p=probabilities)
                current = choices[next_index]
                path.append(current)
                visited.add(current)

            if current != end_node:
                continue

            cost = path_cost(graph, path)
            successful_paths.append((path, cost))
            if cost < best_cost:
                best_cost = cost
                best_path = path

        for edge in pheromone:
            pheromone[edge] *= (1.0 - ACO_EVAPORATION)
            if pheromone[edge] < 0.05:
                pheromone[edge] = 0.05

        for path, cost in successful_paths:
            deposit = ACO_DEPOSIT / max(cost, 1e-6)
            for i in range(len(path) - 1):
                pheromone[edge_key(path[i], path[i + 1])] += deposit

    return best_path, best_cost


def path_overlap_ratio(path_a, path_b):
    set_a = set(path_a[1:-1]) if len(path_a) > 2 else set(path_a)
    set_b = set(path_b[1:-1]) if len(path_b) > 2 else set(path_b)
    if not set_a or not set_b:
        return 1.0
    shared = len(set_a & set_b)
    return shared / max(1, min(len(set_a), len(set_b)))


def add_path_penalties(path, penalty_map, amount):
    for i in range(len(path) - 1):
        edge = edge_key(path[i], path[i + 1])
        penalty_map[edge] = penalty_map.get(edge, 0.0) + amount


def generate_backup_routes(graph, start_node, end_node, primary_path, route_count):
    if not primary_path:
        return []

    routes = [primary_path]
    penalty_map = {}
    add_path_penalties(primary_path, penalty_map, 2.5)

    attempts = 0
    while len(routes) < route_count and attempts < ALTERNATIVE_SEARCH_ATTEMPTS:
        attempts += 1
        try:
            candidate = nx.shortest_path(
                graph,
                source=start_node,
                target=end_node,
                weight=lambda u, v, data: data["weight"] + penalty_map.get(edge_key(u, v), 0.0),
            )
        except nx.NetworkXNoPath:
            break

        overlap = max(path_overlap_ratio(candidate, route) for route in routes)
        if overlap < ALTERNATIVE_OVERLAP_LIMIT:
            routes.append(candidate)
            add_path_penalties(candidate, penalty_map, 2.5 + len(routes))
        else:
            add_path_penalties(candidate, penalty_map, 1.0 + (attempts * 0.25))

    return routes


def compute_route_metrics(graph, path, risk_map, clearance_map, cell_h, cell_w, label):
    if not path:
        return {
            "label": label,
            "cost": float("inf"),
            "length_cells": 0.0,
            "length_px": 0.0,
            "avg_risk": 1.0,
            "max_risk": 1.0,
            "mean_clearance": 0.0,
            "min_clearance": 0.0,
            "safety_score": 0.0,
        }

    risk_values = np.array([risk_map[node] for node in path], dtype=np.float32)
    clearance_values = np.array([clearance_map[node] for node in path], dtype=np.float32)
    length_cells = path_length_cells(path)
    avg_cell_size = (cell_h + cell_w) * 0.5
    length_px = length_cells * avg_cell_size
    avg_risk = float(risk_values.mean())
    max_risk = float(risk_values.max())
    mean_clearance = float(clearance_values.mean())
    min_clearance = float(clearance_values.min())
    normalized_length = length_cells / max(GRID_SIZE * 1.6, 1.0)
    safety_score = 100.0
    safety_score -= avg_risk * 220.0
    safety_score -= max_risk * 70.0
    safety_score -= normalized_length * 10.0
    safety_score += mean_clearance * 28.0
    safety_score += min_clearance * 18.0
    safety_score = float(np.clip(safety_score, 1.0, 99.0))

    return {
        "label": label,
        "cost": path_cost(graph, path),
        "length_cells": length_cells,
        "length_px": length_px,
        "avg_risk": avg_risk,
        "max_risk": max_risk,
        "mean_clearance": mean_clearance,
        "min_clearance": min_clearance,
        "safety_score": safety_score,
    }


def draw_text_block(image, lines, top_left, line_height=24):
    x, y = top_left
    width = 0
    for line in lines:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        width = max(width, text_size[0])

    block_height = (len(lines) * line_height) + 18
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width + 24, y + block_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    cursor_y = y + 24
    for line in lines:
        cv2.putText(image, line, (x + 12, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cursor_y += line_height


def draw_result(image, routes, route_metrics, cell_h, cell_w, risk_map, blocked_mask, start_node, end_node):
    result = image.copy()
    overlay = image.copy()

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            top_left = (c * cell_w, r * cell_h)
            bottom_right = ((c + 1) * cell_w, (r + 1) * cell_h)

            if blocked_mask[r, c]:
                cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1)
            elif risk_map[r, c] >= RISK_THRESHOLD:
                cv2.rectangle(overlay, top_left, bottom_right, (0, 165, 255), -1)

    cv2.addWeighted(overlay, 0.35, result, 0.65, 0, result)

    for route_index in range(min(len(routes), len(ROUTE_COLORS)) - 1, -1, -1):
        path = routes[route_index]
        if not path:
            continue
        points = np.array(
            [[c * cell_w + cell_w // 2, r * cell_h + cell_h // 2] for r, c in path],
            dtype=np.int32,
        )
        thickness = 4 if route_index == 0 else 2
        cv2.polylines(result, [points], False, ROUTE_COLORS[route_index], thickness, cv2.LINE_AA)

    for node, color in ((start_node, (255, 255, 255)), (end_node, (0, 255, 255))):
        center = (node[1] * cell_w + cell_w // 2, node[0] * cell_h + cell_h // 2)
        cv2.circle(result, center, max(5, min(cell_h, cell_w) // 3), color, -1, cv2.LINE_AA)
        cv2.circle(result, center, max(7, min(cell_h, cell_w) // 3 + 2), (0, 0, 0), 2, cv2.LINE_AA)

    info_lines = [
        "Afet guvenli rota planlayici",
        f"Birincil rota skoru: {route_metrics[0]['safety_score']:.1f}/100" if route_metrics else "Birincil rota yok",
        f"Rota uzunlugu: {route_metrics[0]['length_px']:.0f} px" if route_metrics else "Rota uzunlugu: -",
        f"Min aciklik: %{route_metrics[0]['min_clearance'] * 100:.0f}" if route_metrics else "Min aciklik: -",
        f"Alternatif rota sayisi: {max(0, len(routes) - 1)}",
    ]
    draw_text_block(result, info_lines, (18, 18))

    legend_lines = ["Kirmizi: kapali alan", "Turuncu: riskli koridor", "Yesil: birincil rota"]
    if len(routes) > 1:
        legend_lines.append("Mavi/sari: alternatif rotalar")
    draw_text_block(result, legend_lines, (18, 170))

    return result


def draw_risk_heatmap(image, risk_map, blocked_mask, cell_h, cell_w, start_node, end_node):
    risk_8bit = np.clip(risk_map * 255.0 * 2.5, 0, 255).astype(np.uint8)
    heat_small = cv2.applyColorMap(risk_8bit, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heat_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = cv2.addWeighted(image, 0.40, heatmap, 0.60, 0)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not blocked_mask[r, c]:
                continue
            top_left = (c * cell_w, r * cell_h)
            bottom_right = ((c + 1) * cell_w, (r + 1) * cell_h)
            cv2.rectangle(result, top_left, bottom_right, (0, 0, 255), 1)

    for node, color in ((start_node, (255, 255, 255)), (end_node, (0, 255, 255))):
        center = (node[1] * cell_w + cell_w // 2, node[0] * cell_h + cell_h // 2)
        cv2.circle(result, center, max(5, min(cell_h, cell_w) // 3), color, -1, cv2.LINE_AA)
        cv2.circle(result, center, max(7, min(cell_h, cell_w) // 3 + 2), (0, 0, 0), 2, cv2.LINE_AA)

    draw_text_block(result, ["Risk isi haritasi", "Mavi: daha guvenli", "Kirmizi: daha riskli"], (18, 18))
    return result


def generate_route_plan(image):
    _, cell_h, cell_w, raw_risk, buffered_risk, blocked_mask = build_risk_maps(image)
    initial_clearance = compute_clearance_map(blocked_mask)

    start_anchor = find_corner_anchor(blocked_mask, buffered_risk, initial_clearance, "top_left")
    end_anchor = find_corner_anchor(blocked_mask, buffered_risk, initial_clearance, "bottom_right")

    clear_safe_zone(blocked_mask, start_anchor, SAFE_ZONE_RADIUS)
    clear_safe_zone(blocked_mask, end_anchor, SAFE_ZONE_RADIUS)

    clearance_map = compute_clearance_map(blocked_mask)
    graph = build_graph(blocked_mask, buffered_risk, clearance_map)
    default_start = start_anchor if start_anchor is not None else (0, 0)
    default_end = end_anchor if end_anchor is not None else (GRID_SIZE - 1, GRID_SIZE - 1)
    start_node, end_node = choose_endpoints(graph, buffered_risk, clearance_map, default_start, default_end)

    if start_node is None or end_node is None:
        print("Guvenli bir baslangic veya bitis koridoru bulunamadi.")
        raise SystemExit(1)

    primary_path, primary_cost = run_aco(graph, start_node, end_node)
    used_fallback = False

    if not primary_path:
        try:
            primary_path = nx.shortest_path(graph, start_node, end_node, weight="weight")
            primary_cost = path_cost(graph, primary_path)
            used_fallback = True
        except nx.NetworkXNoPath:
            print("Hedefe giden guvenli rota bulunamadi.")
            raise SystemExit(1)

    routes = generate_backup_routes(graph, start_node, end_node, primary_path, ALTERNATIVE_ROUTE_COUNT)
    route_labels = ["Birincil Rota", "Alternatif 1", "Alternatif 2"]
    route_metrics = []
    for index, route in enumerate(routes):
        label = route_labels[index] if index < len(route_labels) else f"Alternatif {index}"
        route_metrics.append(compute_route_metrics(graph, route, buffered_risk, clearance_map, cell_h, cell_w, label))

    result_image = draw_result(image, routes, route_metrics, cell_h, cell_w, buffered_risk, blocked_mask, start_node, end_node)
    risk_image = draw_risk_heatmap(image, buffered_risk, blocked_mask, cell_h, cell_w, start_node, end_node)

    return {
        "cell_h": cell_h,
        "cell_w": cell_w,
        "start_node": start_node,
        "end_node": end_node,
        "primary_cost": primary_cost,
        "used_fallback": used_fallback,
        "routes": routes,
        "route_metrics": route_metrics,
        "result_image": result_image,
        "risk_image": risk_image,
    }


def main():
    image = load_image()
    plan = generate_route_plan(image)

    print(f"Baslangic dugumu: {plan['start_node']}")
    print(f"Bitis dugumu: {plan['end_node']}")
    print("ACO tabanli guvenli rota hesaplaniyor...")
    if plan["used_fallback"]:
        print("ACO yeterli rota bulamadi, agirlikli en kisa yol kullanildi.")

    for metric in plan["route_metrics"]:
        print(
            f"{metric['label']}: skor={metric['safety_score']:.1f}/100 | "
            f"uzunluk={metric['length_cells']:.1f} hucre | "
            f"ortalama risk=%{metric['avg_risk'] * 100:.1f} | "
            f"min aciklik=%{metric['min_clearance'] * 100:.1f}"
        )

    if SAVE_OUTPUTS:
        cv2.imwrite(SONUC_ADI, plan["result_image"])
        cv2.imwrite(RISK_ADI, plan["risk_image"])
        print(f"Sonuc gorseli kaydedildi: {SONUC_ADI}")
        print(f"Risk haritasi kaydedildi: {RISK_ADI}")

    if SHOW_WINDOWS:
        cv2.imshow("Afet Yonetim - Guvenli Rotalar", plan["result_image"])
        cv2.imshow("Afet Yonetim - Risk Haritasi", plan["risk_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
