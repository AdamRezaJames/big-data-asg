"""Connectivity, sampled centrality and route sensitivity on the actual road graph.

Flood profiles are observed aggregates from the input data, never ML predictions.
"""
import json
import time
import networkx as nx
import numpy as np
import pandas as pd
from common import ROOT, OUT, SEED, setup, save_json, sha256


def main():
    setup()
    start = time.perf_counter()
    edges_path = ROOT / 'data/silver/road_edges.csv'
    edges = pd.read_csv(edges_path)
    segments = pd.read_csv(ROOT / 'data/silver/road_segments.csv', dtype={'kelurahan_ids': str}).fillna({'kelurahan_ids': '', 'name': ''})
    nodes = pd.read_csv(ROOT / 'data/silver/road_nodes.csv')
    graph = nx.DiGraph()
    for e in edges.itertuples():
        attrs = {'length_m': float(e.length_m), 'exposure_index': None if pd.isna(e.exposure_index) else float(e.exposure_index),
                 'segment_id': e.segment_id, 'exposure_unknown': bool(e.exposure_unknown)}
        if not graph.has_edge(e.from_node, e.to_node) or e.length_m < graph[e.from_node][e.to_node]['length_m']:
            graph.add_edge(int(e.from_node), int(e.to_node), **attrs)
    weak = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    strong = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    print(f'Graph loaded: {len(graph)} nodes, {graph.number_of_edges()} edges; centrality k=32 next', flush=True)
    # Fixed 32 sampled source vertices; approximation is labelled in all outputs.
    centrality = nx.betweenness_centrality(graph, k=32, normalized=True, weight='length_m', seed=SEED)
    metadata = nodes.set_index('osm_node_id')
    names = {}
    for s in segments.itertuples():
        for n in (s.from_node, s.to_node):
            if s.name:
                names.setdefault(n, set()).add(s.name)
    top = sorted(centrality, key=centrality.get, reverse=True)[:20]
    pd.DataFrame([{'osm_node_id': n, 'sampled_betweenness': centrality[n],
                   'in_degree': graph.in_degree(n), 'out_degree': graph.out_degree(n),
                   'longitude': float(metadata.loc[n, 'longitude']), 'latitude': float(metadata.loc[n, 'latitude']),
                   'incident_names': ' | '.join(sorted(names.get(n, set())))} for n in top]).to_csv(OUT / 'graph_top_nodes.csv', index=False)
    source_routes = json.loads((ROOT / 'data/gold/route_comparison.json').read_text())
    points = source_routes['points']
    route_rows, route_features = [], []
    by_id = nodes.set_index('osm_node_id')
    for destination in points[1:]:
        for lam in [0, 1, 2, 4]:
            for unknown in ([0] if lam == 0 else [0, 1]):
                def cost(u, v, attr):
                    exposure = attr['exposure_index'] if attr['exposure_index'] is not None else unknown
                    return attr['length_m'] * (1 + lam * exposure)
                a, b = points[0]['snapped_osm_node'], destination['snapped_osm_node']
                path = nx.shortest_path(graph, a, b, weight=cost)
                attrs = [graph[u][v] for u, v in zip(path, path[1:])]
                row = {'destination': destination['kelurahan'], 'lambda': lam, 'unknown_penalty': unknown,
                       'distance_m': sum(e['length_m'] for e in attrs),
                       'exposure_weighted_m': sum(e['length_m'] * (e['exposure_index'] or 0) for e in attrs),
                       'unknown_length_m': sum(e['length_m'] for e in attrs if e['exposure_unknown']),
                       'objective_cost_m': sum(cost(u, v, graph[u][v]) for u, v in zip(path, path[1:])),
                       'nodes': len(path)}
                route_rows.append(row)
                route_features.append({'type': 'Feature', 'properties': row,
                    'geometry': {'type': 'LineString', 'coordinates': [[float(by_id.loc[n, 'longitude']), float(by_id.loc[n, 'latitude'])] for n in path]}})
    pd.DataFrame(route_rows).to_csv(OUT / 'graph_routes.csv', index=False)
    save_json(OUT / 'graph_routes.geojson', {'type': 'FeatureCollection', 'features': route_features})
    # A structural counterfactual, not an observed flood closure.
    blocked = top[:5]
    altered = graph.copy()
    altered.remove_nodes_from(blocked)
    disruption = []
    for destination in points[1:]:
        a, b = points[0]['snapped_osm_node'], destination['snapped_osm_node']
        original = nx.shortest_path_length(graph, a, b, weight='length_m')
        exists = a in altered and b in altered and nx.has_path(altered, a, b)
        detour = nx.shortest_path_length(altered, a, b, weight='length_m') if exists else None
        disruption.append({'destination': destination['kelurahan'], 'baseline_m': original,
                           'path_exists_after_removal': exists, 'after_removal_m': detour,
                           'detour_pct': 100 * (detour / original - 1) if exists else None})
    pd.DataFrame(disruption).to_csv(OUT / 'graph_node_removal.csv', index=False)
    # Count only unique segment lengths, not duplicated directions.
    known = segments.exposure_index.notna()
    pd.DataFrame({'weak_component_rank': np.arange(1, len(weak) + 1),
                  'node_count': [len(c) for c in weak]}).to_csv(OUT / 'graph_components.csv', index=False)
    summary = {'nodes': len(graph), 'directed_edges': graph.number_of_edges(), 'segments': len(segments),
        'weak_components': len(weak), 'strong_components': len(strong),
        'largest_weak_nodes': len(weak[0]), 'largest_strong_nodes': len(strong[0]),
        'largest_weak_share_pct': 100*len(weak[0])/len(graph),
        'largest_strong_share_pct': 100*len(strong[0])/len(graph),
        'max_total_degree': max(dict(graph.degree()).values()),
        'known_exposure_segments': int(known.sum()), 'unknown_exposure_segments': int((~known).sum()),
        'segment_total_length_m': float(segments.length_m.sum()),
        'unknown_segment_length_m': float(segments.loc[~known, 'length_m'].sum()),
        'known_segment_length_m': float(segments.loc[known, 'length_m'].sum()),
        'betweenness': {'sample_sources': 32, 'seed': SEED, 'weight': 'length_m', 'normalized': True,
                        'top_node': int(top[0]), 'top_value': float(centrality[top[0]])},
        'node_removal': {'removed_nodes': [int(n) for n in blocked], 'results': disruption,
                         'interpretation': 'hypothetical structural disruption, not observed flooding'},
        'flood_link': 'edge exposure from all 153 observed flood reports aggregated by kelurahan; no predicted depth used',
        'source_sha256': sha256(edges_path), 'elapsed_seconds': time.perf_counter() - start,
        'limitations': ['2020 flood vs 2026 roads', 'bounding box can fragment the graph',
                       'OSM vertices include road-shape points', 'turn restrictions not modelled',
                       'kelurahan exposure is not road inundation', 'centrality is sampled, not exact',
                       'known exposure can still cover only part of a segment']}
    save_json(OUT / 'graph_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['node_removal', 'limitations']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
