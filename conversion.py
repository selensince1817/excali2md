import argparse
import json
import os
from math import hypot
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ast
from typing import Any


DEFAULT_INPUT_PATH = "/Users/selen/Desktop/projects/excali2md/pre-4.excalidraw"
DEFAULT_EXPORT_PATH = "/Users/selen/Desktop/projects/excali2md/graph.csv"
RED = "#e03131"


def rect_bbox(rect: Dict) -> Tuple[float, float, float, float]:
    x1 = float(rect.get("x", 0.0))
    y1 = float(rect.get("y", 0.0))
    x2 = x1 + float(rect.get("width", 0.0))
    y2 = y1 + float(rect.get("height", 0.0))
    return x1, y1, x2, y2


def point_in_rect(px: float, py: float, rect: Dict) -> bool:
    x1, y1, x2, y2 = rect_bbox(rect)
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def text_center(text: Dict) -> Tuple[float, float]:
    cx = float(text.get("x", 0.0)) + float(text.get("width", 0.0)) / 2.0
    cy = float(text.get("y", 0.0)) + float(text.get("height", 0.0)) / 2.0
    return cx, cy


def arrow_endpoints_world(arrow: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = arrow.get("points", [])
    ax = float(arrow.get("x", 0.0))
    ay = float(arrow.get("y", 0.0))
    if not pts:
        return (ax, ay), (ax, ay)
    start = (ax + float(pts[0][0]), ay + float(pts[0][1]))
    end = (ax + float(pts[-1][0]), ay + float(pts[-1][1]))
    return start, end


def point_rect_border_distance(px: float, py: float, rect: Dict) -> float:
    x1, y1, x2, y2 = rect_bbox(rect)
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    if dx == 0.0 and dy == 0.0:
        return min(px - x1, x2 - px, py - y1, y2 - py)
    return (dx * dx + dy * dy) ** 0.5


def map_arrow_to_rects(arrow: Dict, rects: List[Dict], search_radius: float = 120.0) -> Tuple[Optional[str], Optional[str]]:
    (sx, sy), (ex, ey) = arrow_endpoints_world(arrow)
    best_start = (float("inf"), None)
    best_end = (float("inf"), None)
    for r in rects:
        ds = point_rect_border_distance(sx, sy, r)
        de = point_rect_border_distance(ex, ey, r)
        if ds < best_start[0]:
            best_start = (ds, r["id"])  # type: ignore[index]
        if de < best_end[0]:
            best_end = (de, r["id"])  # type: ignore[index]
    start_id = best_start[1] if best_start[0] <= search_radius else None
    end_id = best_end[1] if best_end[0] <= search_radius else None
    return start_id, end_id


def point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projx, projy = x1 + t * dx, y1 + t * dy
    return hypot(px - projx, py - projy)


def build_text_rect_maps(
    texts: List[Dict], rects: List[Dict], id_to_element: Dict[str, Dict]
) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]]]:
    text_id_to_rect_id: Dict[str, Optional[str]] = {}
    rect_id_to_text_ids: Dict[str, List[str]] = {r["id"]: [] for r in rects}  # type: ignore[index]
    for t in texts:
        assigned_rect_id: Optional[str] = None
        container_id = t.get("containerId")
        if container_id and id_to_element.get(container_id, {}).get("type") == "rectangle":
            assigned_rect_id = container_id
        else:
            cx, cy = text_center(t)
            candidates: List[Tuple[float, str]] = []
            for r in rects:
                if point_in_rect(cx, cy, r):
                    x1, y1, x2, y2 = rect_bbox(r)
                    area = (x2 - x1) * (y2 - y1)
                    candidates.append((area, r["id"]))  # type: ignore[index]
            if candidates:
                candidates.sort()
                assigned_rect_id = candidates[0][1]
        text_id_to_rect_id[t["id"]] = assigned_rect_id  # type: ignore[index]
        if assigned_rect_id:
            rect_id_to_text_ids[assigned_rect_id].append(t["id"])  # type: ignore[index]
    return text_id_to_rect_id, rect_id_to_text_ids


def detect_clusters(
    rects: List[Dict], texts: List[Dict], id_to_element: Dict[str, Dict]
) -> Tuple[List[str], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    cluster_rect_ids = [r["id"] for r in rects if r.get("strokeStyle") == "dashed"]  # type: ignore[index]
    rect_id_to_cluster_label: Dict[str, Optional[str]] = {}
    rect_id_to_cluster_rect: Dict[str, Optional[str]] = {}

    # Find label for each cluster rect: prefer red text whose containerId == cluster rect id; else geometry
    for cr_id in cluster_rect_ids:
        label_text_id: Optional[str] = None
        cands = [t for t in texts if t.get("strokeColor") == RED and t.get("containerId") == cr_id]
        if cands:
            # choose largest font size
            cands.sort(key=lambda t: (t.get("fontSize") or 0), reverse=True)
            label_text_id = cands[0]["id"]  # type: ignore[index]
        else:
            cr = id_to_element[cr_id]
            for t in texts:
                if t.get("strokeColor") == RED:
                    cx, cy = text_center(t)
                    if point_in_rect(cx, cy, cr):
                        label_text_id = t["id"]  # type: ignore[index]
                        break
        rect_id_to_cluster_label[cr_id] = label_text_id

    # Membership: rect center inside one of the cluster rects
    for r in rects:
        cx = float(r.get("x", 0.0)) + float(r.get("width", 0.0)) / 2.0
        cy = float(r.get("y", 0.0)) + float(r.get("height", 0.0)) / 2.0
        member: Optional[str] = None
        for cr_id in cluster_rect_ids:
            if point_in_rect(cx, cy, id_to_element[cr_id]):
                member = cr_id
                break
        rect_id_to_cluster_rect[r["id"]] = member  # type: ignore[index]

    return cluster_rect_ids, rect_id_to_cluster_label, rect_id_to_cluster_rect


def map_arrow_with_bindings(
    arrow: Dict, id_to_element: Dict[str, Dict], rects: List[Dict]
) -> Tuple[Optional[str], Optional[str]]:
    start_id = arrow.get("startBinding", {}).get("elementId") if arrow.get("startBinding") else None
    end_id = arrow.get("endBinding", {}).get("elementId") if arrow.get("endBinding") else None
    if start_id and id_to_element.get(start_id, {}).get("type") != "rectangle":
        start_id = None
    if end_id and id_to_element.get(end_id, {}).get("type") != "rectangle":
        end_id = None
    if start_id or end_id:
        return start_id, end_id
    return map_arrow_to_rects(arrow, rects)


def pick_label_candidate(candidates: List[Dict]) -> Optional[Dict]:
    if not candidates:
        return None
    candidates_sorted = sorted(candidates, key=lambda t: (t.get("fontSize") or 0), reverse=True)
    return candidates_sorted[0]


def build_nodes_edges(
    all_elements: List[Dict],
    rects: List[Dict],
    texts: List[Dict],
    arrows: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Optional[Dict]], List[str], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    id_to_element: Dict[str, Dict] = {e["id"]: e for e in all_elements}  # type: ignore[index]

    # Map texts to rects (containerId preferred; fallback geometry)
    text_to_rect, rect_to_texts = build_text_rect_maps(texts, rects, id_to_element)

    # Clusters
    cluster_rect_ids, rect_id_to_cluster_label, rect_id_to_cluster_rect = detect_clusters(
        rects, texts, id_to_element
    )

    # Primary text for each rect (avoid red label for cluster rects)
    rect_id_to_primary_text: Dict[str, Optional[Dict]] = {}
    for r in rects:
        text_ids = rect_to_texts.get(r["id"], [])  # type: ignore[index]
        candidates: List[Dict] = []
        for t_id in text_ids:
            t = id_to_element[t_id]
            if r.get("strokeStyle") == "dashed" and t.get("strokeColor") == RED:
                continue
            candidates.append(t)
        primary: Optional[Dict] = None
        if candidates:
            candidates.sort(key=lambda t: (t.get("fontSize") or 0), reverse=True)
            primary = candidates[0]
        rect_id_to_primary_text[r["id"]] = primary  # type: ignore[index]

    # Build nodes dataframe
    nodes: List[Dict] = []
    for r in rects:
        rid = r["id"]  # type: ignore[index]
        pt = rect_id_to_primary_text.get(rid)
        nodes.append(
            {
                "rect_id": rid,
                "rect_is_cluster": rid in cluster_rect_ids,
                "text_id": (pt["id"] if pt else None),
                "text": (pt.get("text") if pt else None),
                "cluster_rect_id": rect_id_to_cluster_rect.get(rid),
                "cluster_label_text_id": (
                    rect_id_to_cluster_label.get(rect_id_to_cluster_rect.get(rid))
                    if rect_id_to_cluster_rect.get(rid)
                    else None
                ),
            }
        )
    nodes_df = pd.DataFrame(nodes)

    # Preindex texts by containerId to catch labels bound to arrows
    texts_by_container: Dict[str, List[Dict]] = {}
    for t in texts:
        cid = t.get("containerId")
        if cid:
            texts_by_container.setdefault(cid, []).append(t)

    # Build edges with labels: prefer containerId==arrow.id; then arrow.boundElements; fall back to proximity
    edges: List[Dict] = []
    for a in arrows:
        src_id, dst_id = map_arrow_with_bindings(a, id_to_element, rects)

        (sx, sy), (ex, ey) = arrow_endpoints_world(a)

        label_text_obj: Optional[Dict] = None

        # 1) containerId → arrow.id (exclude red)
        bound_text_candidates = [
            t for t in texts_by_container.get(a["id"], [])  # type: ignore[index]
            if t.get("strokeColor") != RED
        ]
        picked = pick_label_candidate(bound_text_candidates)
        if picked:
            label_text_obj = picked
        else:
            # 2) arrow.boundElements → text ids
            be_ids: List[str] = []
            for be in (a.get("boundElements") or []):
                if be.get("type") == "text" and be.get("id"):
                    be_ids.append(be["id"])  # type: ignore[index]
            be_candidates = [
                id_to_element[tid]
                for tid in be_ids
                if tid in id_to_element
                and id_to_element[tid].get("type") == "text"
                and id_to_element[tid].get("strokeColor") != RED
            ]
            picked = pick_label_candidate(be_candidates)
            if picked:
                label_text_obj = picked
            else:
                # 3) proximity fallback (exclude node texts and cluster titles)
                best = (float("inf"), None)
                for t in texts:
                    if t.get("strokeColor") == RED:
                        continue
                    t_rect = text_to_rect.get(t["id"])  # type: ignore[index]
                    if t_rect in (src_id, dst_id):
                        continue
                    cx, cy = text_center(t)
                    d = point_segment_distance(cx, cy, sx, sy, ex, ey)
                    if d < best[0]:
                        best = (d, t)
                if best[1] is not None and best[0] <= 140.0:
                    label_text_obj = best[1]

        edges.append(
            {
                "arrow_id": a["id"],  # type: ignore[index]
                "src_rect_id": src_id,
                "dst_rect_id": dst_id,
                "edge_label_text_id": (label_text_obj["id"] if label_text_obj else None),  # type: ignore[index]
                "edge_label_text": (label_text_obj.get("text") if label_text_obj else None),
            }
        )

    # Build edges_df
    edge_rows: List[Dict] = []
    for e in edges:
        src_r = e["src_rect_id"]
        dst_r = e["dst_rect_id"]
        if not src_r or not dst_r:
            continue
        src_t = rect_id_to_primary_text.get(src_r)
        dst_t = rect_id_to_primary_text.get(dst_r)
        edge_rows.append(
            {
                "arrow_id": e["arrow_id"],
                "src_rect_id": src_r,
                "src_text_id": (src_t["id"] if src_t else None),  # type: ignore[index]
                "src_text": (src_t.get("text") if src_t else None),
                "dst_rect_id": dst_r,
                "dst_text_id": (dst_t["id"] if dst_t else None),  # type: ignore[index]
                "dst_text": (dst_t.get("text") if dst_t else None),
                "edge_label_text_id": e["edge_label_text_id"],
                "edge_label_text": e["edge_label_text"],
            }
        )
    edges_df = pd.DataFrame(edge_rows)

    # Build graph_df
    cluster_rect_to_label_text: Dict[str, Optional[str]] = {
        cr_id: (id_to_element[label_id]["text"] if label_id else None)  # type: ignore[index]
        for cr_id, label_id in detect_clusters(rects, texts, id_to_element)[1].items()
    }
    node_cluster_text: Dict[str, Optional[str]] = {}
    for n in nodes:
        cr = n["cluster_rect_id"]
        node_cluster_text[n["rect_id"]] = cluster_rect_to_label_text.get(cr) if cr else None

    graph_rows: List[Dict] = []
    for e in edge_rows:
        graph_rows.append(
            {
                "from_rect_id": e["src_rect_id"],
                "from_text_id": e["src_text_id"],
                "from_text": e["src_text"],
                "from_cluster_label": node_cluster_text.get(e["src_rect_id"]),
                "to_rect_id": e["dst_rect_id"],
                "to_text_id": e["dst_text_id"],
                "to_text": e["dst_text"],
                "to_cluster_label": node_cluster_text.get(e["dst_rect_id"]),
                "edge_label_text": e["edge_label_text"],
                "edge_label_text_id": e["edge_label_text_id"],
                "arrow_id": e["arrow_id"],
            }
        )
    graph_df = pd.DataFrame(graph_rows)

    return (
        nodes_df,
        edges_df,
        graph_df,
        rect_id_to_primary_text,
        cluster_rect_ids,
        rect_id_to_cluster_label,
        rect_id_to_cluster_rect,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Excalidraw to graph CSV")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_PATH, help="Path to .excalidraw JSON file")
    parser.add_argument("--out", "-o", default=DEFAULT_EXPORT_PATH, help="Path to output CSV file")
    args = parser.parse_args()

    file_path = args.input
    export_path = args.out

    assert os.path.exists(file_path), f"File not found: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_elements: List[Dict] = data.get("elements", [])
    rects: List[Dict] = [e for e in all_elements if e.get("type") == "rectangle"]
    texts: List[Dict] = [e for e in all_elements if e.get("type") == "text"]
    arrows: List[Dict] = [e for e in all_elements if e.get("type") == "arrow"]

    (
        nodes_df,
        edges_df,
        graph_df,
        rect_id_to_primary_text,
        cluster_rect_ids,
        rect_id_to_cluster_label,
        rect_id_to_cluster_rect,
    ) = build_nodes_edges(all_elements, rects, texts, arrows)

    print(f"Nodes: {len(nodes_df)} | Edges: {len(edges_df)} | Rows: {len(graph_df)}")
    print(
        f"Edges with labels: {len(edges_df[~edges_df['edge_label_text'].isna()])} / {len(edges_df)}"
    )

    # Build node-centric aggregated table (one row per rectangle)
    # Outgoing aggregates
    outgoing = (
        edges_df.groupby("src_rect_id").agg(
            outgoing_to_rect_ids=("dst_rect_id", list),
            outgoing_to_texts=("dst_text", list),
            outgoing_edge_labels=("edge_label_text", list),
        )
    )
    # Incoming aggregates
    incoming = (
        edges_df.groupby("dst_rect_id").agg(
            incoming_from_rect_ids=("src_rect_id", list),
            incoming_from_texts=("src_text", list),
            incoming_edge_labels=("edge_label_text", list),
        )
    )

    # Merge into nodes_df (rect_id key)
    nodes_agg = nodes_df.copy()
    nodes_agg = nodes_agg.merge(
        outgoing, how="left", left_on="rect_id", right_index=True
    ).merge(
        incoming, how="left", left_on="rect_id", right_index=True
    )

    # Add human cluster label text
    id_to_element: Dict[str, Dict] = {e["id"]: e for e in all_elements}  # type: ignore[index]
    cluster_rect_to_label_text: Dict[str, Optional[str]] = {
        cr_id: (id_to_element[label_id]["text"] if label_id else None)  # type: ignore[index]
        for cr_id, label_id in rect_id_to_cluster_label.items()
    }
    nodes_agg["cluster_label"] = nodes_agg["cluster_rect_id"].map(
        lambda cr: cluster_rect_to_label_text.get(cr) if cr else None
    )

    # Ensure list columns are lists, not NaN
    for col in [
        "outgoing_to_rect_ids",
        "outgoing_to_texts",
        "outgoing_edge_labels",
        "incoming_from_rect_ids",
        "incoming_from_texts",
        "incoming_edge_labels",
    ]:
        if col in nodes_agg.columns:
            nodes_agg[col] = nodes_agg[col].apply(lambda v: v if isinstance(v, list) else [])

    # Export: default export_path will be the node-centric table (54+ rows)
    # Also export edges separately alongside it.
    base, ext = os.path.splitext(export_path)
    edges_path = f"{base}_edges{ext or '.csv'}"
    nodes_path = export_path

    os.makedirs(os.path.dirname(nodes_path) or ".", exist_ok=True)
    nodes_agg.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    print(f"Exported nodes: {nodes_path}")
    print(f"Exported edges: {edges_path}")

    # Build cleaned, user-friendly nodes table mirroring notebook df_clean
    drop_cols = [
        "rect_id",
        "rect_is_cluster",
        "text_id",
        "cluster_rect_id",
        "cluster_label_text_id",
        "outgoing_to_rect_ids",
        "incoming_from_rect_ids",
    ]
    rename_cols = {
        "text": "node_text",
        "cluster_label": "cluster",
        "outgoing_to_texts": "points_to",
        "outgoing_edge_labels": "outgoing_labels",
        "incoming_from_texts": "pointed_by",
        "incoming_edge_labels": "incoming_labels",
    }
    nodes_final = nodes_agg.drop(columns=drop_cols, errors="ignore").rename(columns=rename_cols)

    # Ensure list-like columns are lists (when reloaded from csv they may be strings)
    def _to_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    return parsed if isinstance(parsed, list) else []
                except Exception:
                    return []
            if s == "" or s.lower() == "nan":
                return []
            return [s]
        return []

    for col in ["points_to", "outgoing_labels", "pointed_by", "incoming_labels"]:
        if col in nodes_final.columns:
            nodes_final[col] = nodes_final[col].apply(_to_list)

    # Save cleaned CSV
    clean_path = f"{base}_clean{ext or '.csv'}"
    nodes_final.to_csv(clean_path, index=False)
    print(f"Exported cleaned nodes: {clean_path}")

    # Save markdown table for cleaned nodes
    try:
        md_table_path = f"{base}_clean.md"
        md_table = nodes_final.to_markdown(index=False)
        with open(md_table_path, "w", encoding="utf-8") as f:
            f.write(md_table)
        print(f"Exported markdown table: {md_table_path}")
    except Exception:
        pass

    # Save narrative markdown document grouped by cluster (not a table)
    doc_path = f"{base.replace('_edges', '')}_doc.md"
    df_doc = nodes_final.copy()
    if "cluster" in df_doc.columns:
        df_doc["cluster"] = df_doc["cluster"].fillna("Unclustered")
    lines: List[str] = []
    lines.append("# Graph Document")
    clusters = sorted(df_doc["cluster"].dropna().unique().tolist()) if "cluster" in df_doc.columns else ["All"]
    for cluster_name in clusters:
        lines.append(f"\n## Cluster: {cluster_name}")
        group_df = df_doc[df_doc["cluster"] == cluster_name] if "cluster" in df_doc.columns else df_doc
        for _, row in group_df.iterrows():
            node_text = str(row.get("node_text") or "").strip()
            if not node_text:
                continue
            lines.append(f"\n### {node_text}")
            # Outgoing
            pts = row.get("points_to") or []
            olabs = row.get("outgoing_labels") or []
            if pts:
                lines.append("\n- Outgoing:")
                for i, tgt in enumerate(pts):
                    lab = olabs[i] if i < len(olabs) else None
                    if lab and str(lab).strip():
                        lines.append(f"  - → {tgt} (label: {lab})")
                    else:
                        lines.append(f"  - → {tgt}")
            # Incoming
            srcs = row.get("pointed_by") or []
            ilabs = row.get("incoming_labels") or []
            if srcs:
                lines.append("\n- Incoming:")
                for i, src in enumerate(srcs):
                    lab = ilabs[i] if i < len(ilabs) else None
                    if lab and str(lab).strip():
                        lines.append(f"  - ← {src} (label: {lab})")
                    else:
                        lines.append(f"  - ← {src}")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Exported narrative markdown: {doc_path}")


if __name__ == "__main__":
    main()


