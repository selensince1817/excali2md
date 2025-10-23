# excali2md

Lightweight CLI to convert an **Excalidraw** `.excalidraw` JSON into node/edge CSV and simple markdown.

## Install
```bash
pip install pandas

Quick run

python excali2md.py -i path/to/diagram.excalidraw -o path/to/graph.csv

Creates: graph.csv (nodes), graph_edges.csv, graph_clean.csv, graph_clean.md, graph_doc.md.

Cluster rule
	•	Make a dashed rectangle for a cluster (strokeStyle == "dashed").
	•	Put cluster name as red text (#e03131) inside it.

Notes
	•	Nodes = type=="rectangle". Texts map via containerId or geometry.
	•	Arrows map via bindings, else nearest rect border (search_radius ≈ 120).
	•	Edge labels: prefer containerId==arrow.id, then boundElements, then nearest text (dist ≤ 140).

Requirements
	•	Python 3.8+
	•	pandas

If you want CLI flags for thresholds or JSON output — say so and I’ll add them.

