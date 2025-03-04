import math
from typing import List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch
import numpy as np

class CurvedEdge(FancyArrowPatch):
    def __init__(self, posA, posB, rad=.1, **kwargs):
        """
        Draw a curved arrow between two points with a specified radius.

        :param posA: Starting position of the arrow (x, y).
        :param posB: Ending position of the arrow (x, y).
        :param rad: Curvature radius of the arrow.
        :param kwargs: Additional keyword arguments for arrow properties.
        """
        # Calculate the mid-point between posA and posB for curvature
        self.posA = posA
        self.posB = posB
        super().__init__(posA, posB, connectionstyle=f"arc3,rad={rad}", **kwargs)

def draw_custom_dependency_graph(
        sent_words: List[str],
        edges: List[Tuple[int, int, float]],
        node_values: List[float],
        node_spacing=4,
        cmap='Reds',
        min_weight_for_coloring=0.2
):
    """
    Draws a directed graph in a single row layout with curvy edges and edge thickness proportional to weights.

    :param edges: List of tuples representing edges in the form (node_idx1, node_idx2, weight).
    :param node_labels: Optional list of labels for nodes. If not provided, node indices will be used.
    """
    # print(sent_words)
    # print(edges)
    print("hello")
    print(min_weight_for_coloring)

    # Create a directed graph
    G = nx.DiGraph()

    # Determine the number of nodes
    num_nodes = len(sent_words)

    # Add nodes explicitly in the desired canonical order
    # otherwise color labeling does not work
    for i in range(len(sent_words)):
        G.add_node(i)

    # Add edges to the graph with weights
    for node1, node2, weight in edges:
        if node1 > num_nodes or node2 > num_nodes:
            raise Exception("invalid graph")
        # todo this seems to have no effect - see below (maybe we separatley are drawing edges)
        w = max(weight, min_weight_for_coloring)
        print(w)
        G.add_edge(node1, node2, weight=w)

    # Create positions for the nodes in a single row
    pos = {i: (i * 4, 0) for i in range(num_nodes)}

    # pos = {i: (2*i, 0) for i in range(num_nodes)}

    norm = plt.Normalize(vmin=0, vmax=1)
    node_colors = plt.cm.get_cmap(cmap)(norm(node_values))
    # node_colors = plt.cm.get_cmap(cmap)(node_values)
    # print(node_values)
    # print(norm(node_values))
    # print(node_colors)
    # Draw the nodes with optional labels
    nx.draw_networkx_nodes(G, pos, node_size=50,
                           # node_color="lightblue"
                           node_color=node_colors,
                           )

    # node_labels = {i: w for i, w in enumerate(sent_words)}

    # Draw the labels for nodes
    # nx.draw_networkx_labels(G, pos,
    #                         # labels=node_labels,
    #                         # font_size=12, font_color="black"
    #                         )

    max_weight = max(edges, key=lambda x: x[2])[2] / 2

    # Draw the curvy edges
    ax = plt.gca()
    for node, (x, y) in pos.items():
        # print(node_values[node])
        ax.text(
            x, y - 0.25 * math.sqrt(num_nodes),  # Adjust label position slightly above the node
            # x, y,  # Adjust label position slightly above the node
            sent_words[node] + f" {round(node_values[node], 2)}",
            fontsize=8,
            ha='right',
            va='center',
            rotation=45,  # Rotate label to 45 degrees
            rotation_mode='anchor'  # Keep the rotation anchored at the position
        )
    for node1, node2, weight in edges:
        # print(f'adding edge bw {node1} {node2} for weight {weight}')
        # print(f'adding edge bw {node1}: {sent_words[node1]} ::  {node2}: '
        #       f'{sent_words[node2]} w weight {weight}')
        posA = pos[node1]
        posB = pos[node2]
        # Adjust curvature radius based on the distance between nodes to avoid overlap
        rad = 0.2 + 0.1 * math.pow(num_nodes, 1/3) * math.pow(np.abs(node1 - node2), 1/3)
        # rad = 0.5
        w = max(weight, min_weight_for_coloring)
        edge = CurvedEdge(posA, posB,
                          rad=rad, color="gray",
                          linewidth=w/max_weight,
                          # linewidth=1,
                          arrowstyle='-|>', mutation_scale=15)
        ax.add_patch(edge)

    # add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, ax=ax, orientation='vertical', label='HHI node score')


    # Display the plot
    # plt.title("Custom Dependency Graph with Curvy Edges")
    plt.axis('off')  # Hide axes for clarity
    plt.xlim(0, num_nodes*node_spacing)
    # plt.subplots_adjust(left=1, right=1)#, top=0.9, bottom=0.1)
    # plt.tight_layout(pad=5.0)
    plt.show()

def make_affinity_graph(
        score_tensor: torch.Tensor,
        multi_tok_indices: List[int],
        sent_word_list: List[str],
        node_values,
        omit_adjacent = False
):
    """
    Draw an "affinity graph" which is basically a dependency tree but with
    edges between highest affinity words
    """
    # print(multi_tok_indices)
    # we transpose first so that we can row-index the columns
    top_vals, top_idxs = torch.topk(score_tensor,
                                    k=score_tensor.shape[0],
                                    dim=0)
    top_vals = top_vals.T.numpy()
    top_idxs = top_idxs.T.numpy()
    # pp(top_idxs)
    # pp(top_vals)
    # make edges
    edges: List[Tuple[int, int, float]] = []
    for idx in range(score_tensor.shape[0]):
        # if the start is multitok, do nothing
        if idx in multi_tok_indices:
            continue
        # note that the i is used to index into the values
        for i, neighbor_idx in enumerate(top_idxs[idx]):
            # todo: experiment - don't allow immediately adjacent
            if omit_adjacent and abs(neighbor_idx-idx) <= 1:
                continue
            if neighbor_idx in multi_tok_indices:
                continue
            # edge from the neighbor to this one, for influence
            edges.append((neighbor_idx, idx, top_vals[idx][i]))
            # edges.append((idx, neighbor_idx, top_vals[idx][i]))
            break
    # print(edges)
    draw_custom_dependency_graph(sent_word_list, edges, node_values)
    # if the end is multitok, then get the next one
