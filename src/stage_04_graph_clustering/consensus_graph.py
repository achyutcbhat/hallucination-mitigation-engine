import networkx as nx
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.agent import AgentCandidateResponse
from src.schemas.graph import (
    GraphNode,
    GraphEdge,
    SemanticCluster,
    ClusteringResult,
)

logger = setup_logger("ConsensusGraph")


def build_consensus_graph(
    candidates: list[AgentCandidateResponse],
    sim_matrix: np.ndarray,
    threshold: float = settings.GRAPH_SIMILARITY_THRESHOLD,
) -> ClusteringResult:
    """Builds an undirected semantic graph, finds connected consensus components,

    and extracts the primary consensus cluster while pruning outlier nodes.
    """
    G = nx.Graph()
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    n = len(candidates)

    # 1. Add Nodes
    for c in candidates:
        snippet = (
            (c.response_text[:80] + "...")
            if len(c.response_text) > 80
            else c.response_text
        )
        node = GraphNode(
            node_id=c.candidate_id,
            candidate_id=c.candidate_id,
            agent_model=c.agent_model,
            text_snippet=snippet,
        )
        nodes.append(node)
        G.add_node(
            c.candidate_id,
            model=c.agent_model,
            text=c.response_text,
        )

    # 2. Add Edges based on Semantic Similarity Threshold
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score >= threshold:
                source_id = candidates[i].candidate_id
                target_id = candidates[j].candidate_id
                edges.append(
                    GraphEdge(
                        source=source_id,
                        target=target_id,
                        similarity_score=round(score, 3),
                    )
                )
                G.add_edge(source_id, target_id, weight=score)

    logger.info(
        f"Consensus Graph built: {len(nodes)} nodes, {len(edges)} consensus edges (Threshold={threshold})"
    )

    # 3. Detect Connected Components / Consensus Clusters
    connected_components = list(nx.connected_components(G))

    # Fallback: If no edges were formed (isolated responses), lower threshold adaptively
    if len(edges) == 0 and n > 1:
        logger.warning(
            "Graph is fully disconnected at threshold. Applying adaptive fallback (threshold=0.60)..."
        )
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i][j])
                if score >= 0.60:
                    edges.append(
                        GraphEdge(
                            source=candidates[i].candidate_id,
                            target=candidates[j].candidate_id,
                            similarity_score=round(score, 3),
                        )
                    )
                    G.add_edge(
                        candidates[i].candidate_id,
                        candidates[j].candidate_id,
                        weight=score,
                    )
        connected_components = list(nx.connected_components(G))

    # 4. Score and Rank Clusters
    clusters: list[SemanticCluster] = []
    cluster_id_counter = 1

    for comp in connected_components:
        member_ids = sorted(list(comp))
        size = len(member_ids)

        # Calculate internal density / average similarity
        if size > 1:
            internal_sims = []
            for m1 in member_ids:
                for m2 in member_ids:
                    if m1 < m2 and G.has_edge(m1, m2):
                        internal_sims.append(G[m1][m2]["weight"])
            avg_sim = (
                float(np.mean(internal_sims)) if internal_sims else threshold
            )
        else:
            avg_sim = 1.0  # Self-similarity for isolated node

        clusters.append(
            SemanticCluster(
                cluster_id=cluster_id_counter,
                member_node_ids=member_ids,
                size=size,
                avg_internal_similarity=round(avg_sim, 3),
                is_primary_consensus=False,
            )
        )
        cluster_id_counter += 1

    # 5. Determine Primary Consensus Cluster (Sort by size DESC, then avg_internal_similarity DESC)
    clusters.sort(
        key=lambda c: (c.size, c.avg_internal_similarity), reverse=True
    )
    primary_cluster = clusters[0]
    primary_cluster.is_primary_consensus = True

    # 6. Identify Pruned / Outlier Candidates
    primary_member_set = set(primary_cluster.member_node_ids)
    pruned_ids = [
        c.candidate_id
        for c in candidates
        if c.candidate_id not in primary_member_set
    ]

    logger.info(
        f"Primary Consensus Cluster identified: Cluster #{primary_cluster.cluster_id} "
        f"({primary_cluster.size} members, AvgSim={primary_cluster.avg_internal_similarity}). "
        f"Pruned Outliers={pruned_ids}"
    )

    return ClusteringResult(
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        primary_cluster=primary_cluster,
        pruned_candidate_ids=pruned_ids,
    )