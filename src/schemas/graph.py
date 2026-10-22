from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    node_id: int
    candidate_id: int
    agent_model: str
    text_snippet: str


class GraphEdge(BaseModel):
    source: int
    target: int
    similarity_score: float


class SemanticCluster(BaseModel):
    cluster_id: int
    member_node_ids: list[int]
    size: int
    avg_internal_similarity: float
    is_primary_consensus: bool = False


class ClusteringResult(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    clusters: list[SemanticCluster]
    primary_cluster: SemanticCluster
    pruned_candidate_ids: list[int] = Field(
        default_factory=list, description="Outlier candidate IDs filtered out"
    )