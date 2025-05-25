from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
import uuid
import time


@dataclass
class Triple:
    """A triple in the knowledge graph: (head, relation, tail)"""
    id: str  # Unique identifier
    head_id: str  # Head entity ID
    head_name: str  # Head entity name
    relation_id: str  # Relation ID
    relation_name: str  # Relation name
    tail_id: str  # Tail entity ID
    tail_name: str  # Tail entity name
    properties: Optional[Dict[str, Any]] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    relevance_score: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary, excluding embedding"""
        result = {
            "id": self.id,
            "head_id": self.head_id,
            "head_name": self.head_name,
            "relation_id": self.relation_id,
            "relation_name": self.relation_name,
            "tail_id": self.tail_id,
            "tail_name": self.tail_name,
            "properties": self.properties,
        }
        if self.relevance_score is not None:
            result["relevance_score"] = float(self.relevance_score)
        return result

    def to_string(self) -> str:
        """Convert to string representation"""
        return f"{self.head_name} {self.relation_name} {self.tail_name}"
    
    def __hash__(self):
        """Hash based on the triple ID"""
        return hash(self.id)


@dataclass
class Entity:
    """An entity in the knowledge graph"""
    id: str  # Unique identifier
    name: str  # Name
    type: Optional[str] = None  # Entity type
    properties: Dict[str, Any] = field(default_factory=dict)  # Additional properties
    aliases: List[str] = field(default_factory=list)  # Alternative names
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "aliases": self.aliases,
        }


@dataclass
class GraphNode:
    """Node for D3.js visualization"""
    id: str
    name: str
    type: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relevance_score: Optional[float] = None
    inclusion_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        result = {
            "id": self.id,
            "name": self.name,
            "properties": self.properties,
        }
        if self.type:
            result["type"] = self.type
        if self.relevance_score is not None:
            result["relevance_score"] = float(self.relevance_score)
        if self.inclusion_reasons:
            result["inclusion_reasons"] = self.inclusion_reasons
        return result


@dataclass
class GraphLink:
    """Link for D3.js visualization"""
    source: str
    target: str
    relation_id: str
    relation_name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relevance_score: Optional[float] = None
    inclusion_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        result = {
            "source": self.source,
            "target": self.target,
            "relation_id": self.relation_id,
            "relation_name": self.relation_name,
            "properties": self.properties,
        }
        if self.relevance_score is not None:
            result["relevance_score"] = float(self.relevance_score)
        if self.inclusion_reasons:
            result["inclusion_reasons"] = self.inclusion_reasons
        return result


@dataclass
class GraphData:
    """Full graph data for visualization"""
    nodes: List[GraphNode] = field(default_factory=list)
    links: List[GraphLink] = field(default_factory=list)
    relevant_paths: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "links": [link.to_dict() for link in self.links],
            "relevant_paths": self.relevant_paths,
        }


@dataclass
class PaginatedGraphData(GraphData):
    """Paginated graph data for full KG browsing"""
    page: int = 1
    limit: int = 500
    total_nodes_in_filter: int = 0
    total_links_in_filter: int = 0
    has_more: bool = False
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        result = super().to_dict()
        result.update({
            "page": self.page,
            "limit": self.limit,
            "total_nodes_in_filter": self.total_nodes_in_filter,
            "total_links_in_filter": self.total_links_in_filter,
            "has_more": self.has_more,
        })
        return result


@dataclass
class QueryRequest:
    """Request model for /query endpoint"""
    question: str
    visualize_graph: bool = True


@dataclass
class FeedbackRequest:
    """Request model for /feedback endpoint"""
    query_id: str
    is_correct: bool
    comment: Optional[str] = None
    expected_answer: Optional[str] = None


@dataclass
class IngestRequest:
    """Request model for /ingest endpoint"""
    triples: List[Dict[str, str]]


@dataclass
class ErrorResponse:
    """Standard error response model"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class TripleNotFoundError(Exception):
    """Raised when a triple is not found"""
    pass


class RetrievalEmpty(Exception):
    """Raised when retrieval returns no results"""
    pass


class EntityLinkingError(Exception):
    """Raised when entity linking fails"""
    pass


class AmbiguousEntityError(Exception):
    """Raised when entity linking is ambiguous"""
    def __init__(self, message, candidates):
        super().__init__(message)
        self.candidates = candidates


def generate_query_id():
    """Generate a unique query ID"""
    return f"q_{uuid.uuid4().hex[:8]}_{int(time.time())}"