// Initialize SubgraphRAG+ schema - Version kg_v1
// Creates basic Entity and REL structure with constraints and indexes

// Create constraints on Entity
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS 
ON (e:Entity) 
ASSERT e.id IS UNIQUE;

// Set up indexes for better performance
CREATE INDEX entity_name_idx IF NOT EXISTS
FOR (e:Entity)
ON (e.name);

CREATE INDEX entity_type_idx IF NOT EXISTS
FOR (e:Entity)
ON (e.type);

// Create index on relationships
CREATE INDEX relation_name_idx IF NOT EXISTS
FOR ()-[r:REL]->()
ON (r.name);

CREATE INDEX relation_id_idx IF NOT EXISTS
FOR ()-[r:REL]->()
ON (r.id);

// Create basic schema info node
MERGE (s:_SchemaVersion {version: "kg_v1"})
SET s.appliedAt = timestamp(), 
    s.description = "Initial schema with Entity nodes and REL relationships";

// Sample comment: This initial schema sets up the base structure needed for SubgraphRAG+
// - Entity nodes have {id, name, type} properties
// - REL relationships have {id, name} properties
// - Schema versioning is tracked via _SchemaVersion nodes