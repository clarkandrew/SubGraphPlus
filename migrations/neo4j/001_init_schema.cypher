// Initialize SubgraphRAG+ schema - Version kg_v1
// Creates basic Entity and REL structure with constraints and indexes

// Create constraints on Entity
CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE;

// Set up indexes for better performance
CREATE INDEX ON :Entity(name);

CREATE INDEX ON :Entity(type);

// Create index on relationships
CREATE INDEX ON :REL(name);

CREATE INDEX ON :REL(id);

// Create basic schema info node
MERGE (s:_SchemaVersion {version: "kg_v1"})
SET s.appliedAt = timestamp(), 
    s.description = "Initial schema with Entity nodes and REL relationships";

// Sample comment: This initial schema sets up the base structure needed for SubgraphRAG+
// - Entity nodes have {id, name, type} properties
// - REL relationships have {id, name} properties
// - Schema versioning is tracked via _SchemaVersion nodes