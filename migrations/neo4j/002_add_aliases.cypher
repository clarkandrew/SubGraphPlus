// Add aliases support to Entity nodes - Version kg_v2
// Adds aliases array property and full-text search index

// Update existing entities to have an empty aliases array if it doesn't exist
MATCH (e:Entity)
WHERE NOT exists(e.aliases)
SET e.aliases = [];

// Create full-text search index on name and aliases
CALL db.index.fulltext.createNodeIndex(
  "entityNamesAndAliases",
  ["Entity"],
  ["name", "aliases"],
  {analyzer: "english"}
);

// Create schema info node for this version
MERGE (s:_SchemaVersion {version: "kg_v2"})
SET s.appliedAt = timestamp(),
    s.description = "Added aliases array property to Entity nodes and full-text search index";

// Sample comment: This migration adds support for entity aliases
// - Entity nodes now have an aliases array property for alternative names
// - Full-text search index enables efficient text search across both name and aliases
// - Existing entities are updated with empty aliases arrays for consistency