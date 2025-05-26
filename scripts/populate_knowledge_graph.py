#!/usr/bin/env python3
"""
Populate Neo4j with comprehensive knowledge graph data for SubgraphRAG+ MVP

This script creates a rich knowledge graph with entities and relationships
to enable proper FAISS training and system functionality.

Following project rules:
- RULE:import-rich-logger-correctly ✅
- RULE:debug-trace-every-step ✅
- RULE:rich-error-handling-required ✅
"""

import os
import sys
import uuid
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

from app.database import neo4j_db
from app.entity_typing import detect_entity_type

# Initialize rich console for pretty CLI output
console = Console()

# Constants
KNOWLEDGE_DOMAINS = {
    "science": [
        ("Albert Einstein", "developed", "Theory of Relativity"),
        ("Albert Einstein", "born_in", "Germany"),
        ("Albert Einstein", "won", "Nobel Prize in Physics"),
        ("Isaac Newton", "discovered", "Law of Gravitation"),
        ("Isaac Newton", "formulated", "Laws of Motion"),
        ("Isaac Newton", "born_in", "England"),
        ("Marie Curie", "discovered", "Radium"),
        ("Marie Curie", "won", "Nobel Prize in Physics"),
        ("Marie Curie", "won", "Nobel Prize in Chemistry"),
        ("Charles Darwin", "wrote", "Origin of Species"),
        ("Charles Darwin", "developed", "Theory of Evolution"),
        ("Nikola Tesla", "invented", "AC Motor"),
        ("Nikola Tesla", "born_in", "Serbia"),
        ("Theory of Relativity", "explains", "Space-Time"),
        ("Theory of Relativity", "predicts", "Time Dilation"),
        ("DNA", "contains", "Genetic Information"),
        ("DNA", "discovered_by", "Watson and Crick"),
        ("Quantum Mechanics", "describes", "Atomic Behavior"),
        ("Quantum Mechanics", "developed_by", "Max Planck"),
    ],
    "technology": [
        ("Python", "created_by", "Guido van Rossum"),
        ("Python", "is_a", "Programming Language"),
        ("Neo4j", "is_a", "Graph Database"),
        ("Neo4j", "stores", "Nodes and Relationships"),
        ("FAISS", "developed_by", "Facebook AI Research"),
        ("FAISS", "is_a", "Vector Search Library"),
        ("Machine Learning", "subset_of", "Artificial Intelligence"),
        ("Deep Learning", "subset_of", "Machine Learning"),
        ("Neural Networks", "inspired_by", "Human Brain"),
        ("Transformer", "is_a", "Neural Network Architecture"),
        ("GPT", "based_on", "Transformer"),
        ("BERT", "based_on", "Transformer"),
        ("Knowledge Graph", "stores", "Structured Information"),
        ("Knowledge Graph", "enables", "Semantic Search"),
        ("SubgraphRAG", "uses", "Knowledge Graph"),
        ("SubgraphRAG", "combines", "Retrieval and Generation"),
        ("Vector Database", "stores", "Embeddings"),
        ("Embeddings", "represent", "Semantic Meaning"),
    ],
    "geography": [
        ("Germany", "is_a", "Country"),
        ("Germany", "located_in", "Europe"),
        ("England", "is_a", "Country"),
        ("England", "part_of", "United Kingdom"),
        ("Serbia", "is_a", "Country"),
        ("Serbia", "located_in", "Balkans"),
        ("Europe", "is_a", "Continent"),
        ("Asia", "is_a", "Continent"),
        ("North America", "is_a", "Continent"),
        ("United States", "located_in", "North America"),
        ("California", "part_of", "United States"),
        ("Silicon Valley", "located_in", "California"),
        ("Stanford University", "located_in", "California"),
        ("MIT", "located_in", "Massachusetts"),
        ("Massachusetts", "part_of", "United States"),
    ],
    "organizations": [
        ("Facebook AI Research", "part_of", "Meta"),
        ("Meta", "is_a", "Technology Company"),
        ("Google", "is_a", "Technology Company"),
        ("Microsoft", "is_a", "Technology Company"),
        ("OpenAI", "is_a", "AI Research Company"),
        ("Stanford University", "is_a", "University"),
        ("MIT", "is_a", "University"),
        ("Nobel Prize", "awarded_by", "Nobel Committee"),
        ("Nobel Committee", "located_in", "Sweden"),
        ("IEEE", "is_a", "Professional Organization"),
        ("ACM", "is_a", "Professional Organization"),
    ],
    "concepts": [
        ("Artificial Intelligence", "aims_to", "Simulate Human Intelligence"),
        ("Machine Learning", "learns_from", "Data"),
        ("Supervised Learning", "uses", "Labeled Data"),
        ("Unsupervised Learning", "finds", "Hidden Patterns"),
        ("Reinforcement Learning", "learns_through", "Trial and Error"),
        ("Natural Language Processing", "processes", "Human Language"),
        ("Computer Vision", "processes", "Visual Information"),
        ("Robotics", "combines", "AI and Engineering"),
        ("Data Science", "extracts", "Insights from Data"),
        ("Big Data", "characterized_by", "Volume Velocity Variety"),
        ("Cloud Computing", "provides", "On-demand Resources"),
        ("Distributed Systems", "span", "Multiple Computers"),
    ]
}

def create_entity_with_type(tx, name: str, entity_type: str = None) -> str:
    """Create an entity with proper typing"""
    # RULE:debug-trace-every-step
    logger.debug(f"Starting create_entity_with_type for: {name}")
    
    if not entity_type:
        logger.debug(f"Detecting entity type for: {name}")
        entity_type = detect_entity_type(name)
        logger.debug(f"Detected type: {entity_type}")
    
    entity_id = str(uuid.uuid4())
    logger.debug(f"Generated entity ID: {entity_id}")
    
    try:
        result = tx.run("""
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.id = $entity_id, e.type = $entity_type
        ON MATCH SET e.type = COALESCE(e.type, $entity_type)
        RETURN e.id as id
        """, name=name, entity_id=entity_id, entity_type=entity_type)
        
        record = result.single()
        final_id = record["id"] if record else entity_id
        logger.debug(f"Finished create_entity_with_type for: {name}")
        return final_id
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Failed to create entity {name}: {e}")
        raise

def create_relationship(tx, head_name: str, relation_name: str, tail_name: str) -> str:
    """Create a relationship between entities"""
    logger.debug(f"Creating relationship: {head_name} -{relation_name}-> {tail_name}")
    
    # Get or create entities
    head_id = create_entity_with_type(tx, head_name)
    tail_id = create_entity_with_type(tx, tail_name)
    
    # Create relationship
    rel_id = str(uuid.uuid4())
    
    result = tx.run("""
    MATCH (h:Entity {name: $head_name})
    MATCH (t:Entity {name: $tail_name})
    MERGE (h)-[r:REL {name: $relation_name}]->(t)
    ON CREATE SET r.id = $rel_id
    RETURN r.id as id
    """, head_name=head_name, tail_name=tail_name, relation_name=relation_name, rel_id=rel_id)
    
    record = result.single()
    return record["id"] if record else rel_id

def populate_domain(domain_name: str, triples: list) -> int:
    """Populate a knowledge domain with triples"""
    logger.info(f"Starting population of {domain_name} domain with {len(triples)} triples")
    
    created_count = 0
    
    for head, relation, tail in triples:
        try:
            def _tx_function(tx):
                return create_relationship(tx, head, relation, tail)
            
            rel_id = neo4j_db.run_transaction(_tx_function)
            if rel_id:
                created_count += 1
                logger.debug(f"Created relationship: {head} -{relation}-> {tail}")
            
        except Exception as e:
            logger.error(f"Failed to create relationship {head} -{relation}-> {tail}: {e}")
    
    logger.info(f"Finished populating {domain_name} domain: {created_count}/{len(triples)} relationships created")
    return created_count

def verify_population() -> dict:
    """Verify the knowledge graph population"""
    logger.info("Verifying knowledge graph population")
    
    try:
        # Count entities
        entity_result = neo4j_db.run_query("MATCH (e:Entity) RETURN count(e) as count")
        entity_count = entity_result[0]["count"] if entity_result else 0
        
        # Count relationships
        rel_result = neo4j_db.run_query("MATCH ()-[r:REL]->() RETURN count(r) as count")
        rel_count = rel_result[0]["count"] if rel_result else 0
        
        # Sample entities by type
        type_result = neo4j_db.run_query("""
        MATCH (e:Entity) 
        WHERE e.type IS NOT NULL 
        RETURN e.type as type, count(*) as count 
        ORDER BY count DESC 
        LIMIT 10
        """)
        
        # Sample relationships
        rel_sample = neo4j_db.run_query("""
        MATCH (h)-[r:REL]->(t) 
        RETURN h.name as head, r.name as relation, t.name as tail 
        LIMIT 10
        """)
        
        stats = {
            "entities": entity_count,
            "relationships": rel_count,
            "entity_types": {r["type"]: r["count"] for r in type_result} if type_result else {},
            "sample_relationships": [
                f"{r['head']} -{r['relation']}-> {r['tail']}" 
                for r in rel_sample
            ] if rel_sample else []
        }
        
        logger.info(f"Knowledge graph stats: {entity_count} entities, {rel_count} relationships")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to verify population: {e}")
        return {"error": str(e)}

def main():
    """Main population function"""
    # RULE:debug-trace-every-step
    logger.debug("Starting main() function")
    
    # RULE:every-src-script-must-log
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Started {__file__} at {timestamp}")
    
    console.print("🚀 [bold green]Starting comprehensive knowledge graph population[/bold green]")
    
    try:
        logger.debug("Verifying Neo4j connectivity...")
        
        # Verify Neo4j connection
        if not neo4j_db.verify_connectivity():
            # RULE:rich-error-handling-required
            logger.error("Cannot connect to Neo4j database")
            console.print("❌ [bold red]Cannot connect to Neo4j database[/bold red]")
            return 1
        
        console.print("✅ [green]Neo4j connection verified[/green]")
        logger.info("Neo4j connection verified")
        
        total_created = 0
        
        # Populate each domain
        console.print("📊 [cyan]Populating knowledge domains...[/cyan]")
        for domain_name, triples in KNOWLEDGE_DOMAINS.items():
            logger.debug(f"Processing domain: {domain_name}")
            console.print(f"  🔄 [yellow]Processing {domain_name} ({len(triples)} triples)[/yellow]")
            
            created = populate_domain(domain_name, triples)
            total_created += created
            
            console.print(f"  ✅ [green]{domain_name}: {created}/{len(triples)} relationships created[/green]")
        
        # Verify population
        logger.debug("Verifying population results...")
        console.print("🔍 [cyan]Verifying population results...[/cyan]")
        stats = verify_population()
        
        if "error" in stats:
            # RULE:rich-error-handling-required
            logger.error(f"Population verification failed: {stats['error']}")
            console.print(f"❌ [bold red]Population verification failed: {stats['error']}[/bold red]")
            return 1
        
        # Success output with rich formatting
        logger.info("Knowledge graph population completed successfully")
        logger.info(f"Total relationships created: {total_created}")
        logger.info(f"Final stats: {stats['entities']} entities, {stats['relationships']} relationships")
        
        console.print("🎉 [bold green]Knowledge graph population completed successfully![/bold green]")
        console.print(f"📈 [cyan]Total relationships created: {total_created}[/cyan]")
        console.print(f"📊 [cyan]Final stats: {stats['entities']} entities, {stats['relationships']} relationships[/cyan]")
        
        # Log sample data for verification
        if stats["sample_relationships"]:
            logger.info("Sample relationships:")
            console.print("🔗 [yellow]Sample relationships:[/yellow]")
            for rel in stats["sample_relationships"][:5]:
                logger.info(f"  {rel}")
                console.print(f"  • [dim]{rel}[/dim]")
        
        if stats["entity_types"]:
            logger.info("Entity type distribution:")
            console.print("🏷️ [yellow]Entity type distribution:[/yellow]")
            for entity_type, count in list(stats["entity_types"].items())[:5]:
                logger.info(f"  {entity_type}: {count}")
                console.print(f"  • [dim]{entity_type}: {count}[/dim]")
        
        logger.info("Knowledge graph is now ready for FAISS training and system functionality")
        console.print("🚀 [bold green]Knowledge graph is now ready for FAISS training and system functionality![/bold green]")
        
        logger.debug("Finished main() function successfully")
        return 0
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Knowledge graph population failed: {e}")
        console.print_exception()
        logger.debug("Finished main() function with error")
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"Finished {__file__} with exit code {exit_code}")
    sys.exit(exit_code) 