#!/usr/bin/env python3
"""
Database Reset Script for SubgraphRAG+

This script clears all databases and indexes to provide a fresh start for testing:
- Neo4j database (clears all nodes and relationships)
- FAISS vector index
- SQLite staging database
- Cache directories

Usage:
    python scripts/reset_dbs.py
    python scripts/reset_dbs.py --confirm
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# RULE:import-rich-logger-correctly
from src.app.log import logger
from src.app.config import config, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

console = Console()

# RULE:uppercase-constants-top
FAISS_INDEX_PATH = getattr(config, 'FAISS_INDEX_PATH', 'data/faiss_index.bin')
SQLITE_DB_PATH = "data/subgraph.db"
STAGING_DB_PATH = "data/staging.db"
CACHE_DIRS = [
    "cache",
    "logs",
    "data/cache",
    "models/cache"
]

def reset_neo4j() -> Dict[str, Any]:
    """Reset Neo4j database by deleting all nodes and relationships"""
    console.print("\n[cyan]🗄️ Resetting Neo4j Database...[/cyan]")
    
    try:
        from neo4j import GraphDatabase
        
        # For neo4j+s:// URIs, encryption is already specified in the URI scheme
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Get initial counts
            initial_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            initial_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            console.print(f"  [yellow]• Found {initial_nodes:,} nodes and {initial_rels:,} relationships[/yellow]")
            
            if initial_nodes > 0 or initial_rels > 0:
                console.print("  [yellow]🧹 Deleting all relationships...[/yellow]")
                session.run("MATCH ()-[r]->() DELETE r")
                
                console.print("  [yellow]🧹 Deleting all nodes...[/yellow]")
                session.run("MATCH (n) DELETE n")
                
                # Verify deletion
                final_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                final_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                console.print(f"  [green]✅ Database cleared: {final_nodes} nodes, {final_rels} relationships remaining[/green]")
            else:
                console.print("  [green]✅ Database was already empty[/green]")
        
        driver.close()
        
        return {
            'status': True,
            'initial_nodes': initial_nodes,
            'initial_relationships': initial_rels,
            'message': 'Neo4j database reset successfully'
        }
        
    except Exception as e:
        console.print(f"  [red]❌ Failed to reset Neo4j: {e}[/red]")
        return {
            'status': False,
            'error': str(e),
            'message': 'Failed to reset Neo4j database'
        }

def reset_faiss_index() -> Dict[str, Any]:
    """Reset FAISS vector index"""
    console.print("\n[cyan]🧠 Resetting FAISS Vector Index...[/cyan]")
    
    try:
        faiss_path = Path(FAISS_INDEX_PATH)
        
        if faiss_path.exists():
            # Get info about existing index
            try:
                import faiss
                index = faiss.read_index(str(faiss_path))
                vector_count = index.ntotal
                console.print(f"  [yellow]• Found existing index with {vector_count:,} vectors[/yellow]")
            except Exception:
                console.print("  [yellow]• Found existing index file (unable to read details)[/yellow]")
            
            # Remove the index file
            faiss_path.unlink()
            console.print(f"  [green]✅ Removed FAISS index: {faiss_path}[/green]")
            
            # Also remove ID map if it exists
            id_map_path = faiss_path.with_suffix('.id_map')
            if id_map_path.exists():
                id_map_path.unlink()
                console.print(f"  [green]✅ Removed ID map: {id_map_path}[/green]")
                
            return {
                'status': True,
                'message': 'FAISS index reset successfully',
                'removed_vectors': vector_count if 'vector_count' in locals() else 'unknown'
            }
        else:
            console.print("  [green]✅ No FAISS index found - already clean[/green]")
            return {
                'status': True,
                'message': 'No FAISS index to reset'
            }
            
    except Exception as e:
        console.print(f"  [red]❌ Failed to reset FAISS index: {e}[/red]")
        return {
            'status': False,
            'error': str(e),
            'message': 'Failed to reset FAISS index'
        }

def reset_sqlite_databases() -> Dict[str, Any]:
    """Reset SQLite databases"""
    console.print("\n[cyan]💾 Resetting SQLite Databases...[/cyan]")
    
    results = []
    
    for db_path in [SQLITE_DB_PATH, STAGING_DB_PATH]:
        db_file = Path(db_path)
        
        if db_file.exists():
            # Get file size for info
            size_mb = db_file.stat().st_size / (1024 * 1024)
            console.print(f"  [yellow]• Found database: {db_path} ({size_mb:.2f} MB)[/yellow]")
            
            try:
                db_file.unlink()
                console.print(f"  [green]✅ Removed: {db_path}[/green]")
                results.append({'path': db_path, 'status': True, 'size_mb': size_mb})
            except Exception as e:
                console.print(f"  [red]❌ Failed to remove {db_path}: {e}[/red]")
                results.append({'path': db_path, 'status': False, 'error': str(e)})
        else:
            console.print(f"  [green]✅ No database found at {db_path} - already clean[/green]")
            results.append({'path': db_path, 'status': True, 'note': 'already clean'})
    
    success_count = sum(1 for r in results if r['status'])
    
    return {
        'status': success_count == len(results),
        'results': results,
        'message': f'Reset {success_count}/{len(results)} SQLite databases'
    }

def reset_cache_directories() -> Dict[str, Any]:
    """Reset cache directories"""
    console.print("\n[cyan]🗂️ Resetting Cache Directories...[/cyan]")
    
    results = []
    
    for cache_dir in CACHE_DIRS:
        cache_path = Path(cache_dir)
        
        if cache_path.exists() and cache_path.is_dir():
            try:
                # Count files for info
                file_count = sum(1 for _ in cache_path.rglob('*') if _.is_file())
                console.print(f"  [yellow]• Found cache directory: {cache_dir} ({file_count} files)[/yellow]")
                
                # Remove all contents
                shutil.rmtree(cache_path)
                
                # Recreate empty directory
                cache_path.mkdir(parents=True, exist_ok=True)
                
                console.print(f"  [green]✅ Cleared cache directory: {cache_dir}[/green]")
                results.append({'path': cache_dir, 'status': True, 'files_removed': file_count})
                
            except Exception as e:
                console.print(f"  [red]❌ Failed to clear {cache_dir}: {e}[/red]")
                results.append({'path': cache_dir, 'status': False, 'error': str(e)})
        else:
            console.print(f"  [green]✅ No cache directory at {cache_dir} - creating empty one[/green]")
            cache_path.mkdir(parents=True, exist_ok=True)
            results.append({'path': cache_dir, 'status': True, 'note': 'created empty'})
    
    success_count = sum(1 for r in results if r['status'])
    
    return {
        'status': success_count == len(results),
        'results': results,
        'message': f'Reset {success_count}/{len(results)} cache directories'
    }

def generate_reset_report(results: Dict[str, Any]):
    """Generate a comprehensive reset report"""
    console.print("\n")
    console.print("=" * 80)
    
    # Summary table
    summary_table = Table(title="🔄 Database Reset Summary", show_header=True, header_style="bold blue")
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Details", style="dim")
    
    for component, result in results.items():
        if result['status']:
            status = "✅ SUCCESS"
            style = "green"
        else:
            status = "❌ FAILED"
            style = "red"
        
        details = result.get('message', '')
        if 'error' in result:
            details = f"Error: {result['error']}"
        
        summary_table.add_row(component.replace('_', ' ').title(), status, details)
    
    console.print(summary_table)
    
    # Overall status
    all_success = all(result['status'] for result in results.values())
    
    if all_success:
        status_text = "🎯 ALL DATABASES RESET SUCCESSFULLY"
        status_style = "bold green"
        border_style = "green"
    else:
        status_text = "⚠️ SOME RESETS FAILED"
        status_style = "bold yellow"
        border_style = "yellow"
    
    console.print("\n")
    final_panel = Panel(
        f"{status_text}\n\n"
        "All databases and indexes have been cleared.\n"
        "The system is ready for fresh testing.",
        title="Reset Complete",
        border_style=border_style
    )
    console.print(final_panel)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Reset all SubgraphRAG+ databases and indexes")
    parser.add_argument("--confirm", action="store_true", 
                       help="Skip confirmation prompt and proceed directly")
    parser.add_argument("--neo4j-only", action="store_true",
                       help="Reset only Neo4j database")
    parser.add_argument("--faiss-only", action="store_true",
                       help="Reset only FAISS index")
    parser.add_argument("--sqlite-only", action="store_true",
                       help="Reset only SQLite databases")
    parser.add_argument("--cache-only", action="store_true",
                       help="Reset only cache directories")
    
    args = parser.parse_args()
    
    logger.info(f"Started database reset at {logger}")
    
    # Display header
    console.print(Panel(
        "SubgraphRAG+ Database Reset Tool\n\n"
        "⚠️  WARNING: This will permanently delete all data in:\n"
        "• Neo4j knowledge graph (all nodes and relationships)\n"
        "• FAISS vector index (all embeddings)\n"
        "• SQLite databases (staging and main)\n"
        "• Cache directories\n\n"
        "🔄 This ensures a completely fresh start for testing.\n"
        "💾 Make sure you have backups if needed!",
        title="🗄️ Database Reset",
        border_style="red"
    ))
    
    # Confirmation
    if not args.confirm:
        if not Confirm.ask("\n[bold red]Are you sure you want to reset ALL databases?[/bold red]"):
            console.print("[yellow]Reset cancelled by user.[/yellow]")
            return
    
    console.print("\n[bold green]🚀 Starting database reset...[/bold green]")
    
    # Determine what to reset
    reset_all = not any([args.neo4j_only, args.faiss_only, args.sqlite_only, args.cache_only])
    
    results = {}
    
    # Reset components
    if reset_all or args.neo4j_only:
        results['neo4j'] = reset_neo4j()
    
    if reset_all or args.faiss_only:
        results['faiss'] = reset_faiss_index()
    
    if reset_all or args.sqlite_only:
        results['sqlite'] = reset_sqlite_databases()
    
    if reset_all or args.cache_only:
        results['cache'] = reset_cache_directories()
    
    # Generate report
    generate_reset_report(results)
    
    # Log completion
    logger.info("Finished database reset")
    
    # Exit code
    all_success = all(result['status'] for result in results.values())
    sys.exit(0 if all_success else 1)

if __name__ == "__main__":
    main() 