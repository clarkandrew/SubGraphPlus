#!/usr/bin/env python3
"""
Fix Logging Rule Violations

This script systematically fixes all violations of the project logging rules:
- RULE:import-rich-logger-correctly
- RULE:debug-trace-every-step  
- RULE:rich-error-handling-required
- RULE:every-src-script-must-log
- RULE:no-print-in-source

Following project rules:
- RULE:import-rich-logger-correctly ✅
- RULE:debug-trace-every-step ✅
- RULE:rich-error-handling-required ✅
"""

import os
import sys
import re
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# RULE:import-rich-logger-correctly - Use centralized rich logger
from src.app.log import logger, log_and_print
from rich.console import Console

# Initialize rich console for pretty CLI output
console = Console()

def fix_file_logging(file_path: Path) -> bool:
    """Fix logging violations in a single file"""
    logger.debug(f"Starting fix_file_logging for: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if file is in scripts/ or src/app/ or src/scripts/
        should_fix = (
            'scripts/' in str(file_path) or 
            'src/app/' in str(file_path) or 
            'src/scripts/' in str(file_path)
        )
        
        if not should_fix:
            logger.debug(f"Skipping {file_path} - not in target directories")
            return True
        
        # Skip if already using centralized logger
        if 'from src.app.log import logger' in content:
            logger.debug(f"Skipping {file_path} - already uses centralized logger")
            return True
        
        # Fix 1: Replace logger creation with centralized import
        if 'logger = logging.getLogger(__name__)' in content:
            logger.info(f"Fixing logger import in {file_path}")
            
            # Remove logging setup
            content = re.sub(
                r'import logging\n',
                '',
                content
            )
            
            content = re.sub(
                r'# Set up logging.*?logger = logging\.getLogger\(__name__\)',
                '# RULE:import-rich-logger-correctly - Use centralized rich logger\nfrom src.app.log import logger, log_and_print\nfrom rich.console import Console\n\n# Initialize rich console for pretty CLI output\nconsole = Console()',
                content,
                flags=re.DOTALL
            )
            
            content = re.sub(
                r'logging\.basicConfig\(.*?\)\nlogger = logging\.getLogger\(__name__\)',
                '# RULE:import-rich-logger-correctly - Use centralized rich logger\nfrom src.app.log import logger, log_and_print\nfrom rich.console import Console\n\n# Initialize rich console for pretty CLI output\nconsole = Console()',
                content,
                flags=re.DOTALL
            )
        
        # Fix 2: Add script start logging for scripts
        if file_path.name.endswith('.py') and 'scripts/' in str(file_path):
            if 'def main():' in content and 'Started {__file__} at' not in content:
                logger.info(f"Adding script start logging to {file_path}")
                
                # Add timestamp import if needed
                if 'import time' not in content:
                    content = content.replace(
                        'from pathlib import Path',
                        'import time\nfrom pathlib import Path'
                    )
                
                # Add logging to main function
                content = re.sub(
                    r'def main\(\):\s*\n\s*"""([^"]+)"""\s*\n',
                    r'def main():\n    """\1"""\n    # RULE:debug-trace-every-step\n    logger.debug("Starting main() function")\n    \n    # RULE:every-src-script-must-log\n    timestamp = time.strftime(\'%Y-%m-%d %H:%M:%S\')\n    logger.info(f"Started {__file__} at {timestamp}")\n    \n',
                    content
                )
        
        # Fix 3: Replace print statements with rich console
        if 'print(' in content and 'src/' in str(file_path):
            logger.info(f"Replacing print statements in {file_path}")
            
            # Replace simple print statements
            content = re.sub(
                r'print\(([^)]+)\)',
                r'console.print(\1)',
                content
            )
        
        # Fix 4: Add rich error handling to try/except blocks
        if 'except Exception as e:' in content:
            logger.info(f"Adding rich error handling to {file_path}")
            
            # Add console.print_exception() after logger.error
            content = re.sub(
                r'except Exception as e:\s*\n\s*logger\.error\(([^)]+)\)',
                r'except Exception as e:\n        # RULE:rich-error-handling-required\n        logger.error(\1)\n        console.print_exception()',
                content
            )
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed logging violations in {file_path}")
            return True
        else:
            logger.debug(f"No changes needed for {file_path}")
            return True
            
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Failed to fix {file_path}: {e}")
        console.print_exception()
        return False

def find_python_files() -> list:
    """Find all Python files that need fixing"""
    logger.debug("Starting find_python_files")
    
    target_dirs = [
        Path("scripts/"),
        Path("src/app/"),
        Path("src/scripts/"),
        Path("evaluation/")
    ]
    
    python_files = []
    
    for target_dir in target_dirs:
        if target_dir.exists():
            for py_file in target_dir.rglob("*.py"):
                if py_file.is_file():
                    python_files.append(py_file)
    
    logger.info(f"Found {len(python_files)} Python files to check")
    logger.debug("Finished find_python_files")
    return python_files

def main():
    """Main function to fix all logging violations"""
    # RULE:debug-trace-every-step
    logger.debug("Starting main() function")
    
    # RULE:every-src-script-must-log
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Started {__file__} at {timestamp}")
    
    console.print("🔧 [bold green]Starting systematic logging rule violation fixes[/bold green]")
    
    try:
        # Find all Python files
        logger.debug("Finding Python files...")
        python_files = find_python_files()
        
        if not python_files:
            logger.warning("No Python files found to fix")
            console.print("⚠️ [yellow]No Python files found to fix[/yellow]")
            return 0
        
        console.print(f"📁 [cyan]Found {len(python_files)} Python files to check[/cyan]")
        
        # Fix each file
        fixed_count = 0
        error_count = 0
        
        for py_file in python_files:
            logger.debug(f"Processing {py_file}")
            console.print(f"  🔄 [dim]Processing {py_file.name}[/dim]")
            
            if fix_file_logging(py_file):
                fixed_count += 1
            else:
                error_count += 1
        
        # Summary
        logger.info(f"Processed {len(python_files)} files: {fixed_count} successful, {error_count} errors")
        
        if error_count == 0:
            console.print("🎉 [bold green]All logging rule violations fixed successfully![/bold green]")
        else:
            console.print(f"⚠️ [yellow]Fixed {fixed_count} files, {error_count} had errors[/yellow]")
        
        console.print(f"📊 [cyan]Summary: {fixed_count} files processed successfully[/cyan]")
        
        logger.debug("Finished main() function successfully")
        return 0 if error_count == 0 else 1
        
    except Exception as e:
        # RULE:rich-error-handling-required
        logger.error(f"Failed to fix logging violations: {e}")
        console.print_exception()
        logger.debug("Finished main() function with error")
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"Finished {__file__} with exit code {exit_code}")
    sys.exit(exit_code) 