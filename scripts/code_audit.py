#!/usr/bin/env python3
"""
SubgraphRAG+ Code Audit Script
Systematically identifies architectural flaws and deviations from specifications
"""

import os
import re
import ast
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AuditIssue:
    """Represents a code audit issue"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # ARCHITECTURE, PERFORMANCE, SECURITY, MAINTAINABILITY
    file_path: str
    line_number: int
    issue_type: str
    description: str
    recommendation: str
    code_snippet: str = ""

class SubgraphRAGAuditor:
    """
    Comprehensive auditor for SubgraphRAG+ codebase
    Identifies deviations from paper specifications and architectural issues
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[AuditIssue] = []
        
        # Patterns that indicate potential issues
        self.anti_patterns = {
            "string_heuristics": [
                r'if.*\.lower\(\).*in.*\[.*".*"\]',  # String matching heuristics
                r'any\(.*in.*\.lower\(\).*for.*in.*\[',  # Pattern matching loops
                r'\.startswith\(.*\).*or.*\.endswith\(',  # Prefix/suffix matching
            ],
            "hardcoded_values": [
                r'["\'][^"\']*\.(inc|corp|llc|ltd|dr|mr|mrs)["\']',  # Hardcoded entity patterns
                r'["\'][^"\']*\.(city|town|country|state)["\']',  # Hardcoded location patterns
                r'[0-9]+\s*#.*magic.*number',  # Magic numbers with comments
            ],
            "naive_implementations": [
                r'def.*detect.*type.*\(.*\):',  # Type detection functions
                r'def.*simple.*\(.*\):.*heuristic',  # Simple heuristic functions
                r'# TODO.*production.*sophisticated',  # TODO comments about production
                r'# In.*production.*would.*be.*more',  # Production disclaimers
            ],
            "missing_error_handling": [
                r'\.run_query\(.*\)(?!\s*try)',  # Database queries without try/catch
                r'requests\.(get|post)\(.*\)(?!\s*try)',  # HTTP requests without error handling
                r'json\.loads\(.*\)(?!\s*try)',  # JSON parsing without error handling
            ],
            "performance_issues": [
                r'for.*in.*range\(.*\):.*query',  # Loops with database queries
                r'time\.sleep\([0-9]+\)',  # Hardcoded sleep statements
                r'\.cache_clear\(\)',  # Cache clearing (potential performance issue)
            ],
            "security_issues": [
                r'eval\(',  # Use of eval
                r'exec\(',  # Use of exec
                r'shell=True',  # Shell injection risk
                r'trust_remote_code=True',  # Untrusted code execution
            ]
        }
        
        # SubgraphRAG paper compliance checks
        self.paper_compliance = {
            "entity_typing": {
                "required": "Schema-driven entity typing from knowledge graph",
                "anti_pattern": r'def.*detect.*type.*\(.*text.*\):',
                "description": "Original paper uses KG schema types, not string heuristics"
            },
            "triple_extraction": {
                "required": "Proper relation extraction model (REBEL, OpenIE, etc.)",
                "anti_pattern": r'\.split\(.*\).*triple',
                "description": "Paper assumes structured triples, not naive text splitting"
            },
            "mlp_integration": {
                "required": "Use pretrained SubgraphRAG MLP model",
                "anti_pattern": r'class.*MLP.*\(.*nn\.Module.*\):.*def.*__init__',
                "description": "Should load pretrained model, not define new architecture"
            },
            "embedding_consistency": {
                "required": "Consistent embedding model across pipeline",
                "anti_pattern": r'embed.*=.*different.*model',
                "description": "All components must use same embedding model"
            }
        }
    
    def audit_file(self, file_path: Path) -> List[AuditIssue]:
        """Audit a single Python file"""
        if not file_path.suffix == '.py':
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []
        
        file_issues = []
        
        # Check for anti-patterns
        for category, patterns in self.anti_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issue = AuditIssue(
                            severity="HIGH" if category in ["string_heuristics", "security_issues"] else "MEDIUM",
                            category="ARCHITECTURE" if category == "string_heuristics" else "PERFORMANCE",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            issue_type=category,
                            description=f"Detected {category.replace('_', ' ')}: {line.strip()}",
                            recommendation=self._get_recommendation(category),
                            code_snippet=line.strip()
                        )
                        file_issues.append(issue)
        
        # Check paper compliance
        for compliance_type, spec in self.paper_compliance.items():
            pattern = spec["anti_pattern"]
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issue = AuditIssue(
                        severity="CRITICAL",
                        category="ARCHITECTURE",
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        issue_type=f"paper_deviation_{compliance_type}",
                        description=f"Deviation from SubgraphRAG paper: {spec['description']}",
                        recommendation=f"Required: {spec['required']}",
                        code_snippet=line.strip()
                    )
                    file_issues.append(issue)
        
        # AST-based analysis for more complex patterns
        try:
            tree = ast.parse(content)
            file_issues.extend(self._analyze_ast(tree, file_path, lines))
        except SyntaxError:
            logger.warning(f"Could not parse AST for {file_path}")
        
        return file_issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[AuditIssue]:
        """Analyze AST for complex patterns"""
        issues = []
        project_root = self.project_root  # Capture for inner class
        
        class AuditVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for functions that should use external services
                if 'extract' in node.name.lower() and 'triple' in node.name.lower():
                    # Look for naive implementations
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and hasattr(child.func, 'attr'):
                            if child.func.attr in ['split', 'replace', 'strip']:
                                issues.append(AuditIssue(
                                    severity="HIGH",
                                    category="ARCHITECTURE", 
                                    file_path=str(file_path.relative_to(project_root)),
                                    line_number=node.lineno,
                                    issue_type="naive_triple_extraction",
                                    description=f"Function {node.name} uses naive string operations for triple extraction",
                                    recommendation="Use proper IE model (REBEL, OpenIE) for triple extraction",
                                    code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else ""
                                ))
                
                # Check for hardcoded model definitions
                if 'mlp' in node.name.lower() or 'model' in node.name.lower():
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and hasattr(child.func, 'attr'):
                            if child.func.attr in ['Linear', 'Sequential', 'ReLU']:
                                issues.append(AuditIssue(
                                    severity="CRITICAL",
                                    category="ARCHITECTURE",
                                    file_path=str(file_path.relative_to(project_root)),
                                    line_number=node.lineno,
                                    issue_type="custom_mlp_definition",
                                    description=f"Function {node.name} defines custom MLP instead of loading pretrained",
                                    recommendation="Load pretrained SubgraphRAG MLP model from checkpoint",
                                    code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else ""
                                ))
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for custom MLP classes
                if 'mlp' in node.name.lower() and any(isinstance(base, ast.Name) and 'Module' in base.id for base in node.bases):
                    issues.append(AuditIssue(
                        severity="CRITICAL",
                        category="ARCHITECTURE",
                        file_path=str(file_path.relative_to(project_root)),
                        line_number=node.lineno,
                        issue_type="custom_mlp_class",
                        description=f"Class {node.name} defines custom MLP architecture",
                        recommendation="Use pretrained SubgraphRAG MLP model instead of custom implementation",
                        code_snippet=lines[node.lineno-1].strip() if node.lineno <= len(lines) else ""
                    ))
                
                self.generic_visit(node)
        
        visitor = AuditVisitor()
        visitor.visit(tree)
        return issues
    
    def _get_recommendation(self, category: str) -> str:
        """Get recommendation for fixing issue category"""
        recommendations = {
            "string_heuristics": "Replace with schema-driven approach using external type mappings",
            "hardcoded_values": "Move to configuration files or external data sources",
            "naive_implementations": "Implement production-grade solution using established libraries/models",
            "missing_error_handling": "Add proper try/catch blocks with logging and graceful degradation",
            "performance_issues": "Optimize with batching, caching, or async operations",
            "security_issues": "Remove unsafe operations and use secure alternatives"
        }
        return recommendations.get(category, "Review and improve implementation")
    
    def audit_project(self) -> Dict[str, Any]:
        """Audit entire project"""
        logger.info(f"Starting audit of {self.project_root}")
        
        # Find all Python files
        python_files = []
        for pattern in ['src/**/*.py', 'scripts/**/*.py', 'tests/**/*.py', 'app/**/*.py']:
            python_files.extend(self.project_root.glob(pattern))
        
        logger.info(f"Found {len(python_files)} Python files to audit")
        
        # Audit each file
        for file_path in python_files:
            file_issues = self.audit_file(file_path)
            self.issues.extend(file_issues)
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info(f"Audit complete: found {len(self.issues)} issues")
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate audit summary"""
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        file_counts = defaultdict(int)
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            category_counts[issue.category] += 1
            file_counts[issue.file_path] += 1
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue.issue_type].append(issue)
        
        return {
            "total_issues": len(self.issues),
            "severity_breakdown": dict(severity_counts),
            "category_breakdown": dict(category_counts),
            "files_with_issues": len(file_counts),
            "most_problematic_files": sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "issues_by_type": {k: len(v) for k, v in issues_by_type.items()},
            "critical_issues": [issue for issue in self.issues if issue.severity == "CRITICAL"],
            "high_priority_issues": [issue for issue in self.issues if issue.severity in ["CRITICAL", "HIGH"]]
        }
    
    def generate_report(self, output_path: str = "audit_report.json"):
        """Generate detailed audit report"""
        summary = self._generate_summary()
        
        # Convert issues to dictionaries for JSON serialization
        issues_dict = []
        for issue in self.issues:
            issues_dict.append({
                "severity": issue.severity,
                "category": issue.category,
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "issue_type": issue.issue_type,
                "description": issue.description,
                "recommendation": issue.recommendation,
                "code_snippet": issue.code_snippet
            })
        
        report = {
            "audit_timestamp": str(Path().cwd()),
            "summary": summary,
            "issues": issues_dict
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Audit report saved to {output_path}")
        return report
    
    def print_summary(self):
        """Print audit summary to console"""
        summary = self._generate_summary()
        
        print("\n" + "="*60)
        print("SUBGRAPHRAG+ CODE AUDIT SUMMARY")
        print("="*60)
        
        print(f"\nTotal Issues Found: {summary['total_issues']}")
        
        print(f"\nSeverity Breakdown:")
        for severity, count in sorted(summary['severity_breakdown'].items()):
            print(f"  {severity}: {count}")
        
        print(f"\nCategory Breakdown:")
        for category, count in sorted(summary['category_breakdown'].items()):
            print(f"  {category}: {count}")
        
        print(f"\nTop Issue Types:")
        for issue_type, count in sorted(summary['issues_by_type'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {issue_type}: {count}")
        
        print(f"\nMost Problematic Files:")
        for file_path, count in summary['most_problematic_files'][:5]:
            print(f"  {file_path}: {count} issues")
        
        if summary['critical_issues']:
            print(f"\nCRITICAL ISSUES ({len(summary['critical_issues'])}):")
            for issue in summary['critical_issues'][:3]:  # Show first 3
                print(f"  {issue.file_path}:{issue.line_number} - {issue.description}")
        
        print("\n" + "="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit SubgraphRAG+ codebase for architectural issues")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default="audit_report.json", help="Output report file")
    parser.add_argument("--summary-only", action="store_true", help="Print summary only")
    
    args = parser.parse_args()
    
    auditor = SubgraphRAGAuditor(args.project_root)
    auditor.audit_project()
    
    if args.summary_only:
        auditor.print_summary()
    else:
        auditor.generate_report(args.output)
        auditor.print_summary()


if __name__ == "__main__":
    main() 