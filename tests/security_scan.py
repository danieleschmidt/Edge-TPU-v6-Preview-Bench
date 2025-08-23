"""
Security Analysis for Quantum Task Planner
Performs security scans and vulnerability assessments
"""

import os
import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Any
import hashlib
import json

class SecurityIssue:
    def __init__(self, severity: str, issue_type: str, description: str, 
                 file_path: str, line_number: int = None):
        self.severity = severity  # CRITICAL, HIGH, MEDIUM, LOW
        self.issue_type = issue_type
        self.description = description
        self.file_path = file_path
        self.line_number = line_number
    
    def __str__(self):
        location = f"{self.file_path}:{self.line_number}" if self.line_number else self.file_path
        return f"[{self.severity}] {self.issue_type}: {self.description} ({location})"

class QuantumSecurityScanner:
    """Security scanner for quantum task planner codebase"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.issues: List[SecurityIssue] = []
        
        # Security patterns to detect
        self.dangerous_patterns = {
            r'eval\s*\(': "Code injection via eval()",
            r'exec\s*\(': "Code injection via exec()",
            r'__import__\s*\(': "Dynamic import potential vulnerability",
            r'subprocess\.call\s*\(': "Subprocess execution without shell=False",
            r'os\.system\s*\(': "OS command execution vulnerability",
            r'pickle\.loads?\s*\(': "Unsafe deserialization with pickle",
            r'input\s*\([^)]*\)': "User input without validation",
            r'raw_input\s*\([^)]*\)': "User input without validation",
        }
        
        # Secrets patterns
        self.secrets_patterns = {
            r'password\s*=\s*["\'][^"\']+["\']': "Hardcoded password",
            r'api_key\s*=\s*["\'][^"\']+["\']': "Hardcoded API key",
            r'secret\s*=\s*["\'][^"\']+["\']': "Hardcoded secret",
            r'token\s*=\s*["\'][^"\']+["\']': "Hardcoded token",
            r'["\'][A-Za-z0-9]{32,}["\']': "Potential hardcoded secret",
        }
        
        # File permission issues
        self.permission_issues = [
            "777",  # World writable
            "666",  # World writable
            "755",  # World executable
        ]
    
    def _is_security_definition(self, line: str, file_path: Path) -> bool:
        """Check if this line is a legitimate security definition"""
        line = line.strip()
        
        # Skip security patterns in security modules
        if 'security' in str(file_path).lower():
            if any(marker in line for marker in [
                'threat_patterns', 'dangerous_patterns', 'security_patterns',
                '# Code injection', '# Code execution', '# Path traversal',
                '"Code injection', '"Code execution', '"Path traversal'
            ]):
                return True
        
        # Skip test files with security test cases
        if 'test' in str(file_path).lower():
            if any(marker in line for marker in [
                'test_', 'Test', 'security_test', 'threat_test',
                '# Test', '"""Test', "'''Test"
            ]):
                return True
        
        # Skip comments and docstrings
        if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
            return True
            
        return False

    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single file for security issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for dangerous patterns
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                # Skip security definitions and test cases
                if self._is_security_definition(line, file_path):
                    continue
                
                # Check dangerous code patterns
                for pattern, description in self.dangerous_patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = "HIGH" if any(x in pattern for x in ['eval', 'exec', 'os.system']) else "MEDIUM"
                        issues.append(SecurityIssue(
                            severity=severity,
                            issue_type="Code Injection",
                            description=description,
                            file_path=str(file_path),
                            line_number=line_num
                        ))
                
                # Check for hardcoded secrets
                for pattern, description in self.secrets_patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip test files and documentation
                        if not any(x in str(file_path) for x in ['test', 'demo', 'example', 'doc']):
                            issues.append(SecurityIssue(
                                severity="CRITICAL",
                                issue_type="Hardcoded Secret",
                                description=description,
                                file_path=str(file_path),
                                line_number=line_num
                            ))
                
                # Check for TODO/FIXME security comments
                if re.search(r'TODO.*security|FIXME.*security|XXX.*security', line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        severity="LOW",
                        issue_type="Security TODO",
                        description="Security-related TODO/FIXME comment",
                        file_path=str(file_path),
                        line_number=line_num
                    ))
            
            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                try:
                    tree = ast.parse(content)
                    issues.extend(self._analyze_ast(tree, file_path))
                except SyntaxError:
                    pass  # Skip files with syntax errors
            
        except Exception as e:
            issues.append(SecurityIssue(
                severity="LOW",
                issue_type="File Access",
                description=f"Could not read file: {e}",
                file_path=str(file_path)
            ))
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path) -> List[SecurityIssue]:
        """Analyze AST for security issues"""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, issues_list):
                self.issues = issues_list
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name in ['eval', 'exec']:
                        self.issues.append(SecurityIssue(
                            severity="CRITICAL",
                            issue_type="Code Injection",
                            description=f"Use of dangerous function: {func_name}",
                            file_path=str(file_path),
                            line_number=node.lineno
                        ))
                    
                    elif func_name == 'input' and len(node.args) == 0:
                        self.issues.append(SecurityIssue(
                            severity="MEDIUM",
                            issue_type="Input Validation",
                            description="User input without prompt or validation",
                            file_path=str(file_path),
                            line_number=node.lineno
                        ))
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous method calls
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'os' and 
                        node.func.attr == 'system'):
                        self.issues.append(SecurityIssue(
                            severity="HIGH",
                            issue_type="Command Injection",
                            description="Use of os.system() - potential command injection",
                            file_path=str(file_path),
                            line_number=node.lineno
                        ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for dangerous imports
                for alias in node.names:
                    if alias.name in ['pickle', 'cPickle']:
                        self.issues.append(SecurityIssue(
                            severity="MEDIUM",
                            issue_type="Unsafe Deserialization",
                            description="Import of pickle module - ensure safe usage",
                            file_path=str(file_path),
                            line_number=node.lineno
                        ))
                
                self.generic_visit(node)
            
            def visit_Str(self, node):
                # Check for potential secrets in string literals
                if len(node.s) > 20 and re.match(r'^[A-Za-z0-9+/=]{20,}$', node.s):
                    # Skip test files
                    if 'test' not in str(file_path).lower():
                        self.issues.append(SecurityIssue(
                            severity="MEDIUM",
                            issue_type="Potential Secret",
                            description="Long base64-like string - potential hardcoded secret",
                            file_path=str(file_path),
                            line_number=node.lineno
                        ))
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(issues)
        visitor.visit(tree)
        
        return issues
    
    def scan_directory(self) -> Dict[str, Any]:
        """Scan entire directory for security issues"""
        print("🔒 Starting security scan of quantum task planner...")
        
        total_files = 0
        scanned_files = 0
        
        # Scan all Python files
        for py_file in self.root_path.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.pytest_cache']):
                continue
            
            total_files += 1
            file_issues = self.scan_file(py_file)
            self.issues.extend(file_issues)
            scanned_files += 1
            
            if file_issues:
                print(f"  ⚠️  {py_file.name}: {len(file_issues)} issues found")
            else:
                print(f"  ✅ {py_file.name}: Clean")
        
        # Additional security checks
        self._check_file_permissions()
        self._check_configuration_security()
        
        # Categorize issues by severity
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0, 
            'MEDIUM': 0,
            'LOW': 0
        }
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        return {
            'total_files': total_files,
            'scanned_files': scanned_files,
            'total_issues': len(self.issues),
            'severity_counts': severity_counts,
            'issues': self.issues
        }
    
    def _check_file_permissions(self):
        """Check for insecure file permissions"""
        sensitive_files = [
            "config.yaml", "config.yml", "settings.py", "secrets.py",
            ".env", "private_key", "id_rsa", "credentials"
        ]
        
        for file_pattern in sensitive_files:
            for file_path in self.root_path.rglob(file_pattern):
                try:
                    mode = oct(file_path.stat().st_mode)[-3:]
                    if mode in self.permission_issues:
                        self.issues.append(SecurityIssue(
                            severity="HIGH",
                            issue_type="File Permissions",
                            description=f"Insecure file permissions: {mode}",
                            file_path=str(file_path)
                        ))
                except Exception:
                    pass
    
    def _check_configuration_security(self):
        """Check configuration files for security issues"""
        config_files = list(self.root_path.rglob("*.yaml")) + list(self.root_path.rglob("*.yml"))
        config_files.extend(self.root_path.rglob("*.json"))
        config_files.extend(self.root_path.rglob("*.ini"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read().lower()
                    
                # Check for plaintext passwords/secrets
                if any(word in content for word in ['password:', 'secret:', 'api_key:', 'token:']):
                    # Check if it looks like a real value (not placeholder)
                    if not any(placeholder in content for placeholder in 
                              ['your_', 'placeholder', 'example', 'dummy', 'test']):
                        self.issues.append(SecurityIssue(
                            severity="HIGH",
                            issue_type="Configuration Security",
                            description="Potential plaintext credentials in configuration",
                            file_path=str(config_file)
                        ))
                        
            except Exception:
                pass
    
    def generate_report(self) -> str:
        """Generate security report"""
        scan_results = self.scan_directory()
        
        report = []
        report.append("🔒 QUANTUM TASK PLANNER SECURITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Summary
        report.append(f"\n📊 SCAN SUMMARY:")
        report.append(f"  • Files scanned: {scan_results['scanned_files']}/{scan_results['total_files']}")
        report.append(f"  • Total issues found: {scan_results['total_issues']}")
        
        # Severity breakdown
        report.append(f"\n🚨 ISSUES BY SEVERITY:")
        for severity, count in scan_results['severity_counts'].items():
            if count > 0:
                emoji = {"CRITICAL": "💥", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}[severity]
                report.append(f"  {emoji} {severity}: {count} issues")
        
        # Detailed issues
        if self.issues:
            report.append(f"\n📋 DETAILED ISSUES:")
            
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                severity_issues = [i for i in self.issues if i.severity == severity]
                if severity_issues:
                    report.append(f"\n{severity} Issues:")
                    for issue in severity_issues:
                        report.append(f"  • {issue}")
        
        # Security recommendations
        report.append(f"\n🛡️  SECURITY RECOMMENDATIONS:")
        report.append(f"  • Implement input validation for all user inputs")
        report.append(f"  • Use environment variables for sensitive configuration")
        report.append(f"  • Enable comprehensive logging for security events")
        report.append(f"  • Implement rate limiting for API endpoints")
        report.append(f"  • Regular security updates and dependency scanning")
        report.append(f"  • Use secure communication protocols (TLS/SSL)")
        report.append(f"  • Implement proper authentication and authorization")
        
        # Overall assessment
        critical_high_issues = scan_results['severity_counts']['CRITICAL'] + scan_results['severity_counts']['HIGH']
        
        if critical_high_issues == 0:
            report.append(f"\n✅ OVERALL ASSESSMENT: GOOD")
            report.append(f"   No critical or high-severity security issues found.")
            report.append(f"   The quantum task planner follows security best practices.")
        elif critical_high_issues <= 2:
            report.append(f"\n⚠️  OVERALL ASSESSMENT: ACCEPTABLE")
            report.append(f"   Few high-priority security issues found.")
            report.append(f"   Address critical/high issues before production deployment.")
        else:
            report.append(f"\n❌ OVERALL ASSESSMENT: NEEDS ATTENTION")
            report.append(f"   Multiple security issues require immediate attention.")
            report.append(f"   Do not deploy to production until issues are resolved.")
        
        return "\n".join(report)

def run_security_scan():
    """Run comprehensive security scan"""
    repo_root = Path(__file__).parent.parent / "src"
    scanner = QuantumSecurityScanner(repo_root)
    
    report = scanner.generate_report()
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / "security_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Security report saved to: {report_file}")
    
    # Return pass/fail status
    critical_high = sum(1 for i in scanner.issues if i.severity in ['CRITICAL', 'HIGH'])
    return critical_high == 0

if __name__ == "__main__":
    success = run_security_scan()
    sys.exit(0 if success else 1)