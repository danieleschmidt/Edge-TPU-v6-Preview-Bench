#!/usr/bin/env python3
"""
Security Audit for Edge TPU v6 Benchmark Suite
Comprehensive security scanning and vulnerability assessment
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import hashlib
import stat

class SecurityAuditor:
    """Comprehensive security auditor for the codebase"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings: List[Dict] = []
        
        # Security patterns to scan for
        self.vulnerability_patterns = {
            'code_injection': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\..*shell\s*=\s*True',
                r'os\.system\s*\(',
                r'os\.popen\s*\(',
                r'__import__\s*\('
            ],
            'path_traversal': [
                r'\.\./.*',
                r'\.\.\\.*',
                r'/etc/',
                r'/proc/',
                r'C:\\Windows',
                r'%SYSTEMROOT%'
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'-----BEGIN PRIVATE KEY-----',
                r'-----BEGIN RSA PRIVATE KEY-----'
            ],
            'unsafe_deserialization': [
                r'pickle\.loads?\s*\(',
                r'cPickle\.loads?\s*\(',
                r'yaml\.load\s*\(',
                r'marshal\.loads?\s*\('
            ],
            'weak_crypto': [
                r'hashlib\.md5\s*\(',
                r'hashlib\.sha1\s*\(',
                r'random\.random\s*\(',
                r'random\.randint\s*\('
            ],
            'debug_code': [
                r'print\s*\([^)]*password',
                r'print\s*\([^)]*secret',
                r'logging\.debug\([^)]*password',
                r'assert\s+False',
                r'TODO.*security',
                r'FIXME.*security'
            ]
        }
        
        # File extensions to scan
        self.scan_extensions = {'.py', '.yml', '.yaml', '.json', '.cfg', '.ini', '.conf'}
        
        # Exclude patterns
        self.exclude_patterns = [
            r'__pycache__',
            r'\.git',
            r'\.pytest_cache',
            r'node_modules',
            r'\.venv',
            r'venv/',
            r'\.tox'
        ]
    
    def should_scan_file(self, file_path: Path) -> bool:
        """Determine if file should be scanned"""
        # Check extension
        if file_path.suffix not in self.scan_extensions:
            return False
        
        # Check exclude patterns
        path_str = str(file_path)
        for pattern in self.exclude_patterns:
            if re.search(pattern, path_str):
                return False
        
        return True
    
    def scan_file_content(self, file_path: Path) -> List[Dict]:
        """Scan file content for security issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Scan for vulnerability patterns
            for category, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            findings.append({
                                'type': 'vulnerability',
                                'category': category,
                                'file': str(file_path),
                                'line': line_num,
                                'pattern': pattern,
                                'match': match.group(),
                                'context': line.strip(),
                                'severity': self.get_severity(category)
                            })
            
        except Exception as e:
            findings.append({
                'type': 'scan_error',
                'file': str(file_path),
                'error': str(e),
                'severity': 'low'
            })
        
        return findings
    
    def scan_file_permissions(self, file_path: Path) -> List[Dict]:
        """Scan file permissions for security issues"""
        findings = []
        
        try:
            file_stat = file_path.stat()
            mode = stat.filemode(file_stat.st_mode)
            
            # Check for overly permissive files
            if file_stat.st_mode & stat.S_IWOTH:  # World writable
                findings.append({
                    'type': 'permission',
                    'category': 'world_writable',
                    'file': str(file_path),
                    'permissions': mode,
                    'severity': 'high'
                })
            
            if file_stat.st_mode & stat.S_IXOTH and file_path.suffix in ['.py', '.sh']:  # World executable
                findings.append({
                    'type': 'permission',
                    'category': 'world_executable',
                    'file': str(file_path),
                    'permissions': mode,
                    'severity': 'medium'
                })
            
        except Exception as e:
            findings.append({
                'type': 'permission_error',
                'file': str(file_path),
                'error': str(e),
                'severity': 'low'
            })
        
        return findings
    
    def get_severity(self, category: str) -> str:
        """Get severity level for vulnerability category"""
        severity_map = {
            'code_injection': 'critical',
            'path_traversal': 'high',
            'hardcoded_secrets': 'critical',
            'unsafe_deserialization': 'high',
            'weak_crypto': 'medium',
            'debug_code': 'low'
        }
        return severity_map.get(category, 'medium')
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return 'unknown'
    
    def scan_project(self) -> Dict:
        """Scan entire project for security issues"""
        print(f"🔍 Scanning {self.project_root} for security vulnerabilities...")
        
        total_files = 0
        scanned_files = 0
        
        # Scan all relevant files
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                total_files += 1
                
                if self.should_scan_file(file_path):
                    scanned_files += 1
                    
                    # Scan content
                    content_findings = self.scan_file_content(file_path)
                    self.findings.extend(content_findings)
                    
                    # Scan permissions
                    perm_findings = self.scan_file_permissions(file_path)
                    self.findings.extend(perm_findings)
        
        # Analyze findings
        analysis = self.analyze_findings()
        
        return {
            'total_files': total_files,
            'scanned_files': scanned_files,
            'findings': self.findings,
            'analysis': analysis
        }
    
    def analyze_findings(self) -> Dict:
        """Analyze findings and provide summary"""
        if not self.findings:
            return {
                'total_issues': 0,
                'by_severity': {},
                'by_category': {},
                'risk_score': 0
            }
        
        # Count by severity
        severity_counts = {}
        for finding in self.findings:
            severity = finding.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by category
        category_counts = {}
        for finding in self.findings:
            category = finding.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate risk score
        risk_weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}
        risk_score = sum(risk_weights.get(sev, 1) * count 
                        for sev, count in severity_counts.items())
        
        return {
            'total_issues': len(self.findings),
            'by_severity': severity_counts,
            'by_category': category_counts,
            'risk_score': risk_score
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate security audit report"""
        analysis = results['analysis']
        
        report = []
        report.append("🛡️  SECURITY AUDIT REPORT")
        report.append("="*60)
        report.append(f"📁 Project: {self.project_root}")
        report.append(f"📄 Files scanned: {results['scanned_files']} / {results['total_files']}")
        report.append(f"🚨 Issues found: {analysis['total_issues']}")
        report.append(f"⚡ Risk score: {analysis['risk_score']}")
        report.append("")
        
        # Severity breakdown
        if analysis['by_severity']:
            report.append("📊 ISSUES BY SEVERITY")
            report.append("-" * 30)
            for severity, count in sorted(analysis['by_severity'].items(), 
                                        key=lambda x: ['critical', 'high', 'medium', 'low'].index(x[0]) 
                                        if x[0] in ['critical', 'high', 'medium', 'low'] else 999):
                emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡', 'low': '💡'}.get(severity, '❓')
                report.append(f"  {emoji} {severity.upper()}: {count}")
            report.append("")
        
        # Category breakdown
        if analysis['by_category']:
            report.append("📋 ISSUES BY CATEGORY")
            report.append("-" * 30)
            for category, count in sorted(analysis['by_category'].items(), 
                                        key=lambda x: x[1], reverse=True):
                report.append(f"  • {category.replace('_', ' ').title()}: {count}")
            report.append("")
        
        # Detailed findings (top 10)
        if self.findings:
            report.append("🔍 DETAILED FINDINGS (Top 10)")
            report.append("-" * 40)
            
            # Sort by severity priority
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_findings = sorted(
                self.findings,
                key=lambda x: (severity_order.get(x.get('severity', 'low'), 4), x.get('file', ''))
            )
            
            for i, finding in enumerate(sorted_findings[:10], 1):
                severity = finding.get('severity', 'unknown')
                emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡', 'low': '💡'}.get(severity, '❓')
                
                report.append(f"{i}. {emoji} {severity.upper()}: {finding.get('category', 'Unknown')}")
                report.append(f"   📁 File: {finding.get('file', 'Unknown')}")
                if 'line' in finding:
                    report.append(f"   📍 Line: {finding['line']}")
                if 'context' in finding:
                    report.append(f"   📝 Context: {finding['context'][:100]}...")
                report.append("")
        
        # Recommendations
        report.append("💡 SECURITY RECOMMENDATIONS")
        report.append("-" * 40)
        
        recommendations = []
        
        if analysis['by_category'].get('code_injection', 0) > 0:
            recommendations.append("• Replace eval() and exec() with safer alternatives")
            recommendations.append("• Use parameterized queries for database operations")
        
        if analysis['by_category'].get('hardcoded_secrets', 0) > 0:
            recommendations.append("• Move secrets to environment variables or secure vaults")
            recommendations.append("• Implement secret scanning in CI/CD pipeline")
        
        if analysis['by_category'].get('weak_crypto', 0) > 0:
            recommendations.append("• Use SHA-256 or stronger hashing algorithms")
            recommendations.append("• Replace random with cryptographically secure alternatives")
        
        if analysis['by_category'].get('path_traversal', 0) > 0:
            recommendations.append("• Implement path validation and sanitization")
            recommendations.append("• Use allowlists for file access patterns")
        
        if analysis['by_category'].get('world_writable', 0) > 0:
            recommendations.append("• Fix file permissions (remove world-writable)")
        
        # Generic recommendations
        recommendations.extend([
            "• Implement input validation for all user inputs",
            "• Add security testing to CI/CD pipeline",
            "• Regular security audits and dependency scanning",
            "• Enable security linting tools (bandit, safety)",
            "• Implement logging and monitoring for security events"
        ])
        
        for rec in recommendations[:10]:  # Top 10 recommendations
            report.append(rec)
        
        report.append("")
        report.append("🎯 SECURITY STATUS")
        report.append("-" * 20)
        
        if analysis['risk_score'] == 0:
            report.append("✅ EXCELLENT - No security issues found")
        elif analysis['risk_score'] < 10:
            report.append("🟢 GOOD - Low risk level")
        elif analysis['risk_score'] < 50:
            report.append("🟡 MODERATE - Some security concerns")
        elif analysis['risk_score'] < 100:
            report.append("🟠 HIGH - Significant security issues")
        else:
            report.append("🔴 CRITICAL - Serious security vulnerabilities")
        
        return "\n".join(report)

def main():
    """Main entry point"""
    project_root = Path(__file__).parent
    
    auditor = SecurityAuditor(str(project_root))
    results = auditor.scan_project()
    
    # Generate and print report
    report = auditor.generate_report(results)
    print(report)
    
    # Save report to file
    report_file = project_root / 'security_audit_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    # Exit with appropriate code
    risk_score = results['analysis']['risk_score']
    if risk_score > 50:  # High risk threshold
        print("\n⚠️  High risk issues detected - review required")
        sys.exit(1)
    else:
        print(f"\n✅ Security audit completed - risk level acceptable ({risk_score})")
        sys.exit(0)

if __name__ == "__main__":
    main()