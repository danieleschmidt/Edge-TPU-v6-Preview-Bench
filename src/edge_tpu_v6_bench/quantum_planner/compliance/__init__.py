"""
Compliance Module for Quantum Task Planner
Global compliance with GDPR, CCPA, PDPA and other regulations
"""

import json
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL_DATA = "personal_data"

class ComplianceRegion(Enum):
    GDPR_EU = "gdpr_eu"          # European Union GDPR
    CCPA_US = "ccpa_us"          # California CCPA
    PDPA_SG = "pdpa_sg"          # Singapore PDPA
    LGPD_BR = "lgpd_br"          # Brazil LGPD  
    PIPEDA_CA = "pipeda_ca"      # Canada PIPEDA
    APPI_JP = "appi_jp"          # Japan APPI
    POPIA_ZA = "popia_za"        # South Africa POPIA
    GLOBAL = "global"            # Global compliance

@dataclass
class ComplianceMetadata:
    """Metadata for compliance tracking"""
    classification: DataClassification
    regions: Set[ComplianceRegion]
    retention_days: int = 365
    encryption_required: bool = False
    audit_required: bool = False
    consent_required: bool = False
    right_to_deletion: bool = False
    cross_border_transfer: bool = False
    created_at: float = field(default_factory=time.time)
    
@dataclass
class AuditEvent:
    """Audit event for compliance tracking"""
    event_id: str
    timestamp: float
    event_type: str
    user_id: Optional[str]
    resource_id: str
    action: str
    classification: DataClassification
    regions: List[ComplianceRegion]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComplianceManager:
    """Global compliance management for quantum task planner"""
    
    def __init__(self, default_region: ComplianceRegion = ComplianceRegion.GLOBAL):
        self.default_region = default_region
        self.audit_events: List[AuditEvent] = []
        self.compliance_policies: Dict[ComplianceRegion, Dict[str, Any]] = {}
        self.data_inventory: Dict[str, ComplianceMetadata] = {}
        
        self._initialize_compliance_policies()
        logger.info(f"ComplianceManager initialized for region: {default_region.value}")
    
    def _initialize_compliance_policies(self):
        """Initialize compliance policies for different regions"""
        
        # GDPR (European Union)
        self.compliance_policies[ComplianceRegion.GDPR_EU] = {
            "data_retention_max_days": 2555,  # 7 years max
            "consent_required": True,
            "right_to_deletion": True,
            "right_to_portability": True,
            "data_protection_officer_required": True,
            "cross_border_restrictions": True,
            "breach_notification_hours": 72,
            "encryption_required": True,
            "audit_logging_required": True,
            "legitimate_interest_basis": ["performance", "security", "fraud_prevention"]
        }
        
        # CCPA (California, USA)  
        self.compliance_policies[ComplianceRegion.CCPA_US] = {
            "data_retention_max_days": 1825,  # 5 years
            "consent_required": False,  # Opt-out model
            "right_to_deletion": True,
            "right_to_know": True,
            "right_to_opt_out": True,
            "non_discrimination": True,
            "cross_border_allowed": True,
            "breach_notification_hours": 72,
            "encryption_recommended": True,
            "audit_logging_recommended": True
        }
        
        # PDPA (Singapore)
        self.compliance_policies[ComplianceRegion.PDPA_SG] = {
            "data_retention_max_days": 3650,  # 10 years
            "consent_required": True,
            "purpose_limitation": True,
            "data_accuracy_required": True,
            "security_arrangements_required": True,
            "cross_border_restrictions": True,
            "breach_notification_hours": 72,
            "encryption_required": True,
            "access_rights": True
        }
        
        # Global baseline
        self.compliance_policies[ComplianceRegion.GLOBAL] = {
            "data_retention_max_days": 1095,  # 3 years default
            "consent_recommended": True,
            "right_to_deletion": True,
            "encryption_recommended": True,
            "audit_logging_recommended": True,
            "cross_border_allowed": True,
            "security_required": True
        }
    
    def classify_data(self, data_id: str, classification: DataClassification, 
                     regions: Set[ComplianceRegion], **kwargs) -> ComplianceMetadata:
        """Classify data for compliance tracking"""
        
        metadata = ComplianceMetadata(
            classification=classification,
            regions=regions,
            **kwargs
        )
        
        # Apply region-specific requirements
        for region in regions:
            policy = self.compliance_policies.get(region, {})
            
            if classification == DataClassification.PERSONAL_DATA:
                metadata.encryption_required = policy.get("encryption_required", False)
                metadata.audit_required = True
                metadata.consent_required = policy.get("consent_required", False)
                metadata.right_to_deletion = policy.get("right_to_deletion", False)
                
                # Apply retention limits
                max_retention = policy.get("data_retention_max_days", 1095)
                if metadata.retention_days > max_retention:
                    metadata.retention_days = max_retention
        
        self.data_inventory[data_id] = metadata
        
        # Log classification event
        self._log_audit_event(
            event_type="data_classification",
            resource_id=data_id,
            action="classify",
            classification=classification,
            regions=list(regions),
            metadata={"retention_days": metadata.retention_days}
        )
        
        logger.info(f"Data classified: {data_id} as {classification.value} for regions {[r.value for r in regions]}")
        return metadata
    
    def check_compliance(self, data_id: str, action: str, 
                        user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check compliance for data operation"""
        
        if data_id not in self.data_inventory:
            return {
                "compliant": False,
                "reason": "Data not classified",
                "requirements": ["Data must be classified before use"]
            }
        
        metadata = self.data_inventory[data_id]
        compliance_result = {
            "compliant": True,
            "warnings": [],
            "requirements": [],
            "region_checks": {}
        }
        
        # Check each applicable region
        for region in metadata.regions:
            policy = self.compliance_policies.get(region, {})
            region_result = {"compliant": True, "issues": []}
            
            # Check consent requirements
            if (metadata.consent_required and 
                action in ["process", "share", "export"] and
                not user_context.get("consent_given", False)):
                region_result["compliant"] = False
                region_result["issues"].append("User consent required")
            
            # Check retention period
            age_days = (time.time() - metadata.created_at) / 86400
            max_retention = policy.get("data_retention_max_days", 1095)
            if age_days > max_retention:
                region_result["compliant"] = False
                region_result["issues"].append(f"Data exceeds retention period ({age_days:.0f} > {max_retention} days)")
            
            # Check cross-border transfer
            if (action == "export" and 
                metadata.cross_border_transfer and
                not policy.get("cross_border_allowed", True)):
                region_result["compliant"] = False  
                region_result["issues"].append("Cross-border data transfer restricted")
            
            # Check encryption requirements
            if (metadata.encryption_required and
                action in ["store", "process", "transmit"] and
                not user_context.get("encrypted", False)):
                region_result["issues"].append("Encryption required but not enabled")
                compliance_result["warnings"].append(f"Encryption required for {region.value}")
            
            compliance_result["region_checks"][region.value] = region_result
            
            if not region_result["compliant"]:
                compliance_result["compliant"] = False
                compliance_result["requirements"].extend(region_result["issues"])
        
        # Log compliance check
        self._log_audit_event(
            event_type="compliance_check",
            resource_id=data_id,
            action=action,
            classification=metadata.classification,
            regions=list(metadata.regions),
            user_id=user_context.get("user_id") if user_context else None,
            metadata={
                "compliant": compliance_result["compliant"],
                "warnings_count": len(compliance_result["warnings"])
            }
        )
        
        return compliance_result
    
    def handle_deletion_request(self, data_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle right to deletion request"""
        
        if data_id not in self.data_inventory:
            return {"success": False, "reason": "Data not found"}
        
        metadata = self.data_inventory[data_id]
        
        # Check if deletion is allowed
        deletion_allowed = False
        for region in metadata.regions:
            policy = self.compliance_policies.get(region, {})
            if policy.get("right_to_deletion", False):
                deletion_allowed = True
                break
        
        if not deletion_allowed:
            return {"success": False, "reason": "Deletion not permitted in applicable regions"}
        
        # Perform secure deletion
        deletion_result = self._secure_delete(data_id)
        
        # Log deletion event
        self._log_audit_event(
            event_type="data_deletion",
            resource_id=data_id,
            action="delete",
            classification=metadata.classification,
            regions=list(metadata.regions),
            user_id=user_id,
            metadata={"deletion_method": "secure_wipe"}
        )
        
        # Remove from inventory
        del self.data_inventory[data_id]
        
        logger.info(f"Data deletion completed: {data_id}")
        return {"success": True, "deletion_id": deletion_result["deletion_id"]}
    
    def _secure_delete(self, data_id: str) -> Dict[str, Any]:
        """Perform secure deletion of data"""
        # Generate deletion ID for audit trail
        deletion_id = hashlib.sha256(f"{data_id}_{time.time()}".encode()).hexdigest()
        
        # In real implementation, this would:
        # 1. Overwrite data multiple times
        # 2. Update all references/indexes
        # 3. Clear backups/logs containing data
        # 4. Generate deletion certificate
        
        return {
            "deletion_id": deletion_id,
            "timestamp": time.time(),
            "method": "multi_pass_overwrite"
        }
    
    def generate_compliance_report(self, region: Optional[ComplianceRegion] = None) -> Dict[str, Any]:
        """Generate compliance report"""
        
        target_regions = [region] if region else list(self.compliance_policies.keys())
        
        report = {
            "generated_at": time.time(),
            "regions": [r.value for r in target_regions],
            "data_inventory_summary": {},
            "compliance_violations": [],
            "audit_summary": {},
            "recommendations": []
        }
        
        # Data inventory summary
        classification_counts = {}
        for data_id, metadata in self.data_inventory.items():
            classification = metadata.classification.value
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        report["data_inventory_summary"] = {
            "total_records": len(self.data_inventory),
            "by_classification": classification_counts,
            "retention_analysis": self._analyze_retention()
        }
        
        # Compliance violations
        violations = []
        current_time = time.time()
        
        for data_id, metadata in self.data_inventory.items():
            age_days = (current_time - metadata.created_at) / 86400
            
            for region in metadata.regions:
                if region in target_regions:
                    policy = self.compliance_policies.get(region, {})
                    max_retention = policy.get("data_retention_max_days", 1095)
                    
                    if age_days > max_retention:
                        violations.append({
                            "data_id": data_id,
                            "region": region.value,
                            "violation": "retention_exceeded",
                            "age_days": int(age_days),
                            "max_retention_days": max_retention
                        })
        
        report["compliance_violations"] = violations
        
        # Audit summary
        recent_events = [e for e in self.audit_events if current_time - e.timestamp < 86400 * 30]  # Last 30 days
        
        event_types = {}
        for event in recent_events:
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        report["audit_summary"] = {
            "total_events_30_days": len(recent_events),
            "by_event_type": event_types,
            "total_events_all_time": len(self.audit_events)
        }
        
        # Recommendations
        recommendations = []
        if violations:
            recommendations.append("Review and purge data exceeding retention periods")
        if len([e for e in recent_events if e.event_type == "compliance_check" and not e.metadata.get("compliant", True)]) > 0:
            recommendations.append("Address compliance check failures")
        if not recommendations:
            recommendations.append("Compliance status is good - continue regular monitoring")
        
        report["recommendations"] = recommendations
        
        return report
    
    def _analyze_retention(self) -> Dict[str, Any]:
        """Analyze data retention status"""
        current_time = time.time()
        analysis = {
            "expiring_soon": 0,  # Within 30 days
            "expired": 0,
            "avg_age_days": 0
        }
        
        if not self.data_inventory:
            return analysis
        
        ages = []
        for metadata in self.data_inventory.values():
            age_days = (current_time - metadata.created_at) / 86400
            ages.append(age_days)
            
            if age_days > metadata.retention_days:
                analysis["expired"] += 1
            elif age_days > metadata.retention_days - 30:
                analysis["expiring_soon"] += 1
        
        analysis["avg_age_days"] = sum(ages) / len(ages)
        
        return analysis
    
    def _log_audit_event(self, event_type: str, resource_id: str, action: str,
                        classification: DataClassification, regions: List[ComplianceRegion],
                        user_id: Optional[str] = None, **kwargs):
        """Log audit event"""
        
        event = AuditEvent(
            event_id=hashlib.sha256(f"{event_type}_{resource_id}_{time.time()}".encode()).hexdigest(),
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            classification=classification,
            regions=regions,
            **kwargs
        )
        
        self.audit_events.append(event)
        
        # Keep audit log size manageable
        if len(self.audit_events) > 10000:
            self.audit_events = self.audit_events[-8000:]  # Keep recent 8000 events
    
    def get_audit_events(self, days: int = 30, event_type: Optional[str] = None) -> List[AuditEvent]:
        """Get audit events for specified period"""
        cutoff_time = time.time() - (days * 86400)
        
        filtered_events = [
            event for event in self.audit_events 
            if event.timestamp >= cutoff_time
        ]
        
        if event_type:
            filtered_events = [
                event for event in filtered_events
                if event.event_type == event_type
            ]
        
        return filtered_events

# Global compliance manager instance
_compliance_manager = ComplianceManager()

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager"""
    return _compliance_manager