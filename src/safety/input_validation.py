"""
Input Validation

Validates and sanitizes user inputs.

Design decisions:
- Multi-layer validation
- Injection detection
- Content filtering
- Configurable policies
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationSeverity(str, Enum):
    """Severity level of validation issues."""
    
    INFO = "info"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    severity: ValidationSeverity = ValidationSeverity.INFO
    issues: list[str] = field(default_factory=list)
    sanitized_input: str | None = None
    
    # Details for debugging
    matched_patterns: list[str] = field(default_factory=list)
    risk_score: float = 0.0


class InjectionDetector:
    """
    Detects potential injection attacks.
    
    Covers:
    - Prompt injection
    - Jailbreak attempts
    - System prompt extraction
    - Delimiter manipulation
    """
    
    # Patterns that may indicate injection attempts
    INJECTION_PATTERNS = [
        # Direct instruction override
        (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?)", "instruction_override"),
        (r"disregard\s+(all\s+)?(previous|above|prior)", "instruction_override"),
        (r"forget\s+(everything|all|what)", "instruction_override"),
        
        # Role manipulation
        (r"you\s+are\s+(now|actually|really)\s+", "role_manipulation"),
        (r"pretend\s+(to\s+be|you\s+are)", "role_manipulation"),
        (r"act\s+as\s+(if\s+you\s+are|a)", "role_manipulation"),
        (r"roleplay\s+as", "role_manipulation"),
        
        # System prompt extraction
        (r"(show|reveal|display|print)\s+(your\s+)?(system\s+)?(prompt|instructions)", "prompt_extraction"),
        (r"what\s+(are|is)\s+your\s+(system\s+)?prompt", "prompt_extraction"),
        (r"repeat\s+(your\s+)?(initial|system|original)\s+(prompt|instructions)", "prompt_extraction"),
        
        # Delimiter attacks
        (r"```system", "delimiter_attack"),
        (r"\[SYSTEM\]", "delimiter_attack"),
        (r"<\|im_start\|>", "delimiter_attack"),
        (r"<\|system\|>", "delimiter_attack"),
        
        # Encoding bypasses
        (r"base64[:\s]", "encoding_bypass"),
        (r"hex[:\s]", "encoding_bypass"),
        (r"\\x[0-9a-fA-F]{2}", "encoding_bypass"),
        
        # Developer mode tricks
        (r"developer\s+mode", "developer_mode"),
        (r"debug\s+mode", "developer_mode"),
        (r"admin\s+mode", "developer_mode"),
        (r"sudo\s+mode", "developer_mode"),
        (r"jailbreak", "developer_mode"),
        (r"DAN\s+mode", "developer_mode"),
    ]
    
    def __init__(self, sensitivity: float = 0.5):
        """
        Initialize detector.
        
        Args:
            sensitivity: 0.0-1.0, higher = more strict
        """
        self._sensitivity = sensitivity
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in self.INJECTION_PATTERNS
        ]
    
    def detect(self, text: str) -> ValidationResult:
        """
        Detect potential injection attempts.
        """
        issues = []
        matched = []
        categories_found = set()
        
        for pattern, category in self._compiled_patterns:
            if pattern.search(text):
                issues.append(f"Potential {category.replace('_', ' ')} detected")
                matched.append(pattern.pattern)
                categories_found.add(category)
        
        # Calculate risk score
        risk_score = len(categories_found) / len(set(c for _, c in self.INJECTION_PATTERNS))
        risk_score = min(1.0, risk_score * 2)  # Amplify for sensitivity
        
        # Determine severity based on sensitivity
        threshold = 1.0 - self._sensitivity
        
        if risk_score >= threshold:
            severity = ValidationSeverity.BLOCKED
            is_valid = False
        elif risk_score > threshold * 0.5:
            severity = ValidationSeverity.WARNING
            is_valid = True
        else:
            severity = ValidationSeverity.INFO
            is_valid = True
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            issues=issues,
            matched_patterns=matched,
            risk_score=risk_score,
        )


class ContentFilter:
    """
    Filters inappropriate or harmful content.
    
    Categories:
    - Violence
    - Hate speech
    - Self-harm
    - Sexual content
    - Personal information
    """
    
    # Simple keyword-based filtering (in production, use ML models)
    CONTENT_PATTERNS = {
        "violence": [
            r"\b(kill|murder|attack|assault|bomb)\b",
        ],
        "hate": [
            r"\b(hate|racist|sexist)\b\s+\b(all|every)\b",
        ],
        "personal_info": [
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ],
    }
    
    def __init__(
        self,
        enabled_categories: set[str] | None = None,
        block_personal_info: bool = True,
    ):
        self._categories = enabled_categories or set(self.CONTENT_PATTERNS.keys())
        self._block_pii = block_personal_info
        
        self._compiled = {}
        for category, patterns in self.CONTENT_PATTERNS.items():
            if category in self._categories:
                self._compiled[category] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]
    
    def filter(self, text: str) -> ValidationResult:
        """
        Filter content for inappropriate material.
        """
        issues = []
        matched = []
        
        for category, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    issues.append(f"Content matched {category} filter")
                    matched.append(pattern.pattern)
        
        # Determine severity
        if "personal_info" in str(matched) and self._block_pii:
            severity = ValidationSeverity.WARNING
            is_valid = True  # Allow but warn
        elif issues:
            severity = ValidationSeverity.WARNING
            is_valid = True
        else:
            severity = ValidationSeverity.INFO
            is_valid = True
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            issues=issues,
            matched_patterns=matched,
        )
    
    def redact_pii(self, text: str) -> str:
        """Redact personal information from text."""
        result = text
        
        for pattern in self._compiled.get("personal_info", []):
            result = pattern.sub("[REDACTED]", result)
        
        return result


class InputValidator:
    """
    Main input validation coordinator.
    
    Combines multiple validation strategies.
    """
    
    def __init__(
        self,
        injection_detector: InjectionDetector | None = None,
        content_filter: ContentFilter | None = None,
        max_length: int = 100000,
        min_length: int = 1,
    ):
        self._injection = injection_detector or InjectionDetector()
        self._content = content_filter or ContentFilter()
        self._max_length = max_length
        self._min_length = min_length
    
    def validate(
        self,
        text: str,
        check_injection: bool = True,
        check_content: bool = True,
        sanitize: bool = False,
    ) -> ValidationResult:
        """
        Validate input text.
        
        Returns combined validation result.
        """
        all_issues = []
        max_severity = ValidationSeverity.INFO
        all_patterns = []
        max_risk = 0.0
        
        # Length check
        if len(text) < self._min_length:
            all_issues.append(f"Input too short (min: {self._min_length})")
            max_severity = ValidationSeverity.BLOCKED
        
        if len(text) > self._max_length:
            all_issues.append(f"Input too long (max: {self._max_length})")
            max_severity = ValidationSeverity.BLOCKED
        
        # Injection detection
        if check_injection:
            injection_result = self._injection.detect(text)
            all_issues.extend(injection_result.issues)
            all_patterns.extend(injection_result.matched_patterns)
            max_risk = max(max_risk, injection_result.risk_score)
            
            if injection_result.severity.value > max_severity.value:
                max_severity = injection_result.severity
        
        # Content filtering
        if check_content:
            content_result = self._content.filter(text)
            all_issues.extend(content_result.issues)
            all_patterns.extend(content_result.matched_patterns)
            
            if content_result.severity.value > max_severity.value:
                max_severity = content_result.severity
        
        # Sanitization
        sanitized = text
        if sanitize:
            sanitized = self._sanitize(text)
        
        return ValidationResult(
            is_valid=max_severity != ValidationSeverity.BLOCKED,
            severity=max_severity,
            issues=all_issues,
            sanitized_input=sanitized if sanitize else None,
            matched_patterns=all_patterns,
            risk_score=max_risk,
        )
    
    def _sanitize(self, text: str) -> str:
        """Apply sanitization to text."""
        result = text
        
        # Remove control characters
        result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result)
        
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result)
        
        # Redact PII
        result = self._content.redact_pii(result)
        
        return result.strip()
