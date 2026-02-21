#!/usr/bin/env python3
"""
Test API Endpoints Script

Usage:
    python scripts/test_api_endpoints.py [BASE_URL]

Default BASE_URL: http://localhost:8000
"""

import sys
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Color(Enum):
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    NC = "\033[0m"


@dataclass
class TestResult:
    name: str
    passed: bool
    status_code: int
    expected_code: int
    response: Optional[str] = None


def colored(text: str, color: Color) -> str:
    """Return colored text for terminal."""
    return f"{color.value}{text}{Color.NC.value}"


def test_endpoint(
    base_url: str,
    method: str,
    endpoint: str,
    expected_code: int,
    data: Optional[dict] = None,
    description: str = "",
) -> TestResult:
    """Test a single API endpoint."""
    url = f"{base_url}{endpoint}"

    try:
        if data:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method=method,
            )
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status_code = e.code
        body = e.read().decode("utf-8") if e.fp else ""
    except urllib.error.URLError as e:
        return TestResult(
            name=description,
            passed=False,
            status_code=0,
            expected_code=expected_code,
            response=f"Connection error: {e.reason}",
        )
    except Exception as e:
        return TestResult(
            name=description,
            passed=False,
            status_code=0,
            expected_code=expected_code,
            response=str(e),
        )

    passed = status_code == expected_code
    return TestResult(
        name=description,
        passed=passed,
        status_code=status_code,
        expected_code=expected_code,
        response=body if not passed else None,
    )


def print_result(result: TestResult) -> None:
    """Print test result."""
    status = colored("PASS", Color.GREEN) if result.passed else colored("FAIL", Color.RED)
    print(f"  {result.name}... {status} (HTTP {result.status_code})")
    if result.response and not result.passed:
        print(f"    Response: {result.response[:200]}")


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    print("=" * 50)
    print("Testing Aegis API Endpoints")
    print(f"Base URL: {base_url}")
    print("=" * 50)

    results: list[TestResult] = []

    # Health endpoints
    print("\n--- Health Endpoints ---")
    tests = [
        ("GET", "/health", 200, None, "Health check"),
        ("GET", "/health/live", 200, None, "Liveness probe"),
        ("GET", "/health/ready", 200, None, "Readiness probe"),
    ]
    for method, endpoint, expected, data, desc in tests:
        result = test_endpoint(base_url, method, endpoint, expected, data, desc)
        results.append(result)
        print_result(result)

    # Documentation
    print("\n--- API Documentation ---")
    tests = [
        ("GET", "/docs", 200, None, "Swagger UI"),
        ("GET", "/openapi.json", 200, None, "OpenAPI spec"),
    ]
    for method, endpoint, expected, data, desc in tests:
        result = test_endpoint(base_url, method, endpoint, expected, data, desc)
        results.append(result)
        print_result(result)

    # Sessions
    print("\n--- Session Endpoints ---")
    tests = [
        ("GET", "/api/v1/sessions", 200, None, "List sessions"),
        ("POST", "/api/v1/sessions", 200, {"user_id": "test-user"}, "Create session"),
    ]
    for method, endpoint, expected, data, desc in tests:
        result = test_endpoint(base_url, method, endpoint, expected, data, desc)
        results.append(result)
        print_result(result)

    # Chat
    print("\n--- Chat Endpoints ---")
    tests = [
        ("POST", "/api/v1/chat", 200, {"message": "Hello!", "stream": False}, "Simple chat"),
    ]
    for method, endpoint, expected, data, desc in tests:
        result = test_endpoint(base_url, method, endpoint, expected, data, desc)
        results.append(result)
        print_result(result)

    # Tools
    print("\n--- Tool Endpoints ---")
    tests = [
        ("GET", "/api/v1/tools", 200, None, "List tools"),
    ]
    for method, endpoint, expected, data, desc in tests:
        result = test_endpoint(base_url, method, endpoint, expected, data, desc)
        results.append(result)
        print_result(result)

    # Admin
    print("\n--- Admin Endpoints ---")
    tests = [
        ("GET", "/api/v1/admin/stats", 200, None, "Admin stats"),
        ("GET", "/api/v1/admin/config", 200, None, "Admin config"),
    ]
    for method, endpoint, expected, data, desc in tests:
        result = test_endpoint(base_url, method, endpoint, expected, data, desc)
        results.append(result)
        print_result(result)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print("\n" + "=" * 50)
    print(
        f"Results: {colored(f'{passed} passed', Color.GREEN)}, {colored(f'{failed} failed', Color.RED)}"
    )
    print("=" * 50)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
