"""
Response Parser

Structured output parsing from LLM responses.

Design decisions:
- Multiple output formats (JSON, XML, markdown)
- Schema validation
- Error recovery
- Tool call extraction
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass
class ToolCallParsed:
    """A parsed tool call."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class ParsedResponse:
    """Result of parsing an LLM response."""

    content: str = ""
    tool_calls: list[ToolCallParsed] = field(default_factory=list)
    json_data: dict[str, Any] | None = None
    structured_data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    has_tool_calls: bool = False
    is_json: bool = False
    parse_errors: list[str] = field(default_factory=list)


class ResponseParser:
    """
    Parse structured data from LLM responses.

    Handles various output formats and tool calls.
    """

    def __init__(self):
        self._json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

    def parse(
        self,
        response: str | dict,
        expect_json: bool = False,
        expect_tool_calls: bool = False,
    ) -> ParsedResponse:
        """
        Parse an LLM response.

        Args:
            response: Raw response string or dict
            expect_json: Whether to expect JSON output
            expect_tool_calls: Whether to expect tool calls

        Returns:
            Parsed response with extracted data
        """
        result = ParsedResponse()

        # Handle dict response (from API)
        if isinstance(response, dict):
            result.content = response.get("content", "")

            # Extract tool calls from API format
            if "tool_calls" in response:
                for tc in response["tool_calls"]:
                    parsed_tc = self._parse_api_tool_call(tc)
                    if parsed_tc:
                        result.tool_calls.append(parsed_tc)
                        result.has_tool_calls = True
        else:
            result.content = response

        # Try to extract JSON
        if expect_json or self._looks_like_json(result.content):
            json_data = self._extract_json(result.content)
            if json_data is not None:
                result.json_data = json_data
                result.is_json = True
                result.structured_data = (
                    json_data if isinstance(json_data, dict) else {"data": json_data}
                )

        # Try to extract tool calls from content
        if expect_tool_calls or self._looks_like_tool_call(result.content):
            content_tool_calls = self._extract_tool_calls(result.content)
            for tc in content_tool_calls:
                if not any(existing.name == tc.name for existing in result.tool_calls):
                    result.tool_calls.append(tc)
                    result.has_tool_calls = True

        return result

    def _looks_like_json(self, content: str) -> bool:
        """Check if content looks like JSON."""
        content = content.strip()
        return (
            (content.startswith("{") and content.endswith("}"))
            or (content.startswith("[") and content.endswith("]"))
            or "```json" in content
        )

    def _looks_like_tool_call(self, content: str) -> bool:
        """Check if content looks like it contains tool calls."""
        patterns = [
            r"<tool_call>",
            r"<function_call>",
            r'"function":\s*{',
            r'"tool":\s*"',
        ]
        return any(re.search(p, content) for p in patterns)

    def _extract_json(self, content: str) -> Any:
        """Extract JSON from content."""
        # Try patterns in order
        for pattern in self._json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return None

    def _parse_api_tool_call(self, tc: dict) -> ToolCallParsed | None:
        """Parse tool call from API format."""
        try:
            function_data = tc.get("function", {})
            name = function_data.get("name", "")

            # Arguments might be string or dict
            args = function_data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            return ToolCallParsed(
                id=tc.get("id", str(uuid4())),
                name=name,
                arguments=args,
                raw=json.dumps(tc),
            )
        except Exception:
            return None

    def _extract_tool_calls(self, content: str) -> list[ToolCallParsed]:
        """Extract tool calls from content."""
        tool_calls = []

        # Pattern 1: XML-style <tool_call>
        xml_pattern = r"<tool_call>\s*([\s\S]*?)\s*</tool_call>"
        for match in re.findall(xml_pattern, content):
            tc = self._parse_xml_tool_call(match)
            if tc:
                tool_calls.append(tc)

        # Pattern 2: JSON function calls
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,?\s*"arguments"?\s*:\s*\{[^{}]*\}[^{}]*\}'
        for match in re.findall(json_pattern, content):
            tc = self._parse_json_tool_call(match)
            if tc:
                tool_calls.append(tc)

        return tool_calls

    def _parse_xml_tool_call(self, content: str) -> ToolCallParsed | None:
        """Parse XML-style tool call."""
        try:
            # Try to parse as JSON inside XML
            data = json.loads(content.strip())
            return ToolCallParsed(
                name=data.get("name", ""),
                arguments=data.get("arguments", {}),
                raw=content,
            )
        except json.JSONDecodeError:
            # Try to extract name and arguments manually
            name_match = re.search(r'name["\s:]+([^\s,"]+)', content)
            if name_match:
                name = name_match.group(1).strip('"')
                return ToolCallParsed(name=name, raw=content)
        return None

    def _parse_json_tool_call(self, content: str) -> ToolCallParsed | None:
        """Parse JSON tool call."""
        try:
            data = json.loads(content)
            return ToolCallParsed(
                name=data.get("name", ""),
                arguments=data.get("arguments", {}),
                raw=content,
            )
        except json.JSONDecodeError:
            return None

    def extract_code_blocks(
        self,
        content: str,
        language: str | None = None,
    ) -> list[str]:
        """Extract code blocks from content."""
        if language:
            pattern = rf"```{language}\s*([\s\S]*?)\s*```"
        else:
            pattern = r"```(?:\w*)\s*([\s\S]*?)\s*```"

        return re.findall(pattern, content)

    def extract_sections(
        self,
        content: str,
        headers: list[str] | None = None,
    ) -> dict[str, str]:
        """Extract sections by headers."""
        sections: dict[str, str] = {}

        # Split by markdown headers
        pattern = r"^#{1,3}\s+(.+)$"
        parts = re.split(pattern, content, flags=re.MULTILINE)

        current_header = "introduction"
        for i, part in enumerate(parts):
            if i % 2 == 1:  # This is a header
                current_header = part.strip().lower()
            else:  # This is content
                if headers is None or current_header in [h.lower() for h in headers]:
                    sections[current_header] = part.strip()

        return sections
