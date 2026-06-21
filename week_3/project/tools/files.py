"""
Sandboxed file tools - opencode style (see week_3/2_agent_class.md)

every path gets resolved inside WORKSPACE_ROOT so the model can't wander outside
the project folder. read_file paginates with start_line/read_lines and tells the
model if there's more (has_more). edit_file is line-based (replace/delete/append)
and always returns a small diff so mistakes are visible in the tool log, not silent.
"""

import os
import glob as glob_module

WORKSPACE_ROOT = os.path.abspath(os.environ.get("WORKSPACE_ROOT", "."))
MAX_READ_CHARS = 12_000


def resolve_path(path: str) -> str:
    """resolve a path inside WORKSPACE_ROOT. raises ValueError if it escapes."""
    full = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    if not (full == WORKSPACE_ROOT or full.startswith(WORKSPACE_ROOT + os.sep)):
        raise ValueError(f"path escapes workspace: {path}")
    return full


def read_file(path: str, start_line: int = 1, read_lines: int = 200) -> dict:
    """read a window of lines, numbered, with has_more so the model can page through big files."""
    try:
        full = resolve_path(path)
    except ValueError as e:
        return {"error": str(e)}

    if not os.path.isfile(full):
        return {"error": f"not found: {path}"}

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.read().splitlines()

    total = len(all_lines)
    start_line = max(1, start_line)

    if total > 0 and start_line > total:
        return {"error": f"start_line {start_line} is past end of file ({total} lines)"}

    end_line = min(total, start_line + read_lines - 1)
    window = all_lines[start_line - 1:end_line]
    numbered = "\n".join(f"{i:>5}\t{line}" for i, line in enumerate(window, start=start_line))

    truncated = False
    if len(numbered) > MAX_READ_CHARS:
        numbered = numbered[:MAX_READ_CHARS]
        truncated = True

    return {
        "content": numbered,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total,
        "has_more": end_line < total,
        "truncated": truncated,
    }


def write_file(path: str, content: str) -> dict:
    """create or overwrite a file. use for new notes; use edit_file to update existing ones."""
    try:
        full = resolve_path(path)
    except ValueError as e:
        return {"error": str(e)}

    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)

    return {"path": path, "bytes_written": len(content.encode("utf-8"))}


def edit_file(
    path: str,
    operation: str,
    start_line: int,
    end_line: int | None = None,
    content: str | None = None,
) -> dict:
    """
    line-based edit on an existing file:
      replace(start_line, end_line, content) - swap lines start..end (inclusive) for content
      delete(start_line, end_line)           - remove lines start..end (inclusive)
      append(start_line, content)             - insert content after start_line (0 = before line 1)
    returns a diff preview so mistakes are visible immediately.
    """
    try:
        full = resolve_path(path)
    except ValueError as e:
        return {"error": str(e)}

    if not os.path.isfile(full):
        return {"error": f"not found: {path}"}

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    total = len(lines)
    new_lines = content.splitlines() if content else []

    if operation == "replace":
        if end_line is None:
            return {"error": "replace needs end_line"}
        if start_line < 1 or end_line > total or start_line > end_line:
            return {"error": f"bad range {start_line}-{end_line} for a {total}-line file"}
        old = lines[start_line - 1:end_line]
        lines[start_line - 1:end_line] = new_lines
        diff = "\n".join(f"-{l}" for l in old) + "\n" + "\n".join(f"+{l}" for l in new_lines)

    elif operation == "delete":
        end_line = end_line if end_line is not None else start_line
        if start_line < 1 or end_line > total or start_line > end_line:
            return {"error": f"bad range {start_line}-{end_line} for a {total}-line file"}
        old = lines[start_line - 1:end_line]
        del lines[start_line - 1:end_line]
        diff = "\n".join(f"-{l}" for l in old)

    elif operation == "append":
        if start_line < 0 or start_line > total:
            return {"error": f"bad start_line {start_line} for a {total}-line file"}
        lines[start_line:start_line] = new_lines
        diff = "\n".join(f"+{l}" for l in new_lines)

    else:
        return {"error": f"unknown operation: {operation} (use replace/delete/append)"}

    with open(full, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return {
        "path": path,
        "operation": operation,
        "new_total_lines": len(lines),
        "diff": diff[:2000],
    }


def list_files(path: str = ".", pattern: str = "*") -> dict:
    """glob for files under path (relative to workspace). use '**/*.ext' for recursive."""
    try:
        full = resolve_path(path)
    except ValueError as e:
        return {"error": str(e)}

    if not os.path.isdir(full):
        return {"error": f"not a directory: {path}"}

    recursive = "**" in pattern
    matches = glob_module.glob(os.path.join(full, pattern), recursive=recursive)
    rels = sorted(os.path.relpath(m, WORKSPACE_ROOT) for m in matches)
    return {"path": path, "pattern": pattern, "files": rels, "count": len(rels)}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file with numbered lines. Paginated for big files - pass start_line/"
                "read_lines and check has_more to know if there's more to read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "path relative to workspace root"},
                    "start_line": {"type": "integer", "description": "first line to read (1-indexed), default 1"},
                    "read_lines": {"type": "integer", "description": "how many lines to return, default 200"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. Use for new notes; "
                "use edit_file instead if the file already exists and you only want to change part of it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "e.g. notes/topic-name.md"},
                    "content": {"type": "string", "description": "full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Line-based edit on an existing file: replace, delete, or append lines. "
                "Always read_file immediately before this to confirm current line numbers. "
                "Returns a diff preview of what changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["replace", "delete", "append"]},
                    "start_line": {"type": "integer", "description": "for append, 0 means insert before line 1"},
                    "end_line": {"type": "integer", "description": "required for replace/delete (inclusive)"},
                    "content": {"type": "string", "description": "required for replace/append - new lines, newline-separated"},
                },
                "required": ["path", "operation", "start_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a directory with a glob pattern (e.g. '*.md', '**/*.py' for recursive). Use to explore before read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "directory relative to workspace root, default '.'"},
                    "pattern": {"type": "string", "description": "glob pattern, default '*'"},
                },
                "required": [],
            },
        },
    },
]

DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
}
