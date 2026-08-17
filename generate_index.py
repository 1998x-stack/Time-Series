#!/usr/bin/env python3
"""从磁盘扫描章节/.md/.py,自动生成 README.md 的章节索引(夹在标记之间)。"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_START = "<!-- INDEX:START -->"
INDEX_END = "<!-- INDEX:END -->"
CHAPTER_PAT = re.compile(r"^第\d+章_")


def scan_chapters():
    """扫描磁盘, 返回 [(章名, [(md文件名, py文件名或None), ...]), ...]。"""
    chapters = []
    for name in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, name)
        if os.path.isdir(d) and CHAPTER_PAT.match(name):
            pynames = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".py")}
            mds = sorted(f for f in os.listdir(d) if f.endswith(".md"))
            entries = []
            for md in mds:
                base = os.path.splitext(md)[0]
                entries.append((md, base + ".py" if base in pynames else None))
            chapters.append((name, entries))
    return chapters


def render(chapters):
    out = []
    for name, entries in chapters:
        out.append(f"## {name}")
        out.append("")
        for md, py_name in entries:
            label = os.path.splitext(md)[0]
            out.append(f"- [{label}](.//{name}/{md})")
            if py_name:
                out.append(f"- [代码: {label}](.//{name}/{py_name})")
        out.append("")
    return "\n".join(out)


def main():
    index_block = render(scan_chapters())
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find(INDEX_START)
    end = content.find(INDEX_END)
    if start == -1 or end == -1:
        raise SystemExit("README.md 缺少 INDEX marker; 请先写入标记。")
    head = content[: start + len(INDEX_START)] + "\n\n"
    tail = "\n" + content[end:]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(head + index_block + tail)
    print("README 索引已更新。")


if __name__ == "__main__":
    main()