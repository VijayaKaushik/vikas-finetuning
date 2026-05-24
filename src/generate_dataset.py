"""
Generates instruction-tuning Q&A pairs from equity_raw.json.

Produces pairs covering definitions, processes, Vijaya_equity workflows,
and compliance topics. Saves as both a plain JSON file and a Hugging Face Dataset.
"""

import json
import os
import random

from datasets import Dataset


def load_raw(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Section parsing — group paragraphs under their nearest heading hierarchy
# ---------------------------------------------------------------------------

def parse_sections(paragraphs: list[dict]) -> list[dict]:
    """Parse raw paragraphs into sections, each with heading context and body text."""
    sections = []
    h1 = h2 = h3 = ""
    body_lines: list[str] = []
    current_heading = ""

    def flush():
        nonlocal body_lines
        if current_heading and body_lines:
            sections.append({
                "h1": h1, "h2": h2, "h3": h3,
                "heading": current_heading,
                "body": "\n".join(body_lines),
            })
        body_lines = []

    for p in paragraphs:
        if p["type"] == "heading":
            flush()
            level = p["heading_level"]
            if level == 1:
                h1 = p["text"]; h2 = ""; h3 = ""
            elif level == 2:
                h2 = p["text"]; h3 = ""
            elif level == 3:
                h3 = p["text"]
            current_heading = p["text"]
        else:
            body_lines.append(p["text"])

    flush()
    return sections


# ---------------------------------------------------------------------------
# Question generators — each returns a list of {instruction, input, output}
# ---------------------------------------------------------------------------

def _qa(instruction: str, output: str) -> dict:
    return {"instruction": instruction, "input": "", "output": output}


def gen_definition_pairs(sections: list[dict]) -> list[dict]:
    """Generate 'What is X?' style definition questions."""
    pairs = []
    for s in sections:
        heading = s["heading"]
        body = s["body"]
        if not body or len(body) < 30:
            continue

        # Basic definition
        pairs.append(_qa(
            f"What is {heading} in equity plan management?",
            body,
        ))

        # Explain variation
        if s["h1"] and s["h1"] != heading:
            pairs.append(_qa(
                f"Explain {heading} in the context of {s['h1']}.",
                body,
            ))

    return pairs


def gen_comparison_pairs(sections: list[dict]) -> list[dict]:
    """Generate comparison questions between sibling sections."""
    pairs = []
    # Group by h1+h2 to find siblings at h3 level
    siblings: dict[str, list[dict]] = {}
    for s in sections:
        key = f"{s['h1']}||{s['h2']}"
        if s["h3"]:
            siblings.setdefault(key, []).append(s)

    for key, group in siblings.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                pairs.append(_qa(
                    f"What is the difference between {a['h3']} and {b['h3']}?",
                    f"{a['h3']}: {a['body']}\n\n{b['h3']}: {b['body']}",
                ))
    return pairs


def gen_process_pairs(sections: list[dict]) -> list[dict]:
    """Generate process/how-to questions."""
    process_keywords = [
        "exercise", "settlement", "release", "forfeiture", "cancellation",
        "vesting", "withholding", "transaction", "administration", "tracking",
        "processing", "grant", "reporting", "integration", "workflows",
        "implementation", "governance",
    ]
    pairs = []
    for s in sections:
        body = s["body"]
        heading_lower = s["heading"].lower()
        if not body or len(body) < 30:
            continue

        if any(kw in heading_lower for kw in process_keywords):
            pairs.append(_qa(
                f"How does {s['heading']} work in equity plan management?",
                body,
            ))
            pairs.append(_qa(
                f"Describe the process of {s['heading']}.",
                body,
            ))
    return pairs


def gen_compliance_pairs(sections: list[dict]) -> list[dict]:
    """Generate compliance and regulatory questions."""
    compliance_keywords = [
        "tax", "compliance", "irc", "section", "rule", "asc", "fas",
        "blackout", "409a", "162(m)", "280g", "10b5", "withholding",
        "regulatory", "disposition",
    ]
    pairs = []
    for s in sections:
        body = s["body"]
        heading_lower = s["heading"].lower()
        if not body or len(body) < 30:
            continue

        if any(kw in heading_lower for kw in compliance_keywords):
            pairs.append(_qa(
                f"What are the compliance requirements for {s['heading']}?",
                body,
            ))
            pairs.append(_qa(
                f"Why is {s['heading']} important for regulatory compliance?",
                body,
            ))
    return pairs


def gen_vijaya_equity_pairs(sections: list[dict]) -> list[dict]:
    """Generate Vijaya_equity company-specific workflow questions."""
    pairs = []
    for s in sections:
        body = s["body"]
        if not body or len(body) < 20:
            continue

        if "vijaya" in s["h1"].lower() or "vijaya" in s["heading"].lower():
            pairs.append(_qa(
                f"How does Vijaya_equity handle {s['heading']}?",
                body,
            ))
            pairs.append(_qa(
                f"What is Vijaya_equity's approach to {s['heading']}?",
                body,
            ))
            pairs.append(_qa(
                f"Describe {s['heading']} at Vijaya_equity.",
                body,
            ))
    return pairs


def gen_list_pairs(sections: list[dict]) -> list[dict]:
    """Generate questions asking to list items when body has bullet-like content."""
    pairs = []
    for s in sections:
        body = s["body"]
        if not body:
            continue
        lines = body.strip().split("\n")
        if len(lines) >= 3:
            pairs.append(_qa(
                f"List the key aspects of {s['heading']}.",
                body,
            ))
    return pairs


def gen_scenario_pairs(sections: list[dict]) -> list[dict]:
    """Generate scenario-based questions for practical topics."""
    pairs = []
    scenario_map = {
        "Stock Split": "If a company performs a 2-for-1 stock split, how are existing equity awards affected?",
        "Reverse Stock Split": "What happens to an employee's stock options during a reverse stock split?",
        "Change in Control (CIC)": "What happens to employee equity awards when a company undergoes a change in control?",
        "Single-Trigger Acceleration": "When does single-trigger acceleration apply, and what does it mean for employees?",
        "Double-Trigger Acceleration": "Under what conditions does double-trigger acceleration take effect?",
        "Forfeiture": "When would an employee forfeit their equity awards?",
        "Blackout Period": "What restrictions apply during an equity blackout period?",
        "Disqualifying Disposition": "What happens when an employee makes a disqualifying disposition of ISO shares?",
        "Qualifying Disposition": "What conditions must be met for a qualifying disposition of ISO shares?",
        "Post-Termination Exercise Period (PTEP)": "How long does an employee have to exercise options after leaving the company?",
        "Underwater Options": "What does it mean when stock options are underwater, and what are the implications?",
        "Clawback / Recoupment": "Under what circumstances can a company claw back equity compensation?",
        "Spin-off": "How does a corporate spin-off affect existing equity awards?",
        "Dilution": "How does equity dilution impact existing shareholders and option holders?",
    }
    section_map = {s["heading"]: s for s in sections}
    for heading, question in scenario_map.items():
        if heading in section_map and section_map[heading]["body"]:
            pairs.append(_qa(question, section_map[heading]["body"]))

    return pairs


def gen_best_practice_pairs(sections: list[dict]) -> list[dict]:
    """Generate best-practice questions for advisory sections."""
    pairs = []
    for s in sections:
        body = s["body"]
        if not body or len(body) < 30:
            continue
        if s["h1"] == "Best Practices and Considerations":
            pairs.append(_qa(
                f"What are the best practices for {s['heading']}?",
                body,
            ))
            pairs.append(_qa(
                f"What should organizations consider regarding {s['heading']}?",
                body,
            ))
    return pairs


def gen_summary_pairs(sections: list[dict]) -> list[dict]:
    """Generate section-summary questions by grouping h3 content under h2."""
    h2_groups: dict[str, list[dict]] = {}
    for s in sections:
        if s["h2"] and s["body"]:
            h2_groups.setdefault(s["h2"], []).append(s)

    pairs = []
    for h2, group in h2_groups.items():
        if len(group) >= 2:
            combined = "\n\n".join(
                f"{s['heading']}: {s['body']}" for s in group
            )
            pairs.append(_qa(
                f"Give an overview of {h2} in equity plan management.",
                combined,
            ))
    return pairs


# ---------------------------------------------------------------------------
# Deduplication and main
# ---------------------------------------------------------------------------

def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove pairs with duplicate instructions."""
    seen = set()
    unique = []
    for p in pairs:
        key = p["instruction"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    raw_path = "data/equity_raw.json"
    json_output = "data/equity_dataset.json"
    hf_output = "data/equity_hf_dataset"

    paragraphs = load_raw(raw_path)
    sections = parse_sections(paragraphs)
    print(f"Parsed {len(sections)} sections from {len(paragraphs)} paragraphs.")

    # Generate all pair types
    all_pairs = []
    generators = [
        ("Definitions", gen_definition_pairs),
        ("Comparisons", gen_comparison_pairs),
        ("Processes", gen_process_pairs),
        ("Compliance", gen_compliance_pairs),
        ("Vijaya_equity", gen_vijaya_equity_pairs),
        ("Lists", gen_list_pairs),
        ("Scenarios", gen_scenario_pairs),
        ("Best Practices", gen_best_practice_pairs),
        ("Summaries", gen_summary_pairs),
    ]

    for name, gen_fn in generators:
        pairs = gen_fn(sections)
        print(f"  {name}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    all_pairs = deduplicate(all_pairs)
    random.seed(42)
    random.shuffle(all_pairs)

    print(f"\nTotal unique pairs: {len(all_pairs)}")

    if len(all_pairs) < 200:
        print(f"WARNING: Only {len(all_pairs)} pairs generated (target: 200+).")

    # Save JSON
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_output}")

    # Save HF Dataset
    dataset = Dataset.from_list(all_pairs)
    os.makedirs(hf_output, exist_ok=True)
    dataset.save_to_disk(hf_output)
    print(f"Saved HF Dataset to {hf_output}/")


if __name__ == "__main__":
    main()
