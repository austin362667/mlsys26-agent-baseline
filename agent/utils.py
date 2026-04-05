import argparse
import json
import os
import re

import yaml

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

DATASET_ROOT_CANDIDATES = {
    "flashinfer-trace": [
        os.path.join(REPO_TOP_PATH, "datasets", "flashinfer-trace"),
        os.path.join(REPO_TOP_PATH, "flashinfer-trace"),
        os.path.join(REPO_TOP_PATH, "../flashinfer-contest-playground/mlsys26-contest")
    ],
    "mlsys26-contest": [
        os.path.join(REPO_TOP_PATH, "datasets", "mlsys26-contest"),
        os.path.join(REPO_TOP_PATH, "mlsys26-contest"),
        os.path.join(REPO_TOP_PATH, "../flashinfer-contest-playground/mlsys26-contest")
    ],
}


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        code = code_match.group(1).strip()
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()
        return code

    return output_string


def str_replace(
    file_content: str,
    old_str: str,
    new_str: str | None,
    encoding: str = "utf-8",
) -> str:
    """
    Implement the str_replace command, which replaces old_str with new_str in the file content.

    Args:
        file_content: The original file content
        old_str: String to replace
        new_str: Replacement string
        encoding: The encoding to use (auto-detected by decorator)
    """
    new_str = new_str or ""

    pattern = re.escape(old_str)
    occurrences = [
        (
            file_content.count("\n", 0, match.start()) + 1,
            match.group(),
            match.start(),
        )
        for match in re.finditer(pattern, file_content)
    ]

    if not occurrences:
        old_str = old_str.strip()
        new_str = new_str.strip()
        pattern = re.escape(old_str)
        occurrences = [
            (
                file_content.count("\n", 0, match.start()) + 1,
                match.group(),
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]
        if not occurrences:
            print(
                f"[Warning] No replacement was performed, old_str\n ```\n{old_str}\n```\ndid not appear verbatim in the file."
            )
            return file_content
    if len(occurrences) > 1:
        line_numbers = sorted(set(line for line, _, _ in occurrences))
        print(
            f"[Warning] No replacement was performed. Multiple occurrences of old_str\n ```\n{old_str}\n```\nin lines {line_numbers}. Please ensure it is unique."
        )
        return file_content

    replacement_line, matched_text, idx = occurrences[0]
    new_file_content = (
        file_content[:idx] + new_str + file_content[idx + len(matched_text) :]
    )
    return new_file_content


def extract_edits(output: str):
    """Extract str_replace edits from LLM output."""
    edits = []
    for line in output.split("\n"):
        if line.strip().startswith("<old_str_"):
            try:
                idx = int(line.strip().split("_")[2].split(">")[0])
                raw_old_str = output.split("<old_str_" + str(idx) + ">")[1].split(
                    "</old_str_" + str(idx) + ">"
                )[0]
                raw_new_str = output.split("<new_str_" + str(idx) + ">")[1].split(
                    "</new_str_" + str(idx) + ">"
                )[0]
                edits.append((raw_old_str, raw_new_str))
            except:
                continue
    return edits


# Dataset paths


def get_dataset_root(test_source: str) -> str:
    """Return dataset root path for the given test source."""
    if test_source not in DATASET_ROOT_CANDIDATES:
        raise ValueError(
            f"Unknown test_source: {test_source}, expected one of {list(DATASET_ROOT_CANDIDATES)}"
        )

    for candidate in DATASET_ROOT_CANDIDATES[test_source]:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Dataset root for '{test_source}' not found. Checked: "
        + ", ".join(DATASET_ROOT_CANDIDATES[test_source])
    )


def construct_flashinfer_trace_dataset(
    op_type: str, dataset_root: str = None,
) -> list[str]:
    """Return sorted list of problem names for given op_type."""
    if dataset_root is None:
        dataset_root = get_dataset_root("flashinfer-trace")
    op_dir = os.path.join(dataset_root, "definitions", op_type)
    return sorted(f[:-5] for f in os.listdir(op_dir) if f.endswith(".json"))


def load_flashinfer_trace_definition(
    op_type: str,
    problem_name: str,
    dataset_root: str = None,
) -> dict:
    """Load definition JSON for given op_type and problem_name."""
    if dataset_root is None:
        dataset_root = get_dataset_root("flashinfer-trace")
    path = os.path.join(dataset_root, "definitions", op_type, f"{problem_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def load_test_source(test_source: str, level, problem_id):
    """
    Load reference architecture source code.
    level=op_type (str), problem_id=problem_name (str)
    """
    dataset_root = get_dataset_root(test_source)
    definition = load_flashinfer_trace_definition(
        str(level), str(problem_id), dataset_root=dataset_root
    )
    return str(problem_id), definition


def load_tasks_from_test_list(
    tasks_path: str, test_source: str = "mlsys26-contest",
) -> list[dict]:
    """
    Load tasks from a test list file.

    Args:
        tasks_path: Path to the tasks file
        test_source: Dataset source identifier

    Returns:
        List of task dicts with 'level' and 'problem_id' keys
    """
    dataset_root = get_dataset_root(test_source)

    with open(tasks_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    tasks = []
    for line in lines:
        parts = line.split(" ", 1)
        level = parts[0]
        if len(parts) == 1:
            problems = construct_flashinfer_trace_dataset(level, dataset_root=dataset_root)
        else:
            problems = [p.strip() for p in parts[1].split(",") if p.strip()]
        tasks.extend({"level": level, "problem_id": str(p)} for p in problems)

    return tasks


def load_config_from_yaml(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """
    Load configuration from YAML file and update args.
    Command line arguments take precedence over YAML values.
    """
    if args.config is None:
        return args

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Set YAML values as new defaults, then re-parse so CLI args override them
    parser.set_defaults(**{k: v for k, v in config_dict.items() if hasattr(args, k)})
    return parser.parse_args()
