import re
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_FILE = PROJECT_ROOT / "results" / "v0.1_evaluation.md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"


def extract_metrics(md_text, section_marker):
    """Extracts Alpha, Mean, and Std Dev from a specific Markdown table section."""
    # Find the section starting with the marker
    section_match = re.search(rf"{section_marker}.*?(?=###|\Z)", md_text, re.DOTALL)
    if not section_match:
        raise ValueError(f"Section '{section_marker}' not found in Markdown.")

    section_text = section_match.group(0)

    alphas, means, stds = [], [], []
    # Match table rows that start with | and contain ± (ignoring header rows)
    for line in section_text.strip().split('\n'):
        if line.startswith('|') and '±' in line:
            # Clean bold markdown markers (**)
            clean_line = line.replace('**', '')
            cols = [c.strip() for c in clean_line.split('|') if c.strip()]

            if len(cols) >= 5:
                alpha = float(cols[1])
                # Parse Hit@1 (column index 2)
                mean_val, std_val = map(float, cols[2].split('±'))

                alphas.append(alpha)
                means.append(mean_val)
                stds.append(std_val)

    # Sort by Alpha to ensure correct chronological plotting order
    sorted_data = sorted(zip(alphas, means, stds))
    if not sorted_data:
        return [], [], []

    return zip(*sorted_data)


# Read the markdown file
md_text = MD_FILE.read_text(encoding="utf-8")

# Extract data for both lines
t_alphas, t_means, t_stds = extract_metrics(md_text, "### Target: Eight-Pointed Star")
o_alphas, o_means, o_stds = extract_metrics(md_text, "### Overall Multi-Class")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 5))
offset = 0.01
o_alphas_shifted = [a + offset for a in o_alphas]

# Plot Target with error bars
plt.errorbar(
    t_alphas, t_means, yerr=t_stds,
    marker="o", linewidth=2, capsize=5, capthick=1.5,
    label="Eight-pointed star"
)

# Plot Overall with error bars
plt.errorbar(
    o_alphas_shifted, o_means, yerr=o_stds,
    marker="s", linewidth=2, capsize=5, capthick=1.5,
    label="Overall multi-class", alpha=0.8
)

plt.title("Symbolic Detective v0.1 — Retrieval Performance")
plt.xlabel("OpenCV Weight (α)")
plt.ylabel("Hit@1")
plt.xticks(t_alphas)
# Expand y-limit slightly above 1.0 to accommodate upper error bar bounds
plt.ylim(0, 1.1)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "v0.1_hit1_vs_alpha.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
