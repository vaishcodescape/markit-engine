"""
Notification Logging

Maintains logs/notifications.md and logs/notifications.jsonl for audit trail.
"""
import json
import os
from datetime import datetime


def log_notification(alert_type, subject, claude_response, data,
                     layers_triggered=None, recommendation=None):
    """
    Log a notification to both markdown and JSONL files.

    Appends to:
    - logs/notifications.md   (human-readable)
    - logs/notifications.jsonl (machine-readable)

    Args:
        alert_type:       'urgent' or 'digest'.
        subject:          Notification subject line.
        claude_response:  Full LLM response text.
        data:             Dict of all layer results.
        layers_triggered: Optional list of layer names that triggered the alert.
        recommendation:   Optional recommendation string.
    """
    os.makedirs("logs", exist_ok=True)

    now = datetime.now()
    prices = data.get("prices", {}) if data else {}

    # -- Human-readable markdown -----------------------------------------------
    md_lines = [
        "\n---\n",
        f"## {now.strftime('%Y-%m-%d %H:%M ET')} -- {alert_type.upper()} LOGGED\n",
        f"**Subject:** {subject}\n",
    ]

    if layers_triggered:
        md_lines.append(f"**Layers triggered:** {', '.join(layers_triggered)}\n")

    if recommendation:
        md_lines.append(f"**Recommendation:** {recommendation}\n")

    md_lines.append("\n**Portfolio at time of alert:**\n")
    for ticker, p in prices.items():
        if ticker.startswith("__") or not isinstance(p, dict) or p.get("error"):
            continue
        md_lines.append(
            f"- {ticker}: ${p.get('price', 0):.2f} "
            f"({p.get('change_pct', 0):+.1f}%) | "
            f"P&L: {p.get('pnl_pct', 0):+.1f}%\n"
        )

    md_lines.append("\n**Analysis:**\n")
    md_lines.append(
        claude_response[:800] + ("..." if len(claude_response) > 800 else "")
    )
    md_lines.append("\n")

    with open("logs/notifications.md", "a") as f:
        f.writelines(md_lines)

    # -- Machine-readable JSON -------------------------------------------------
    entry = {
        "timestamp":          now.isoformat(),
        "type":               alert_type,
        "subject":            subject,
        "layers_triggered":   layers_triggered or [],
        "recommendation":     recommendation,
        "portfolio_snapshot": {
            k: {
                "price":      v.get("price"),
                "pnl_pct":    v.get("pnl_pct"),
                "change_pct": v.get("change_pct"),
            }
            for k, v in prices.items()
            if not k.startswith("__") and isinstance(v, dict) and not v.get("error")
        },
        "response_preview": claude_response[:400],
    }

    with open("logs/notifications.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    print("  Logged to notifications.md and notifications.jsonl")
