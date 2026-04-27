from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


HISTORICAL_INPUT_PATH = Path("data/processed/harden_bucks_guard_comparison.csv")
SIMILAR_INPUT_PATH = Path("data/processed/harden_similar_creator_guards.csv")

OUTPUT_DIR = Path("reports/final_dashboard")
OUTPUT_PATH = OUTPUT_DIR / "Harden_Bucks_Performance_Fit_Dashboard.html"

DOCS_DIR = Path("docs")
DOCS_OUTPUT_PATH = DOCS_DIR / "index.html"


HARDEN_COLOR = "#7C3AED"
BUCKS_COLOR = "#00471B"
CONTRACT_COLOR = "#C99700"
DARK = "#111827"

GROUP_COLORS = {
    "James Harden": HARDEN_COLOR,
    "Milwaukee Bucks Guard": BUCKS_COLOR,
}

CONTRACT_TIER_COLORS = {
    "Max / Near-Max Creator": "#7C3AED",
    "Premium Veteran Guard": "#2563EB",
    "Mid-Tier Rotation Guard": "#C99700",
    "Low-Cost Guard": "#059669",
    "Contract Unknown": "#64748B",
}


def validate_historical_input(df):
    required_columns = [
        "SEASON",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "player_group",
        "GP",
        "MIN",
        "PTS",
        "AST",
        "TOV",
        "AST_PCT",
        "AST_TO",
        "TS_PCT",
        "OFF_RATING",
        "USG_PCT",
        "playmaking_value_index",
        "half_court_creator_index",
        "performance_argument_score",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing historical dashboard columns: {missing}")

    if df.empty:
        raise ValueError("Historical dashboard input is empty.")

    if "James Harden" not in df["player_group"].unique():
        raise ValueError("James Harden rows missing from historical dataset.")


def validate_similar_input(df):
    required_columns = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "AGE",
        "PTS",
        "AST",
        "TOV",
        "AST_PCT",
        "USG_PCT",
        "TS_PCT",
        "OFF_RATING",
        "performance_argument_score",
        "harden_similarity_score",
        "aav",
        "aav_millions",
        "contract_tier",
        "performance_score_per_1m_aav",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing similar-player dashboard columns: {missing}")

    if df.empty:
        raise ValueError("Similar-player dashboard input is empty.")

    if "James Harden" not in df["PLAYER_NAME"].values:
        raise ValueError("James Harden missing from similar-player dataset.")


def get_harden_row(similar_df):
    harden = similar_df[similar_df["PLAYER_NAME"] == "James Harden"]

    if harden.empty:
        raise ValueError("James Harden row missing from similar dataset.")

    return harden.iloc[0]


def get_decay_rate_for_age(age):
    if age <= 31:
        return 0.045
    if age <= 33:
        return 0.055
    if age <= 35:
        return 0.070
    if age <= 37:
        return 0.090
    return 0.120


def simulate_harden_trajectory(similar_df, threshold=32.0, max_years=5):
    harden = get_harden_row(similar_df)

    current_age = float(harden["AGE"])
    current_score = float(harden["performance_argument_score"])

    rows = []
    age = current_age
    score = current_score

    rows.append(
        {
            "season_step": "Current",
            "age": age,
            "projected_score": score,
            "status": "Starter-Level Creator" if score >= threshold else "Below Threshold",
        }
    )

    productive_future_years = 0

    for year in range(1, max_years + 1):
        decay_rate = get_decay_rate_for_age(age)
        score = score * (1 - decay_rate)
        age = age + 1

        if score >= threshold:
            productive_future_years += 1

        rows.append(
            {
                "season_step": f"+{year} Year",
                "age": age,
                "projected_score": score,
                "status": "Starter-Level Creator" if score >= threshold else "Below Threshold",
            }
        )

    trajectory_df = pd.DataFrame(rows)

    return trajectory_df, productive_future_years


def build_summary(historical_df, similar_df):
    harden = historical_df[historical_df["player_group"] == "James Harden"]
    bucks = historical_df[historical_df["player_group"] == "Milwaukee Bucks Guard"]

    harden_current = get_harden_row(similar_df)
    trajectory_df, projected_productive_future_years = simulate_harden_trajectory(similar_df)

    return {
        "harden_avg_pts": harden["PTS"].mean(),
        "bucks_avg_pts": bucks["PTS"].mean(),
        "harden_avg_ast": harden["AST"].mean(),
        "bucks_avg_ast": bucks["AST"].mean(),
        "harden_avg_creator": harden["half_court_creator_index"].mean(),
        "bucks_avg_creator": bucks["half_court_creator_index"].mean(),
        "harden_current_similarity": harden_current["harden_similarity_score"],
        "harden_current_aav": harden_current["aav_millions"],
        "similar_players": len(similar_df),
        "harden_current_age": harden_current["AGE"],
        "projected_productive_future_years": projected_productive_future_years,
        "current_harden_score": harden_current["performance_argument_score"],
        "age_curve_terminal_score": trajectory_df["projected_score"].iloc[-1],
    }


def create_creation_trend_chart(df):
    harden = df[df["player_group"] == "James Harden"].sort_values("SEASON")

    bucks_avg = (
        df[df["player_group"] == "Milwaukee Bucks Guard"]
        .groupby("SEASON")
        .agg(
            avg_half_court_creator_index=("half_court_creator_index", "mean"),
        )
        .reset_index()
        .sort_values("SEASON")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=harden["SEASON"],
            y=harden["half_court_creator_index"],
            mode="lines+markers",
            name="James Harden",
            line=dict(color=HARDEN_COLOR, width=4),
            marker=dict(size=9),
            hovertemplate=(
                "<b>James Harden</b><br>"
                "Season: %{x}<br>"
                "Half-Court Creator Index: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bucks_avg["SEASON"],
            y=bucks_avg["avg_half_court_creator_index"],
            mode="lines+markers",
            name="Bucks Guard Average",
            line=dict(color=BUCKS_COLOR, width=4, dash="dash"),
            marker=dict(size=9),
            hovertemplate=(
                "<b>Bucks Guard Average</b><br>"
                "Season: %{x}<br>"
                "Half-Court Creator Index: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Half-Court Creation: Harden vs Bucks Guard Average",
        height=400,
        margin=dict(l=70, r=40, t=60, b=60),
        xaxis_title="Season",
        yaxis_title="Half-Court Creator Index",
        template="plotly_white",
        font=dict(family="Arial", size=12, color=DARK),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )

    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

    return fig


def create_playmaking_gap_chart(df):
    group_summary = (
        df.groupby("player_group")
        .agg(
            avg_ast=("AST", "mean"),
            avg_ast_pct=("AST_PCT", "mean"),
            avg_ast_to=("AST_TO", "mean"),
            avg_tov=("TOV", "mean"),
        )
        .reset_index()
    )

    group_summary["color"] = group_summary["player_group"].map(GROUP_COLORS)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=group_summary["player_group"],
            y=group_summary["avg_ast"],
            marker=dict(color=group_summary["color"]),
            text=group_summary["avg_ast"].round(1),
            textposition="outside",
            customdata=group_summary[["avg_ast_pct", "avg_ast_to", "avg_tov"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "AST: %{y:.1f}<br>"
                "AST%: %{customdata[0]:.3f}<br>"
                "AST/TO: %{customdata[1]:.2f}<br>"
                "TOV: %{customdata[2]:.1f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Playmaking Gap: Average Assists Per Game",
        height=400,
        margin=dict(l=70, r=40, t=60, b=70),
        xaxis_title="Player Group",
        yaxis_title="Assists Per Game",
        template="plotly_white",
        showlegend=False,
        font=dict(family="Arial", size=12, color=DARK),
    )

    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

    return fig


def create_ast_pct_vs_usage_chart(df):
    fig = go.Figure()

    for group, group_df in df.groupby("player_group"):
        color = HARDEN_COLOR if group == "James Harden" else BUCKS_COLOR

        fig.add_trace(
            go.Scatter(
                x=group_df["USG_PCT"],
                y=group_df["AST_PCT"],
                mode="markers",
                name=group,
                text=group_df["PLAYER_NAME"] + " | " + group_df["SEASON"],
                marker=dict(
                    size=group_df["MIN"] * 0.35,
                    color=color,
                    opacity=0.78,
                    line=dict(width=1, color="#111827"),
                ),
                customdata=group_df[["PTS", "AST", "TOV", "TS_PCT"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "USG%: %{x:.3f}<br>"
                    "AST%: %{y:.3f}<br>"
                    "PTS: %{customdata[0]:.1f}<br>"
                    "AST: %{customdata[1]:.1f}<br>"
                    "TOV: %{customdata[2]:.1f}<br>"
                    "TS%: %{customdata[3]:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Creation Load: Assist Percentage vs Usage",
        height=400,
        margin=dict(l=70, r=40, t=60, b=70),
        xaxis_title="Usage Percentage",
        yaxis_title="Assist Percentage",
        template="plotly_white",
        font=dict(family="Arial", size=12, color=DARK),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

    return fig


def create_top_player_seasons_chart(df):
    chart_df = (
        df.sort_values("performance_argument_score", ascending=False)
        .head(14)
        .sort_values("performance_argument_score", ascending=True)
        .copy()
    )

    chart_df["label"] = (
        chart_df["PLAYER_NAME"]
        + " | "
        + chart_df["SEASON"]
        + " | "
        + chart_df["TEAM_ABBREVIATION"]
    )

    colors = [
        HARDEN_COLOR if group == "James Harden" else BUCKS_COLOR
        for group in chart_df["player_group"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["performance_argument_score"],
            y=chart_df["label"],
            orientation="h",
            marker=dict(color=colors),
            text=chart_df["performance_argument_score"].round(1),
            textposition="inside",
            insidetextanchor="end",
            customdata=chart_df[["PTS", "AST", "TOV", "OFF_RATING", "TS_PCT"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Performance Argument Score: %{x:.1f}<br>"
                "PTS: %{customdata[0]:.1f}<br>"
                "AST: %{customdata[1]:.1f}<br>"
                "TOV: %{customdata[2]:.1f}<br>"
                "OFF Rating: %{customdata[3]:.1f}<br>"
                "TS%: %{customdata[4]:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Top Historical Player-Seasons by Performance Argument Score",
        height=400,
        margin=dict(l=230, r=60, t=60, b=50),
        xaxis_title="Performance Argument Score",
        yaxis_title="",
        template="plotly_white",
        showlegend=False,
        font=dict(family="Arial", size=12, color=DARK),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=False)

    return fig


def create_similarity_chart(similar_df):
    chart_df = (
        similar_df.sort_values("harden_similarity_score", ascending=False)
        .head(10)
        .sort_values("harden_similarity_score", ascending=True)
        .copy()
    )

    chart_df["label"] = (
        chart_df["PLAYER_NAME"]
        + " | "
        + chart_df["TEAM_ABBREVIATION"]
    )

    colors = [
        HARDEN_COLOR if player == "James Harden" else CONTRACT_TIER_COLORS.get(tier, "#64748B")
        for player, tier in zip(chart_df["PLAYER_NAME"], chart_df["contract_tier"])
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["harden_similarity_score"],
            y=chart_df["label"],
            orientation="h",
            marker=dict(color=colors),
            text=chart_df["harden_similarity_score"].round(1),
            textposition="inside",
            insidetextanchor="end",
            customdata=chart_df[
                [
                    "PTS",
                    "AST",
                    "AST_PCT",
                    "USG_PCT",
                    "TS_PCT",
                    "performance_argument_score",
                    "aav_millions",
                    "contract_tier",
                ]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Similarity to Harden: %{x:.1f}<br>"
                "PTS: %{customdata[0]:.1f}<br>"
                "AST: %{customdata[1]:.1f}<br>"
                "AST%: %{customdata[2]:.3f}<br>"
                "USG%: %{customdata[3]:.3f}<br>"
                "TS%: %{customdata[4]:.3f}<br>"
                "Performance Score: %{customdata[5]:.1f}<br>"
                "AAV: $%{customdata[6]:.1f}M<br>"
                "Contract Tier: %{customdata[7]}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Current Creator Guards: Similarity to Harden",
        height=400,
        margin=dict(l=210, r=60, t=60, b=50),
        xaxis_title="Similarity Score",
        yaxis_title="",
        template="plotly_white",
        showlegend=False,
        font=dict(family="Arial", size=12, color=DARK),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=False)

    return fig


def create_contract_value_chart(similar_df):
    fig = go.Figure()

    for tier, group in similar_df.groupby("contract_tier"):
        fig.add_trace(
            go.Scatter(
                x=group["aav_millions"],
                y=group["performance_argument_score"],
                mode="markers+text",
                name=tier,
                text=group["PLAYER_NAME"],
                textposition="top center",
                marker=dict(
                    size=group["harden_similarity_score"] * 0.35 + 10,
                    color=CONTRACT_TIER_COLORS.get(tier, "#64748B"),
                    opacity=0.78,
                    line=dict(width=1, color="#111827"),
                ),
                customdata=group[
                    [
                        "TEAM_ABBREVIATION",
                        "PTS",
                        "AST",
                        "TS_PCT",
                        "harden_similarity_score",
                        "performance_score_per_1m_aav",
                    ]
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Team: %{customdata[0]}<br>"
                    "AAV: $%{x:.1f}M<br>"
                    "Performance Score: %{y:.1f}<br>"
                    "PTS: %{customdata[1]:.1f}<br>"
                    "AST: %{customdata[2]:.1f}<br>"
                    "TS%: %{customdata[3]:.3f}<br>"
                    "Similarity to Harden: %{customdata[4]:.1f}<br>"
                    "Performance Score per $1M AAV: %{customdata[5]:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Contract Value vs Performance Argument Score",
        height=400,
        margin=dict(l=70, r=40, t=60, b=80),
        xaxis_title="Average Annual Value ($M)",
        yaxis_title="Performance Argument Score",
        template="plotly_white",
        font=dict(family="Arial", size=12, color=DARK),
        legend=dict(orientation="h", y=-0.32, x=0.5, xanchor="center"),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

    return fig


def create_age_performance_chart(similar_df):
    df = similar_df.copy()

    df["marker_color"] = np.where(df["PLAYER_NAME"] == "James Harden", HARDEN_COLOR, "#94A3B8")
    df["marker_size"] = np.where(df["PLAYER_NAME"] == "James Harden", 22, 14)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["AGE"],
            y=df["performance_argument_score"],
            mode="markers+text",
            text=df["PLAYER_NAME"],
            textposition="top center",
            marker=dict(
                size=df["marker_size"],
                color=df["marker_color"],
                opacity=0.85,
                line=dict(width=1, color="#111827"),
            ),
            customdata=df[
                [
                    "TEAM_ABBREVIATION",
                    "PTS",
                    "AST",
                    "TS_PCT",
                    "aav_millions",
                    "harden_similarity_score",
                ]
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Age: %{x:.0f}<br>"
                "Performance Score: %{y:.1f}<br>"
                "Team: %{customdata[0]}<br>"
                "PTS: %{customdata[1]:.1f}<br>"
                "AST: %{customdata[2]:.1f}<br>"
                "TS%: %{customdata[3]:.3f}<br>"
                "AAV: $%{customdata[4]:.1f}M<br>"
                "Similarity to Harden: %{customdata[5]:.1f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    valid = df[["AGE", "performance_argument_score"]].dropna()

    if len(valid) >= 2:
        coeffs = np.polyfit(valid["AGE"], valid["performance_argument_score"], 1)
        x_line = np.linspace(valid["AGE"].min(), valid["AGE"].max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Age-Performance Trend",
                line=dict(color=CONTRACT_COLOR, width=3, dash="dot"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Age vs Performance Argument Score",
        height=400,
        margin=dict(l=70, r=40, t=60, b=60),
        xaxis_title="Age",
        yaxis_title="Performance Argument Score",
        template="plotly_white",
        font=dict(family="Arial", size=12, color=DARK),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

    return fig


def create_harden_trajectory_chart(similar_df):
    trajectory_df, projected_productive_future_years = simulate_harden_trajectory(similar_df, threshold=32.0, max_years=5)

    threshold_line = [32.0] * len(trajectory_df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trajectory_df["age"],
            y=trajectory_df["projected_score"],
            mode="lines+markers+text",
            name="James Harden Trajectory",
            line=dict(color=HARDEN_COLOR, width=4),
            marker=dict(size=10),
            text=trajectory_df["season_step"],
            textposition="top center",
            customdata=trajectory_df[["status"]],
            hovertemplate=(
                "<b>James Harden</b><br>"
                "Age: %{x:.0f}<br>"
                "Projected Performance Score: %{y:.1f}<br>"
                "Status: %{customdata[0]}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trajectory_df["age"],
            y=threshold_line,
            mode="lines",
            name="Starter-Level Threshold",
            line=dict(color="#EF4444", width=3, dash="dash"),
            hovertemplate=(
                "<b>Threshold</b><br>"
                "Age: %{x:.0f}<br>"
                "Threshold Score: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Harden Aging Curve: Projected Productive Window = {projected_productive_future_years} More Years",
        height=400,
        margin=dict(l=70, r=40, t=60, b=60),
        xaxis_title="Age",
        yaxis_title="Projected Performance Argument Score",
        template="plotly_white",
        font=dict(family="Arial", size=12, color=DARK),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

    return fig


def build_html(summary, figures):
    html_blocks = []

    for index, fig in enumerate(figures):
        html_blocks.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs="cdn" if index == 0 else False,
                config={"displayModeBar": False},
            )
        )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Harden to Milwaukee Performance Fit Dashboard</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #F3F4F6;
            font-family: Arial, Helvetica, sans-serif;
            color: #111827;
        }}

        .page {{
            max-width: 1500px;
            margin: 0 auto;
            padding: 32px;
        }}

        .hero {{
            background: linear-gradient(135deg, #00471B, #111827);
            color: white;
            padding: 30px 34px;
            border-radius: 18px;
            margin-bottom: 24px;
        }}

        .hero h1 {{
            margin: 0;
            font-size: 34px;
            letter-spacing: -0.5px;
        }}

        .hero p {{
            margin: 10px 0 0 0;
            max-width: 1120px;
            line-height: 1.5;
            color: #E5E7EB;
            font-size: 15px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: white;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
            border: 1px solid #E5E7EB;
        }}

        .kpi-label {{
            font-size: 12px;
            color: #6B7280;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 10px;
        }}

        .kpi-value {{
            font-size: 22px;
            font-weight: 800;
            color: #111827;
            line-height: 1.2;
        }}

        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
        }}

        .chart-card {{
            background: white;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
            border: 1px solid #E5E7EB;
            overflow: hidden;
        }}

        .talking-points {{
            margin-top: 24px;
            background: white;
            border-radius: 16px;
            padding: 22px;
            font-size: 14px;
            color: #374151;
            line-height: 1.55;
            border: 1px solid #E5E7EB;
        }}

        .talking-points strong {{
            color: #111827;
        }}

        @media (max-width: 1200px) {{
            .kpi-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>Harden to Milwaukee: Performance Fit Dashboard</h1>
            <p>
                A negotiation dashboard positioning James Harden as a lead guard, half-court organizer,
                and creation engine for Milwaukee. The argument is not that Harden must be the old Houston version.
                The argument is that Milwaukee needs reliable half-court creation, playmaking stability,
                and a possession manager next to Giannis.
            </p>
        </section>

        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Harden PPG</div>
                <div class="kpi-value">{summary["harden_avg_pts"]:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Bucks Guard PPG</div>
                <div class="kpi-value">{summary["bucks_avg_pts"]:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Harden APG</div>
                <div class="kpi-value">{summary["harden_avg_ast"]:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Creator Gap</div>
                <div class="kpi-value">{summary["harden_avg_creator"] - summary["bucks_avg_creator"]:.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Harden Age</div>
                <div class="kpi-value">{summary["harden_current_age"]:.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Projected Productive Years</div>
                <div class="kpi-value">{summary["projected_productive_future_years"]}</div>
            </div>
        </section>

        <section class="grid">
            <div class="chart-card">{html_blocks[0]}</div>
            <div class="chart-card">{html_blocks[1]}</div>
            <div class="chart-card">{html_blocks[2]}</div>
            <div class="chart-card">{html_blocks[3]}</div>
            <div class="chart-card">{html_blocks[4]}</div>
            <div class="chart-card">{html_blocks[5]}</div>
            <div class="chart-card">{html_blocks[6]}</div>
            <div class="chart-card">{html_blocks[7]}</div>
        </section>

        <section class="talking-points">
            <strong>Negotiation Performance Argument:</strong>
            Harden gives Milwaukee a proven half-court creation profile that the recent Bucks guard history has rarely matched.
            His value is not only scoring. It is advantage creation, assist volume, pace control, and late-clock organization.
            The similar-player contract layer shows that Harden belongs in the same creator conversation as premium guards.
            The age-performance view shows that he is older, but still materially above several active comparison guards in creation value.
            The aging trajectory does not guarantee future output, but it supports a business case for a short-term competitive window.
            <br><br>
            <strong>Methodology Note:</strong>
            Performance data is pulled through NBA API. Contract data is a manual public contract layer and should be verified before final submission.
            AAV means average annual value. The Performance Argument Score is a custom negotiation metric combining playmaking, scoring efficiency,
            offensive rating, turnover risk, assist percentage, usage, and true shooting. The aging curve is a scenario-based model with progressive
            annual decay and a starter-level threshold set at 32.0.
        </section>
    </div>
</body>
</html>
"""

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_PATH.write_text(html, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    historical_df = pd.read_csv(HISTORICAL_INPUT_PATH)
    similar_df = pd.read_csv(SIMILAR_INPUT_PATH)

    validate_historical_input(historical_df)
    validate_similar_input(similar_df)

    summary = build_summary(historical_df, similar_df)

    figures = [
        create_creation_trend_chart(historical_df),
        create_playmaking_gap_chart(historical_df),
        create_ast_pct_vs_usage_chart(historical_df),
        create_top_player_seasons_chart(historical_df),
        create_similarity_chart(similar_df),
        create_contract_value_chart(similar_df),
        create_age_performance_chart(similar_df),
        create_harden_trajectory_chart(similar_df),
    ]

    build_html(summary, figures)

    print("")
    print("HARDEN BUCKS DASHBOARD UPDATED")
    print("------------------------------")
    print(f"Dashboard: {OUTPUT_PATH}")
    print(f"GitHub Pages copy: {DOCS_OUTPUT_PATH}")
    print("")
    print("Open with:")
    print(f"open {OUTPUT_PATH}")
    print("")


if __name__ == "__main__":
    main()