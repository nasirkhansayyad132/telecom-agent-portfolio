"""Enhanced PPT generator with AI insights and charts."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE


class ConversionError(Exception):
    """Raised when a file cannot be converted."""


def _load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Load a dataframe from file content."""
    ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(content)

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(buffer)
    elif ext == ".csv":
        df = pd.read_csv(buffer)
    else:
        raise ConversionError("Unsupported file type. Upload .xlsx, .xls, or .csv.")

    if df.empty or df.shape[1] == 0:
        raise ConversionError("The uploaded file is empty or has no columns.")

    return df


def _quick_stats(df: pd.DataFrame) -> dict:
    """Generate quick statistics about the dataframe."""
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_values = int(df.isna().sum().sum())
    
    stats = {
        "rows": rows,
        "cols": cols,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "missing_values": missing_values,
        "missing_percentage": round((missing_values / (rows * cols)) * 100, 1) if rows * cols > 0 else 0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }
    
    # Add column-level stats for numeric columns
    if numeric_cols:
        numeric_stats = {}
        for col in numeric_cols[:5]:  # Limit to first 5
            numeric_stats[col] = {
                "mean": round(df[col].mean(), 2) if not df[col].isna().all() else None,
                "min": round(df[col].min(), 2) if not df[col].isna().all() else None,
                "max": round(df[col].max(), 2) if not df[col].isna().all() else None,
            }
        stats["numeric_summary"] = numeric_stats
    
    return stats


def _generate_ai_insights(df: pd.DataFrame, stats: dict) -> List[str]:
    """Generate AI-powered insights about the data.
    
    Uses heuristics when OpenAI API is not available, or LLM when available.
    """
    insights = []
    
    # Try to use OpenAI for insights
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Prepare data summary for LLM
            sample_data = df.head(5).to_string()
            column_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns[:10]])
            
            prompt = f"""Analyze this dataset and provide 3-4 key insights:

Dataset Overview:
- Rows: {stats['rows']}, Columns: {stats['cols']}
- Columns: {column_info}
- Missing values: {stats['missing_percentage']}%

Sample data:
{sample_data}

Provide concise, actionable insights about patterns, trends, or notable characteristics. Format as a bulleted list."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            
            ai_text = response.choices[0].message.content or ""
            # Parse bullet points
            for line in ai_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    insights.append(line.lstrip('•-* ').strip())
            
            if insights:
                return insights[:4]  # Limit to 4 insights
                
        except Exception:
            pass  # Fall back to heuristic insights
    
    # Heuristic-based insights
    rows, cols = stats["rows"], stats["cols"]
    
    # Size insight
    if rows > 10000:
        insights.append(f"Large dataset with {rows:,} records - consider pagination for analysis")
    elif rows < 100:
        insights.append(f"Small dataset ({rows} rows) - results may not be statistically significant")
    
    # Missing data insight
    if stats["missing_percentage"] > 10:
        insights.append(f"High missing data rate ({stats['missing_percentage']}%) - data cleaning recommended")
    elif stats["missing_percentage"] > 0:
        insights.append(f"Some missing values detected ({stats['missing_values']} cells)")
    
    # Column type insight
    numeric_pct = (stats["numeric_columns"] / cols * 100) if cols > 0 else 0
    if numeric_pct > 70:
        insights.append("Primarily numeric dataset - suitable for statistical analysis and charts")
    elif numeric_pct < 30:
        insights.append("Primarily categorical data - consider frequency analysis and grouping")
    
    # Numeric summary insights
    if "numeric_summary" in stats:
        for col, col_stats in stats["numeric_summary"].items():
            if col_stats.get("min") is not None and col_stats.get("max") is not None:
                range_val = col_stats["max"] - col_stats["min"]
                if range_val > 0 and col_stats["max"] > 10 * abs(col_stats["min"]):
                    insights.append(f"'{col}' shows wide range ({col_stats['min']} to {col_stats['max']})")
                    break
    
    return insights[:4]


def _add_chart_slide(prs: Presentation, df: pd.DataFrame) -> bool:
    """Add a chart slide if suitable numeric data exists."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if not numeric_cols or len(df) < 2:
        return False
    
    try:
        # Find a suitable column for chart
        chart_col = numeric_cols[0]
        
        # Determine x-axis (use first categorical or index)
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if categorical_cols and df[categorical_cols[0]].nunique() <= 10:
            # Group by categorical column
            grouped = df.groupby(categorical_cols[0])[chart_col].mean().head(8)
            categories = [str(x)[:15] for x in grouped.index]
            values = grouped.values.tolist()
            chart_title = f"Average {chart_col} by {categorical_cols[0]}"
        else:
            # Use first N rows
            sample = df.head(8)
            categories = [str(i) for i in range(len(sample))]
            values = sample[chart_col].fillna(0).tolist()
            chart_title = f"{chart_col} Overview"
        
        # Create chart slide
        slide_layout = prs.slide_layouts[5]  # Blank
        slide = prs.slides.add_slide(slide_layout)
        
        # Add title
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
        title_frame = title_box.text_frame
        title_para = title_frame.paragraphs[0]
        title_para.text = chart_title
        title_para.font.size = Pt(24)
        title_para.font.bold = True
        
        # Create chart data
        chart_data = CategoryChartData()
        chart_data.categories = categories
        chart_data.add_series(chart_col, values)
        
        # Add chart
        x, y, cx, cy = Inches(0.5), Inches(1.2), Inches(9), Inches(5)
        chart = slide.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, cx, cy, chart_data
        ).chart
        
        return True
        
    except Exception:
        return False


def build_ppt(filename: str, content: bytes) -> io.BytesIO:
    """Create an enhanced PPTX report from the uploaded file content."""
    df = _load_dataframe(filename, content)
    stats = _quick_stats(df)
    insights = _generate_ai_insights(df, stats)
    
    preview = df.head(10).iloc[:, :6].fillna("")
    
    prs = Presentation()
    
    # Slide 1: Title
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "📊 Data Intelligence Report"
    subtitle = slide.placeholders[1]
    subtitle.text = f"Source: {filename}\n{stats['rows']:,} rows × {stats['cols']} columns"

    # Slide 2: Quick Stats
    bullet_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_layout)
    slide.shapes.title.text = "Quick Statistics"
    body = slide.shapes.placeholders[1].text_frame
    body.text = f"📋 Total Rows: {stats['rows']:,}"
    body.add_paragraph().text = f"📊 Total Columns: {stats['cols']}"
    body.add_paragraph().text = f"🔢 Numeric Columns: {stats['numeric_columns']}"
    body.add_paragraph().text = f"📝 Categorical Columns: {stats['categorical_columns']}"
    body.add_paragraph().text = f"⚠️ Missing Values: {stats['missing_values']:,} ({stats['missing_percentage']}%)"
    body.add_paragraph().text = f"💾 Memory Usage: {stats['memory_mb']} MB"

    # Slide 3: AI Insights
    if insights:
        slide = prs.slides.add_slide(bullet_layout)
        slide.shapes.title.text = "🤖 AI-Powered Insights"
        body = slide.shapes.placeholders[1].text_frame
        body.text = insights[0]
        for insight in insights[1:]:
            body.add_paragraph().text = insight

    # Slide 4: Chart (if applicable)
    _add_chart_slide(prs, df)

    # Slide 5: Data Preview Table
    table_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(table_layout)
    slide.shapes.title.text = "Data Preview (First Rows)"

    rows_count, cols_count = preview.shape
    table = slide.shapes.add_table(
        rows=rows_count + 1,
        cols=cols_count,
        left=Inches(0.3),
        top=Inches(1.6),
        width=Inches(9.4),
        height=Inches(4.5),
    ).table

    # Header
    for col_idx, col_name in enumerate(preview.columns):
        cell = table.cell(0, col_idx)
        cell.text = str(col_name)[:20]  # Truncate long names

    # Data rows
    for row_idx in range(rows_count):
        for col_idx in range(cols_count):
            value = preview.iat[row_idx, col_idx]
            cell = table.cell(row_idx + 1, col_idx)
            # Truncate long values
            text = str(value)
            cell.text = text[:30] if len(text) > 30 else text

    buffer = io.BytesIO()
    prs.save(buffer)
    buffer.seek(0)
    return buffer


__all__ = ["build_ppt", "ConversionError"]
