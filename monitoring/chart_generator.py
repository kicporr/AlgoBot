"""Utility to generate equity curve chart images for trading reports.

Supports both matplotlib (primary) and a PIL-based custom drawing fallback
in environments where matplotlib is not installed.
"""

import os
from loguru import logger

def generate_equity_chart(equity_series: list, output_path: str) -> bool:
    """Generate and save an equity curve chart.

    Args:
        equity_series: List of float values representing cumulative equity.
        output_path: Path where the generated image should be saved.

    Returns:
        True if the chart was successfully generated and saved, False otherwise.
    """
    if not equity_series:
        logger.warning("Cannot generate chart: equity series is empty.")
        return False

    # Try matplotlib first
    try:
        import matplotlib
        matplotlib.use('Agg')  # Headless mode
        import matplotlib.pyplot as plt
        
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')

        # Plot equity line
        ax.plot(equity_series, color='#3b82f6', linewidth=2.5, label='Krzywa kapitału (Equity)')
        
        # Fill underneath the line
        ax.fill_between(range(len(equity_series)), equity_series, min(equity_series), color='#3b82f6', alpha=0.15)
        
        # Grid styling
        ax.grid(True, linestyle='--', color='#1e293b', alpha=0.5)
        
        # Title & Labels
        ax.set_title('Krzywa Kapitału Portfela (Equity)', fontsize=14, fontweight='bold', color='#f8fafc', pad=15)
        ax.set_xlabel('Transakcje', fontsize=11, color='#94a3b8')
        ax.set_ylabel('Kapitał ($)', fontsize=11, color='#94a3b8')
        
        # Tick colors
        ax.tick_params(colors='#94a3b8', labelsize=9)
        
        # Highlight final equity
        final_eq = equity_series[-1]
        ax.text(
            len(equity_series) - 1, final_eq, f"  ${final_eq:,.2f}",
            color='#10b981' if final_eq >= equity_series[0] else '#ef4444',
            fontweight='bold', va='center'
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        logger.info(f"Successfully generated equity chart using matplotlib at {output_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Matplotlib chart generation failed, using Pillow fallback: {e}")

    # Fallback: Pillow Custom Drawing (Premium Dark Theme)
    try:
        from PIL import Image, ImageDraw
        
        width = 800
        height = 400
        
        # Create base image
        img = Image.new('RGB', (width, height), color='#0f172a')
        
        min_eq = min(equity_series)
        max_eq = max(equity_series)
        eq_range = max_eq - min_eq if max_eq != min_eq else 1.0
        
        # Padding
        pad_left = 80
        pad_right = 40
        pad_top = 50
        pad_bottom = 50
        
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom
        
        n = len(equity_series)
        
        # Draw grid and labels
        draw = ImageDraw.Draw(img)
        
        # Y-axis grids (5 levels)
        for i in range(5):
            y = pad_top + int(plot_h * i / 4)
            val = max_eq - (eq_range * i / 4)
            draw.line([(pad_left, y), (width - pad_right, y)], fill='#1e293b', width=1)
            # Y label
            draw.text((10, y - 5), f"${val:,.1f}", fill='#64748b')
            
        # X-axis grid lines (e.g. 5 ticks)
        for i in range(5):
            x = pad_left + int(plot_w * i / 4)
            idx = int((n - 1) * i / 4) if n > 1 else 0
            draw.line([(x, pad_top), (x, height - pad_bottom)], fill='#1e293b', width=1)
            # X label
            draw.text((x - 10, height - pad_bottom + 10), str(idx), fill='#64748b')

        # Calculate point coordinates
        points = []
        for idx, eq in enumerate(equity_series):
            x = pad_left + int(plot_w * idx / (n - 1)) if n > 1 else pad_left
            y = pad_top + int(plot_h * (1.0 - (eq - min_eq) / eq_range))
            points.append((x, y))

        if len(points) > 1:
            # Create alpha overlay for filled polygon
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Construct polygon points (area under the curve)
            poly_points = [points[0]] + points + [points[-1], (points[-1][0], height - pad_bottom), (points[0][0], height - pad_bottom), points[0]]
            # Fill with 15% opacity blue
            overlay_draw.polygon(poly_points, fill=(59, 130, 246, 38))
            
            # Combine overlay with base
            img = Image.alpha_composite(img.convert('RGBA'), overlay)
            
            # Draw line on top of composite image
            draw_composite = ImageDraw.Draw(img)
            draw_composite.line(points, fill='#3b82f6', width=3)
            
            # Draw final value text
            final_x, final_y = points[-1]
            final_val = equity_series[-1]
            text_color = '#10b981' if final_val >= equity_series[0] else '#ef4444'
            draw_composite.text((final_x + 5, final_y - 5), f"${final_val:,.2f}", fill=text_color)
        else:
            # Single point fallback
            draw.ellipse([(points[0][0]-4, points[0][1]-4), (points[0][0]+4, points[0][1]+4)], fill='#3b82f6')

        # Title
        draw = ImageDraw.Draw(img)
        draw.text((pad_left, 15), "Krzywa Kapitalu Portfela (Equity)", fill='#f8fafc')
        
        # Save as PNG
        img.convert('RGB').save(output_path, 'PNG')
        logger.info(f"Successfully generated equity chart using Pillow fallback at {output_path}")
        return True
        
    except Exception as ex:
        logger.error(f"Pillow fallback chart generation failed: {ex}")
        return False
