from PyQt6.QtGui import QPainter, QPageSize, QPageLayout, QFont, QPixmap, QImage, QColor, QPen, QBrush
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtCore import QRectF, Qt, QDate

def generate_pdf_report(filepath, metadata, image_pixmap, calculation_groups):
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(filepath)
    printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    
    # Create Painter
    painter = QPainter()
    if not painter.begin(printer):
        print("Failed to open printer")
        return False

    # Page Dimensions (in printer units)
    page_rect = printer.pageLayout().paintRectPixels(printer.resolution())
    width = page_rect.width()
    height = page_rect.height()
    
    x_margin = 100
    y_curr = 150 # Increased top margin
    line_height = 150
    
    # Fonts
    title_font = QFont("Arial", 24, QFont.Weight.Bold)
    header_font = QFont("Arial", 14, QFont.Weight.Bold)
    normal_font = QFont("Arial", 12)
    table_header_font = QFont("Arial", 11, QFont.Weight.Bold)
    table_cell_font = QFont("Arial", 11)
    
    # Colors
    header_bg_color = QColor("#E0E0E0")
    border_color = QColor("#000000")
    
    # 1. Title
    painter.setFont(title_font)
    painter.drawText(QRectF(x_margin, y_curr, width - 2*x_margin, line_height), Qt.AlignmentFlag.AlignCenter, "Weld Inspection Report")
    y_curr += line_height * 2.5 # Increased spacing
    
    # 2. Metadata
    painter.setFont(normal_font)
    
    # Define columns
    col1_x = x_margin
    col2_x = x_margin + (width - 2*x_margin) / 2
    col_width = (width - 2*x_margin) / 2
    
    # Row 1
    rect1 = QRectF(col1_x, y_curr, col_width, line_height)
    painter.drawText(rect1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Part Name: {metadata.get('Part Name', '')}")
    
    rect2 = QRectF(col2_x, y_curr, col_width, line_height)
    painter.drawText(rect2, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Measure Unit: {metadata.get('Measure Unit', '')}")
    
    y_curr += line_height
    
    # Row 2
    rect3 = QRectF(col1_x, y_curr, col_width, line_height)
    painter.drawText(rect3, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Part Number: {metadata.get('Part Number', '')}")
    
    rect4 = QRectF(col2_x, y_curr, col_width, line_height)
    painter.drawText(rect4, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Done By: {metadata.get('Done By', '')}")
    
    y_curr += line_height
    
    # Row 3
    rect5 = QRectF(col1_x, y_curr, col_width, line_height)
    painter.drawText(rect5, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Date: {metadata.get('Date', '')}")
    
    y_curr += line_height * 2.5 # Increased spacing
    
    # 3. Image
    if image_pixmap and not image_pixmap.isNull():
        # Scale image to fit width, maintain aspect ratio
        avail_width = width - 2*x_margin
        avail_height = height / 2.5 # Limit height
        
        scaled_pixmap = image_pixmap.scaled(int(avail_width), int(avail_height), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        # Center image
        img_x = x_margin + (avail_width - scaled_pixmap.width()) / 2
        
        painter.drawPixmap(int(img_x), int(y_curr), scaled_pixmap)
        y_curr += scaled_pixmap.height() + line_height * 2 # Increased spacing
    
    # 4. Tables (Loop through groups)
    
    # Table Headers
    headers = ["S.No", "Measure Name", "Value", "Pass/Fail", "Remarks"]
    col_widths = [0.1, 0.3, 0.2, 0.15, 0.25] # Percentages
    col_widths_px = [w * (width - 2*x_margin) for w in col_widths]
    
    for group in calculation_groups:
        # Check for page break (Header + Table Header + 1 Row approx)
        if y_curr + line_height * 4 > height - x_margin:
            printer.newPage()
            y_curr = x_margin
            
        # Group Title
        painter.setFont(header_font)
        painter.drawText(int(x_margin), int(y_curr), group.get("title", "Region"))
        y_curr += line_height
        
        # Draw Table Header
        painter.setFont(table_header_font)
        x_curr = x_margin
        
        for i, header in enumerate(headers):
            rect = QRectF(x_curr, y_curr, col_widths_px[i], line_height)
            
            # Background
            painter.fillRect(rect, QBrush(header_bg_color))
            # Border
            painter.setPen(QPen(border_color))
            painter.drawRect(rect)
            # Text
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, header)
            
            x_curr += col_widths_px[i]
            
        y_curr += line_height
        
        # Draw Data Rows
        painter.setFont(table_cell_font)
        
        for i, row_data in enumerate(group.get("measurements", [])):
            x_curr = x_margin
            
            # S.No
            rect = QRectF(x_curr, y_curr, col_widths_px[0], line_height)
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(i + 1))
            x_curr += col_widths_px[0]
            
            # Measure Name
            rect = QRectF(x_curr, y_curr, col_widths_px[1], line_height)
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(row_data.get("name", "")))
            x_curr += col_widths_px[1]
            
            # Value
            rect = QRectF(x_curr, y_curr, col_widths_px[2], line_height)
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(row_data.get("value", "")))
            x_curr += col_widths_px[2]
            
            # Pass/Fail
            rect = QRectF(x_curr, y_curr, col_widths_px[3], line_height)
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(row_data.get("pass_fail", "")))
            x_curr += col_widths_px[3]
            
            # Remarks
            rect = QRectF(x_curr, y_curr, col_widths_px[4], line_height)
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(row_data.get("remarks", "")))
            x_curr += col_widths_px[4]
            
            y_curr += line_height
            
            # Check for page break
            if y_curr > height - x_margin:
                printer.newPage()
                y_curr = x_margin
        
        y_curr += line_height # Spacing between tables
    
    painter.end()
    return True
