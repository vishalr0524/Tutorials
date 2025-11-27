import xlsxwriter
from PyQt6.QtCore import QBuffer, QIODevice
from PyQt6.QtGui import QPixmap

def generate_excel_report(filepath, metadata, image_pixmap, calculation_groups):
    workbook = xlsxwriter.Workbook(filepath)
    worksheet = workbook.add_worksheet("Inspection Report")
    
    # Formats
    title_format = workbook.add_format({'font_size': 14, 'align': 'left'}) # Plain text style
    header_format = workbook.add_format({'bold': True, 'font_size': 12, 'bg_color': '#E0E0E0', 'border': 1, 'align': 'center'})
    cell_format = workbook.add_format({'font_size': 11, 'border': 1, 'align': 'center'})
    meta_label_format = workbook.add_format({'bold': True, 'font_size': 11})
    meta_value_format = workbook.add_format({'font_size': 11})
    group_title_format = workbook.add_format({'bold': True, 'font_size': 14, 'underline': True})
    
    # Column Widths
    worksheet.set_column('A:A', 5)   # S.No
    worksheet.set_column('B:B', 20)  # Measure Name
    worksheet.set_column('C:C', 15)  # Value
    worksheet.set_column('D:D', 10)  # Pass/Fail
    worksheet.set_column('E:E', 30)  # Remarks
    
    row = 0
    
    # 1. Title
    worksheet.merge_range('A1:E1', "Weld Inspection Report", title_format)
    row += 2
    
    # 2. Metadata
    worksheet.write(row, 0, "Part Name:", meta_label_format)
    worksheet.write(row, 1, metadata.get('Part Name', ''), meta_value_format)
    worksheet.write(row, 3, "Measure Unit:", meta_label_format)
    worksheet.write(row, 4, metadata.get('Measure Unit', ''), meta_value_format)
    row += 1
    
    worksheet.write(row, 0, "Part Number:", meta_label_format)
    worksheet.write(row, 1, metadata.get('Part Number', ''), meta_value_format)
    worksheet.write(row, 3, "Done By:", meta_label_format)
    worksheet.write(row, 4, metadata.get('Done By', ''), meta_value_format)
    row += 1
    
    worksheet.write(row, 0, "Date:", meta_label_format)
    worksheet.write(row, 1, metadata.get('Date', ''), meta_value_format)
    row += 2
    
    # 3. Image
    if image_pixmap and not image_pixmap.isNull():
        # Save pixmap to buffer
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        image_pixmap.save(buffer, "PNG")
        image_data = buffer.data()
        
        # Insert image (using BytesIO for xlsxwriter)
        from io import BytesIO
        image_stream = BytesIO(image_data)
        
        # Scale image to fit reasonably (e.g., width of 5 columns approx 400px?)
        # Default col width 8.43 chars approx 64px. 
        # A:5, B:20, C:15, D:10, E:30 -> Total ~80 chars ~ 600px
        
        worksheet.insert_image(row, 0, "image.png", {'image_data': image_stream, 'x_scale': 0.8, 'y_scale': 0.8})
        
        # Estimate rows taken by image
        img_height = image_pixmap.height() * 0.8
        row += int(img_height / 20) + 2 # Approx 20px per row
    
    row += 1
    
    # 4. Tables
    headers = ["S.No", "Measure Name", "Value", "Pass/Fail", "Remarks"]
    
    for group in calculation_groups:
        # Group Title
        worksheet.write(row, 0, group.get("title", "Region"), group_title_format)
        row += 1
        
        # Table Header
        for col, header in enumerate(headers):
            worksheet.write(row, col, header, header_format)
        row += 1
        
        # Data Rows
        for i, data in enumerate(group.get("measurements", [])):
            worksheet.write(row, 0, i + 1, cell_format)
            worksheet.write(row, 1, data.get("name", ""), cell_format)
            worksheet.write(row, 2, data.get("value", ""), cell_format)
            worksheet.write(row, 3, data.get("pass_fail", ""), cell_format)
            worksheet.write(row, 4, data.get("remarks", ""), cell_format)
            row += 1
        
        row += 2 # Spacing between tables
        
    workbook.close()
    return True
