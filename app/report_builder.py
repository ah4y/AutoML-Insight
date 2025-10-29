"""PDF Report Builder for AutoML-Insight."""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from pathlib import Path
import json


class ReportBuilder:
    """Generate PDF reports for AutoML results."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.report_dir = Path("results/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
    
    def generate_report(
        self,
        data,
        profile,
        results,
        task_type,
        recommendation=None
    ) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            data: Original dataset
            profile: Dataset profile
            results: Model evaluation results
            task_type: 'Classification' or 'Clustering'
            recommendation: Model recommendation dict
            
        Returns:
            Path to generated PDF
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"AutoML_Report_{timestamp}.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for report elements
        story = []
        
        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(
            data, profile, results, task_type, recommendation
        ))
        story.append(PageBreak())
        
        # Dataset information
        story.extend(self._create_dataset_section(data, profile))
        story.append(PageBreak())
        
        # Model results
        story.extend(self._create_results_section(results, task_type))
        story.append(PageBreak())
        
        # Recommendation
        if recommendation:
            story.extend(self._create_recommendation_section(recommendation))
            story.append(PageBreak())
        
        # Appendix
        story.extend(self._create_appendix(timestamp))
        
        # Build PDF
        doc.build(story)
        
        return str(report_path)
    
    def _create_title_page(self):
        """Create title page."""
        elements = []
        
        elements.append(Spacer(1, 2 * inch))
        
        title = Paragraph("AutoML-Insight Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3 * inch))
        
        subtitle = Paragraph(
            "Automated Machine Learning Analysis",
            self.styles['Normal']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.5 * inch))
        
        date = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        )
        elements.append(date)
        
        return elements
    
    def _create_executive_summary(self, data, profile, results, task_type, recommendation):
        """Create executive summary."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.heading_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Dataset overview
        summary_text = f"""
        <b>Dataset:</b> {data.shape[0]} samples, {data.shape[1]} features<br/>
        <b>Task:</b> {task_type}<br/>
        <b>Models Evaluated:</b> {len(results)}<br/>
        """
        
        if recommendation:
            summary_text += f"<b>Best Model:</b> {recommendation.get('recommended_model', 'N/A')}<br/>"
            summary_text += f"<b>Best Score:</b> {recommendation.get('score', 0):.4f}<br/>"
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _create_dataset_section(self, data, profile):
        """Create dataset information section."""
        elements = []
        
        elements.append(Paragraph("Dataset Profile", self.heading_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Profile table
        profile_data = [['Metric', 'Value']]
        for key, value in list(profile.items())[:15]:  # Limit to top 15
            if isinstance(value, float):
                profile_data.append([key, f"{value:.4f}"])
            else:
                profile_data.append([key, str(value)])
        
        table = Table(profile_data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _create_results_section(self, results, task_type):
        """Create model results section."""
        elements = []
        
        elements.append(Paragraph("Model Performance", self.heading_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Results table
        if task_type == "Classification":
            table_data = [['Model', 'Accuracy', 'F1-Score', 'ROC-AUC']]
            for model_name, result in results.items():
                table_data.append([
                    model_name,
                    f"{result.get('accuracy_mean', 0):.4f}",
                    f"{result.get('f1_macro_mean', 0):.4f}",
                    f"{result.get('roc_auc_ovr_mean', 0):.4f}"
                ])
        else:
            table_data = [['Model', 'Silhouette', 'Davies-Bouldin', 'N Clusters']]
            for model_name, result in results.items():
                table_data.append([
                    model_name,
                    f"{result.get('silhouette', 0):.4f}",
                    f"{result.get('davies_bouldin', 0):.4f}",
                    str(result.get('n_clusters', 0))
                ])
        
        table = Table(table_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _create_recommendation_section(self, recommendation):
        """Create recommendation section."""
        elements = []
        
        elements.append(Paragraph("Recommendation", self.heading_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Best model
        best_model_text = f"""
        <b>Recommended Model:</b> {recommendation.get('recommended_model', 'N/A')}<br/>
        <b>Score:</b> {recommendation.get('score', 0):.4f}<br/>
        <b>95% CI:</b> [{recommendation.get('ci_lower', 0):.4f}, {recommendation.get('ci_upper', 0):.4f}]<br/>
        """
        elements.append(Paragraph(best_model_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Rationale
        elements.append(Paragraph("<b>Rationale:</b>", self.styles['Normal']))
        for idx, reason in enumerate(recommendation.get('rationale', []), 1):
            elements.append(Paragraph(f"{idx}. {reason}", self.styles['Normal']))
            elements.append(Spacer(1, 0.1 * inch))
        
        return elements
    
    def _create_appendix(self, timestamp):
        """Create appendix."""
        elements = []
        
        elements.append(Paragraph("Appendix", self.heading_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        metadata_text = f"""
        <b>Report Metadata:</b><br/>
        - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        - AutoML-Insight Version: 1.0<br/>
        - Run ID: {timestamp}<br/>
        """
        elements.append(Paragraph(metadata_text, self.styles['Normal']))
        
        return elements
