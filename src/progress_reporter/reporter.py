"""
Progress Reporter
==================
Generates weekly progress reports for students,
tracking improvement and mastery changes.
"""

import numpy as np
from datetime import datetime


def generate_progress_report(student_id, student_data, mastery_scores, 
                            weak_areas, gap_result, study_plan):
    """
    Generate a comprehensive progress report for a student.
    
    Args:
        student_id: Student identifier
        student_data: Dict of student demographic/academic data
        mastery_scores: Dict of concept -> mastery probability
        weak_areas: List of identified weak areas
        gap_result: Result from gap detector
        study_plan: Generated study plan
    """
    report = {
        'student_id': student_id,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'overview': {},
        'grades': {},
        'mastery': {},
        'risk_assessment': {},
        'recommendations': [],
        'weekly_goals': []
    }
    
    # Overview
    report['overview'] = {
        'name': f"Student #{student_id}",
        'school': student_data.get('school', 'N/A'),
        'age': student_data.get('age', 'N/A'),
        'gender': student_data.get('sex', 'N/A'),
    }
    
    # Grade progression
    g1 = student_data.get('G1', 0)
    g2 = student_data.get('G2', 0)
    g3 = student_data.get('G3', 0)
    
    grade_change = g3 - g1
    if grade_change > 0:
        trend = 'Improving'
    elif grade_change < 0:
        trend = 'Declining'
    else:
        trend = 'Stable'
    
    report['grades'] = {
        'G1': g1,
        'G2': g2,
        'G3': g3,
        'grade_change': grade_change,
        'trend': trend,
        'average': round((g1 + g2 + g3) / 3, 1)
    }
    
    # Mastery breakdown
    report['mastery'] = {
        'scores': mastery_scores,
        'overall': round(np.mean(list(mastery_scores.values())) * 100, 1) if mastery_scores else 0,
        'strongest': max(mastery_scores, key=mastery_scores.get) if mastery_scores else 'N/A',
        'weakest': min(mastery_scores, key=mastery_scores.get) if mastery_scores else 'N/A',
    }
    
    # Risk assessment
    report['risk_assessment'] = {
        'is_at_risk': gap_result.get('is_at_risk', False),
        'risk_probability': round(gap_result.get('risk_probability', 0) * 100, 1),
        'weak_areas': weak_areas,
        'n_high_severity': sum(1 for a in weak_areas if a['severity'] == 'high'),
        'n_medium_severity': sum(1 for a in weak_areas if a['severity'] == 'medium'),
    }
    
    # Weekly goals
    if gap_result.get('is_at_risk', False):
        report['weekly_goals'] = [
            f"Improve {report['mastery']['weakest']} mastery by 10%",
            "Complete all daily study plan activities",
            "Attend all classes (reduce absences)",
            "Take the weekend practice assessment",
            "Seek help from teacher/tutor for high-severity areas"
        ]
    else:
        report['weekly_goals'] = [
            "Maintain current study routine",
            f"Push {report['mastery']['weakest']} mastery above 80%",
            "Try advanced practice problems",
            "Help peers who are struggling",
            "Prepare for upcoming assessments"
        ]
    
    # Study plan summary
    report['study_plan_summary'] = {
        'intensity': study_plan.get('intensity', 'moderate'),
        'total_hours': study_plan.get('total_hours', 14),
        'daily_hours': study_plan.get('daily_hours', 2),
    }
    
    return report


def format_progress_report(report):
    """Format the progress report as a readable string."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  WEEKLY PROGRESS REPORT")
    lines.append(f"  Generated: {report['generated_at']}")
    lines.append(f"{'='*60}")
    
    # Overview
    ov = report['overview']
    lines.append(f"\n  STUDENT: {ov['name']}")
    lines.append(f"  School: {ov['school']} | Age: {ov['age']} | Gender: {ov['gender']}")
    
    # Grades
    g = report['grades']
    lines.append(f"\n  GRADE PROGRESSION:")
    lines.append(f"  G1: {g['G1']}/20 -> G2: {g['G2']}/20 -> G3: {g['G3']}/20")
    lines.append(f"  Trend: {g['trend']} (change: {g['grade_change']:+d})")
    lines.append(f"  Average: {g['average']}/20")
    
    # Mastery
    m = report['mastery']
    lines.append(f"\n  MASTERY LEVELS:")
    lines.append(f"  Overall Mastery: {m['overall']}%")
    lines.append(f"  Strongest: {m['strongest']}")
    lines.append(f"  Weakest: {m['weakest']}")
    if m.get('scores'):
        for concept, score in m['scores'].items():
            bar = '#' * int(score * 20) + '-' * (20 - int(score * 20))
            lines.append(f"    {concept:<15} [{bar}] {score:.0%}")
    
    # Risk
    r = report['risk_assessment']
    status = "AT RISK" if r['is_at_risk'] else "ON TRACK"
    lines.append(f"\n  RISK STATUS: {status} ({r['risk_probability']}% risk)")
    if r['weak_areas']:
        lines.append(f"  Weak Areas:")
        for area in r['weak_areas']:
            lines.append(f"    [{area['severity'].upper()}] {area['area']}: {area['detail']}")
    
    # Goals
    lines.append(f"\n  WEEKLY GOALS:")
    for i, goal in enumerate(report.get('weekly_goals', []), 1):
        lines.append(f"    {i}. {goal}")
    
    # Study plan
    sp = report.get('study_plan_summary', {})
    lines.append(f"\n  STUDY PLAN: {sp.get('intensity', 'moderate').upper()}")
    lines.append(f"  {sp.get('daily_hours', 2)}h/day, {sp.get('total_hours', 14)}h/week")
    
    lines.append(f"\n{'='*60}")
    return '\n'.join(lines)
