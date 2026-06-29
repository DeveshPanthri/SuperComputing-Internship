"""
Study Planner
==============
Generates personalized 7-day study plans based on student's
mastery levels, weak areas, and recommended resources.
"""


def generate_study_plan(student_id, mastery_scores, weak_areas, recommendations, student_data=None):
    """
    Generate a personalized 7-day study plan.
    
    Args:
        student_id: Student identifier
        mastery_scores: Dict of concept -> mastery probability
        weak_areas: List of identified weak areas
        recommendations: List of recommended resources
        student_data: Optional dict of student demographic data
    
    Returns:
        Dict containing the 7-day study plan
    """
    # Determine study intensity based on risk level
    risk_score = sum(1 for area in weak_areas if area['severity'] == 'high') * 3 + \
                 sum(1 for area in weak_areas if area['severity'] == 'medium') * 2 + \
                 sum(1 for area in weak_areas if area['severity'] == 'low') * 1
    
    if risk_score >= 6:
        intensity = 'intensive'
        daily_hours = 3
    elif risk_score >= 3:
        intensity = 'moderate'
        daily_hours = 2
    else:
        intensity = 'light'
        daily_hours = 1.5
    
    # Get student study time preference if available
    if student_data and 'studytime' in student_data:
        st = student_data['studytime']
        if st == 1:
            daily_hours = min(daily_hours, 1.5)
        elif st >= 3:
            daily_hours = max(daily_hours, 2.5)
    
    # Build 7-day plan
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Sort weak areas by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_areas = sorted(weak_areas, key=lambda x: severity_order.get(x['severity'], 2))
    
    plan = {
        'student_id': student_id,
        'intensity': intensity,
        'daily_hours': daily_hours,
        'total_hours': daily_hours * 7,
        'days': []
    }
    
    for i, day in enumerate(days):
        day_plan = {
            'day': day,
            'day_number': i + 1,
            'hours': daily_hours,
            'activities': []
        }
        
        if i < 2:  # Mon-Tue: Focus on highest priority weak areas
            if sorted_areas:
                area = sorted_areas[0]
                day_plan['focus'] = f"Core Review: {area['area']}"
                day_plan['activities'] = [
                    f"Review fundamentals of {area['area']} ({area['detail']})",
                    f"Complete practice problems (30 min)",
                ]
                if recommendations:
                    day_plan['activities'].append(
                        f"Watch: {recommendations[0].get('name', 'Recommended Video')}"
                    )
                day_plan['activities'].append("Self-assessment quiz (15 min)")
            else:
                day_plan['focus'] = "General Review"
                day_plan['activities'] = ["Review course materials", "Practice problems"]
                
        elif i < 4:  # Wed-Thu: Secondary weak areas
            if len(sorted_areas) > 1:
                area = sorted_areas[min(1, len(sorted_areas)-1)]
                day_plan['focus'] = f"Targeted Practice: {area['area']}"
                day_plan['activities'] = [
                    f"Deep dive into {area['area']}",
                    "Solve 10 practice problems",
                ]
                if len(recommendations) > 1:
                    day_plan['activities'].append(
                        f"Study: {recommendations[1].get('name', 'Recommended Resource')}"
                    )
                day_plan['activities'].append("Review mistakes from previous days")
            else:
                day_plan['focus'] = "Concept Reinforcement"
                day_plan['activities'] = ["Practice exercises", "Review notes"]
                
        elif i == 4:  # Friday: Integration
            day_plan['focus'] = "Integration & Application"
            day_plan['activities'] = [
                "Combine concepts from the week",
                "Solve mixed-topic problems",
                "Create summary notes",
            ]
            if len(recommendations) > 2:
                day_plan['activities'].append(
                    f"Explore: {recommendations[2].get('name', 'Additional Resource')}"
                )
                
        elif i == 5:  # Saturday: Practice test
            day_plan['focus'] = "Practice Assessment"
            day_plan['hours'] = daily_hours + 0.5
            day_plan['activities'] = [
                "Take a practice test covering all weak areas",
                "Time yourself to simulate exam conditions",
                "Review all incorrect answers",
                "Update study notes with corrections"
            ]
            
        else:  # Sunday: Light review + planning
            day_plan['focus'] = "Review & Plan Ahead"
            day_plan['hours'] = max(1, daily_hours - 0.5)
            day_plan['activities'] = [
                "Light review of the week's material",
                "Organize notes and flashcards",
                "Set goals for next week",
                "Rest and recharge"
            ]
        
        # Add mastery context
        if mastery_scores:
            weakest = min(mastery_scores, key=mastery_scores.get)
            strongest = max(mastery_scores, key=mastery_scores.get)
            day_plan['mastery_tip'] = f"Focus more on '{weakest}' (mastery: {mastery_scores[weakest]:.0%})"
        
        plan['days'].append(day_plan)
    
    return plan


def format_study_plan(plan):
    """Format the study plan as a readable string."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  PERSONALIZED 7-DAY STUDY PLAN")
    lines.append(f"  Student ID: {plan['student_id']}")
    lines.append(f"  Intensity: {plan['intensity'].upper()}")
    lines.append(f"  Total Hours: {plan['total_hours']:.1f}h over 7 days")
    lines.append(f"{'='*60}")
    
    for day in plan['days']:
        lines.append(f"\n  --- {day['day']} (Day {day['day_number']}) ---")
        lines.append(f"  Focus: {day.get('focus', 'General Study')}")
        lines.append(f"  Duration: {day['hours']:.1f} hours")
        lines.append(f"  Activities:")
        for activity in day.get('activities', []):
            lines.append(f"    * {activity}")
        if 'mastery_tip' in day:
            lines.append(f"  Tip: {day['mastery_tip']}")
    
    lines.append(f"\n{'='*60}")
    return '\n'.join(lines)
