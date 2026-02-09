import praw
import os
import re
from collections import Counter

class FacilityWatch:
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Scrapes trucking discussions for facility-specific bottlenecks.
        Authentication is optional for the mock_scan method, but required for real Reddit access.
        """
        self.reddit = None
        if client_id and client_secret:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent or "AgriAlpha Bot v0.1"
            )
            
        self.targets = ["JBS", "Tyson", "Cargill", "National Beef"]
        self.keywords = ["wait time", "lumper", "line out to road", "broken down", "avoid", "hours"]
        
    def mock_scan(self):
        """
        Simulates scraping r/Truckers for demonstration purposes.
        """
        mock_posts = [
            "Avoid JBS in Grand Island, line is out to the highway, 8 hour wait.",
            "Tyson in Amarillo is running smooth today.",
            "Anyone stuck at Cargill? Lumper service is down.",
            "Just getting started trucking, any tips?",
            "JBS Green Bay is a mess, computers down, trucks parked everywhere."
        ]
        
        findings = []
        
        for post in mock_posts:
            # Case insensitive search
            text = post.lower()
            
            # Check for Target Facility
            detected_facility = None
            for facility in self.targets:
                if facility.lower() in text:
                    detected_facility = facility
                    break
            
            if detected_facility:
                # Check for negative sentiment/keywords
                for kw in self.keywords:
                    if kw in text:
                        findings.append({
                            "facility": detected_facility,
                            "signal": "HIGH_WAIT",
                            "raw_text": post,
                            "keyword": kw
                        })
                        break
                        
        return findings

    def analyze_sentiment(self):
        findings = self.mock_scan()
        
        # Aggregation
        counts = Counter([f['facility'] for f in findings])
        
        alerts = []
        for facility, count in counts.items():
            if count >= 2: # Threshold for signal
                alerts.append({
                    "facility": facility,
                    "alert_level": "CRITICAL",
                    "mentions": count,
                    "msg": f"Multiple driver reports of delays at {facility}"
                })
        
        return alerts

if __name__ == "__main__":
    watcher = FacilityWatch()
    results = watcher.analyze_sentiment()
    for res in results:
        print(f"🏭 FACILITY ALERT: {res['facility']} [{res['alert_level']}]")
        print(f"   {res['msg']}")
