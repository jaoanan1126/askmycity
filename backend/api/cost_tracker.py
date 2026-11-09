"""
Cost tracking and budget management
Monitor LLM spending and provide optimization insights
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import threading

logger = logging.getLogger(__name__)

class CostTracker:
    """
    Track LLM costs and provide budget insights
    """
    
    # Pricing in USD per million tokens
    PRICING = {
        "claude-haiku-4-5-20251001": {
            "input": 1.0,   # $1 per 1M input tokens
            "output": 5.0   # $5 per 1M output tokens
        },
        "claude-3-5-sonnet-20241022": {
            "input": 3.0,   # $3 per 1M input tokens
            "output": 15.0  # $15 per 1M output tokens
        }
    }
    
    def __init__(self, data_file: str = "cost_data.json"):
        """Initialize cost tracker"""
        self.data_file = Path(data_file)
        self.data = self._load_data()
        # track_request is called from RAG worker threads; serialize
        # updates and file writes so concurrent requests don't clobber them.
        self._lock = threading.Lock()
        logger.info("✓ Cost tracker initialized")
    
    def _load_data(self) -> Dict[str, Any]:
        """Load cost data from file"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        
        return {
            "requests": [],
            "daily_totals": {},
            "budget": {
                "daily_limit": None,
                "monthly_limit": None,
                "alerts_enabled": False
            }
        }
    
    def _save_data(self) -> None:
        """Save cost data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate cost for a request
        
        Returns:
            {
                "input_cost": 0.006,
                "output_cost": 0.0075,
                "total_cost": 0.0135
            }
        """
        pricing = self.PRICING.get(model, self.PRICING["claude-3-5-sonnet-20241022"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }
    
    def track_request(
        self,
        conversation_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Track a request and its cost"""
        
        # Calculate cost
        costs = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Apply cache discount (cached requests are ~90% cheaper)
        if cached:
            costs["total_cost"] *= 0.1  # 90% discount
            costs["cached"] = True
        
        # Store request
        request_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_id": conversation_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached": cached,
            **costs,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.data["requests"].append(request_data)

            # Update daily totals
            date_key = datetime.utcnow().strftime("%Y-%m-%d")
            if date_key not in self.data["daily_totals"]:
                self.data["daily_totals"][date_key] = {
                    "total_cost": 0,
                    "total_requests": 0,
                    "total_tokens": 0
                }

            self.data["daily_totals"][date_key]["total_cost"] += costs["total_cost"]
            self.data["daily_totals"][date_key]["total_requests"] += 1
            self.data["daily_totals"][date_key]["total_tokens"] += input_tokens + output_tokens

            self._save_data()
        
        return costs
    
    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost summary for last N days
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter recent requests
        recent_requests = [
            r for r in self.data["requests"]
            if datetime.fromisoformat(r["timestamp"]) > cutoff_date
        ]
        
        if not recent_requests:
            return {
                "period_days": days,
                "total_cost": 0,
                "total_requests": 0,
                "total_tokens": 0,
                "average_cost_per_request": 0,
                "cache_savings": 0
            }
        
        # Calculate totals
        total_cost = sum(r["total_cost"] for r in recent_requests)
        total_requests = len(recent_requests)
        total_tokens = sum(r["total_tokens"] for r in recent_requests)
        cached_requests = sum(1 for r in recent_requests if r.get("cached", False))
        
        # Estimate cache savings (cached requests save ~90%)
        cache_savings = cached_requests * 0.013 * 0.9  # Rough estimate
        
        return {
            "period_days": days,
            "total_cost": round(total_cost, 4),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "average_cost_per_request": round(total_cost / total_requests, 6) if total_requests > 0 else 0,
            "cached_requests": cached_requests,
            "cache_hit_rate": round((cached_requests / total_requests) * 100, 2) if total_requests > 0 else 0,
            "cache_savings": round(cache_savings, 4)
        }
    
    def get_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed cost breakdown
        """
        # By day
        daily_costs = {}
        for date, data in self.data["daily_totals"].items():
            daily_costs[date] = {
                "cost": round(data["total_cost"], 4),
                "requests": data["total_requests"],
                "tokens": data["total_tokens"]
            }
        
        # By conversation (top 10 most expensive)
        conversation_costs = defaultdict(lambda: {"cost": 0, "requests": 0})
        for req in self.data["requests"]:
            conv_id = req["conversation_id"]
            conversation_costs[conv_id]["cost"] += req["total_cost"]
            conversation_costs[conv_id]["requests"] += 1
        
        top_conversations = sorted(
            conversation_costs.items(),
            key=lambda x: x[1]["cost"],
            reverse=True
        )[:10]
        
        return {
            "daily_costs": daily_costs,
            "top_conversations": [
                {
                    "conversation_id": conv_id,
                    "total_cost": round(data["cost"], 4),
                    "requests": data["requests"]
                }
                for conv_id, data in top_conversations
            ]
        }
    
    def get_predictions(self) -> Dict[str, Any]:
        """
        Predict future costs based on current usage
        """
        # Get last 7 days average
        last_week = self.get_summary(days=7)
        daily_average = last_week["total_cost"] / 7 if last_week["total_cost"] > 0 else 0
        
        # Project forward
        return {
            "daily_average": round(daily_average, 4),
            "projected_weekly": round(daily_average * 7, 2),
            "projected_monthly": round(daily_average * 30, 2),
            "projected_yearly": round(daily_average * 365, 2)
        }
    
    def get_optimization_tips(self) -> List[str]:
        """
        Provide cost optimization suggestions based on usage patterns
        """
        tips = []
        
        summary = self.get_summary(days=7)
        
        # Check cache hit rate
        if summary["cache_hit_rate"] < 30:
            tips.append(
                f"💰 Low cache hit rate ({summary['cache_hit_rate']}%). "
                "Enable caching to save up to 90% on repeat questions."
            )
        
        # Check average cost
        if summary["average_cost_per_request"] > 0.02:
            tips.append(
                f"💰 High cost per request (${summary['average_cost_per_request']:.4f}). "
                "Consider reducing context size or using fewer retrieved documents."
            )
        
        # Check daily spending
        predictions = self.get_predictions()
        if predictions["projected_monthly"] > 100:
            tips.append(
                f"💰 High projected monthly cost (${predictions['projected_monthly']:.2f}). "
                "Consider implementing request limits or optimizing prompts."
            )
        
        if not tips:
            tips.append("✅ Your costs look optimized! Keep up the good work.")
        
        return tips
    
    def set_budget(
        self,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        alerts_enabled: bool = True
    ):
        """Set budget limits"""
        self.data["budget"] = {
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "alerts_enabled": alerts_enabled
        }
        self._save_data()
    
    def check_budget_alert(self) -> Optional[Dict[str, Any]]:
        """Check if budget limits are exceeded"""
        budget = self.data["budget"]
        
        if not budget["alerts_enabled"]:
            return None
        
        alerts = []
        
        # Check daily limit
        if budget["daily_limit"]:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            today_cost = self.data["daily_totals"].get(today, {}).get("total_cost", 0)
            
            if today_cost > budget["daily_limit"]:
                alerts.append({
                    "type": "daily",
                    "limit": budget["daily_limit"],
                    "current": today_cost,
                    "exceeded_by": today_cost - budget["daily_limit"]
                })
        
        # Check monthly limit
        if budget["monthly_limit"]:
            month_start = datetime.utcnow().replace(day=1).strftime("%Y-%m-%d")
            month_cost = sum(
                data["total_cost"]
                for date, data in self.data["daily_totals"].items()
                if date >= month_start
            )
            
            if month_cost > budget["monthly_limit"]:
                alerts.append({
                    "type": "monthly",
                    "limit": budget["monthly_limit"],
                    "current": month_cost,
                    "exceeded_by": month_cost - budget["monthly_limit"]
                })
        
        return alerts if alerts else None

# Global cost tracker instance
cost_tracker = CostTracker()