# Not in use
metrics = {
    "total_requests": 0,
    "failed_requests": 0,
    "response_times": []
}


'''def update_metrics(success: bool, response_time: float):
    """
    Updates system-level monitoring metrics.

    Parameters:
    - success (bool): Whether the request was processed successfully
    - response_time (float): Time taken to process the request (in seconds)

    Tracks:
    - Total number of requests
    - Failed requests
    - Response time distribution

    Why important:
    - Helps monitor system health
    - Detects failures and latency issues
    - Useful for dashboards (Grafana/Prometheus later)
    """

    metrics["total_requests"] += 1

    if not success:
        metrics["failed_requests"] += 1

    metrics["response_times"].append(response_time)
'''

'''
def get_avg_response_time():
    """
    Computes average response time.

    Returns:
    - float: Average latency in seconds

    Why important:
    - Measures system performance
    - Helps identify bottlenecks (LLM vs retrieval)
    """

    if not metrics["response_times"]:
        return 0

    return sum(metrics["response_times"]) / len(metrics["response_times"])'''



import time

def start_timer():
    return time.time()

def end_timer(start_time):
    return round(time.time() - start_time, 3)