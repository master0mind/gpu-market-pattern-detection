"""
Simple Cloud Function for testing deployment
"""

def health_check(request):
    """
    HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`.
    """
    return {
        'status': 'success',
        'message': 'Health check successful!'
    }
