from functools import wraps

def error_handler(func):
    """
    A decorator to gracefully handle exceptions in functions, 
    typically those making API calls.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log or print a user-friendly error message
            print(f"An error occurred in function '{func.__name__}': {e}")
            # Return a default value, like None or an empty dict/list
            return None
    return wrapper
