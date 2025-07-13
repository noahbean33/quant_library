import pandas as pd
from typing import List, Dict
from ..analysis import calculate_ratios

def screen_stocks(tickers: List[str], criteria: Dict[str, float]) -> pd.DataFrame:
    """
    Screens a list of stocks based on user-defined financial criteria.

    Args:
        tickers (List[str]): A list of stock ticker symbols to screen.
        criteria (Dict[str, float]): A dictionary defining the screening criteria.
            Keys should be in the format '{ratio_name}_{min|max}'.
            Example: {'pe_ratio_max': 20, 'return_on_equity_min': 0.15}

    Returns:
        pd.DataFrame: A DataFrame containing the stocks that pass the screen,
                      along with their key ratios. Returns an empty DataFrame
                      if no stocks pass the criteria.
    """
    screened_list = []
    
    for ticker in tickers:
        print(f"Screening {ticker}...")
        ratios = calculate_ratios(ticker)
        
        if not ratios or not any(ratios.values()):
            continue
            
        passes_screen = True
        for key, threshold in criteria.items():
            parts = key.split('_')
            if len(parts) < 2 or parts[-1] not in ['min', 'max']:
                print(f"Warning: Invalid criteria key format '{key}'. Should be 'ratio_name_min' or 'ratio_name_max'.")
                continue

            metric_name = '_'.join(parts[:-1])
            condition = parts[-1]
            
            value = ratios.get(metric_name)
            
            if value is None:
                passes_screen = False
                break
            
            if condition == 'max' and value > threshold:
                passes_screen = False
                break
            elif condition == 'min' and value < threshold:
                passes_screen = False
                break
        
        if passes_screen:
            ratios['ticker'] = ticker
            screened_list.append(ratios)
            
    if not screened_list:
        print("No stocks passed the screening criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(screened_list)
    if not df.empty:
        # Reorder columns to have 'ticker' first
        cols = ['ticker'] + [col for col in df.columns if col != 'ticker']
        df = df[cols]
    return df
