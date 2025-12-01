import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def rankings_to_pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rankings data to pairwise comparisons for Bradley-Terry model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where columns are tests, rows are ranking positions,
        values are benchmarker IDs
    
    Returns:
    --------
    pd.DataFrame with columns:
        - winner: ID of winning benchmarker
        - loser: ID of losing benchmarker
        - test: name of the test
    """
    comparisons = []
    
    for test_name in df.columns:
        test_rankings = df[test_name].dropna()
        
        # For each position, that benchmarker beats all lower positions
        for i, winner_id in enumerate(test_rankings):
            for j in range(i + 1, len(test_rankings)):
                loser_id = test_rankings.iloc[j]
                if not pd.isna(winner_id) and not pd.isna(loser_id):
                    comparisons.append({
                        'winner': int(winner_id),
                        'loser': int(loser_id),
                        'test': test_name
                    })
    
    return pd.DataFrame(comparisons)


def fit_bradley_terry_model(pairwise_df: pd.DataFrame, 
                           all_benchmarkers: List[int]) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit Bradley-Terry model using logistic regression.
    
    Parameters:
    -----------
    pairwise_df : pd.DataFrame
        DataFrame with pairwise comparisons
    all_benchmarkers : List[int]
        List of all benchmarker IDs
    
    Returns:
    --------
    Fitted statsmodels GLM model
    """
    # Create design matrix for Bradley-Terry model
    # For each comparison, create indicators for winner and loser
    n_comparisons = len(pairwise_df)
    n_benchmarkers = len(all_benchmarkers)
    
    # Create dictionary to map benchmarker ID to index
    benchmarker_to_idx = {bid: idx for idx, bid in enumerate(all_benchmarkers)}
    
    # Create design matrix (difference in indicators)
    # We'll use the last benchmarker as reference (no coefficient for it)
    X = np.zeros((n_comparisons, n_benchmarkers - 1))
    
    for i, row in pairwise_df.iterrows():
        winner_idx = benchmarker_to_idx[row['winner']]
        loser_idx = benchmarker_to_idx[row['loser']]
        
        # Set +1 for winner, -1 for loser (if not reference category)
        if winner_idx < n_benchmarkers - 1:
            X[i, winner_idx] = 1
        if loser_idx < n_benchmarkers - 1:
            X[i, loser_idx] = -1
    
    # Response is always 1 (winner wins)
    y = np.ones(n_comparisons)
    
    # Add column names for clarity
    feature_names = [f'ability_{bid}' for bid in all_benchmarkers[:-1]]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Fit logistic regression
    model = sm.GLM(y, X_df, family=sm.families.Binomial())
    result = model.fit()
    
    return result, benchmarker_to_idx


def two_stage_fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float], float]:
    """
    Implement two-stage FDR procedure (Benjamini-Krieger-Yekutieli).
    
    This provides more power than single-stage Benjamini-Hochberg while
    still controlling the false discovery rate.
    
    Parameters:
    -----------
    p_values : List[float]
        Raw p-values to correct
    alpha : float
        Desired FDR level
    
    Returns:
    --------
    Tuple of:
        - reject: List[bool] - whether to reject each null hypothesis
        - adjusted_p: List[float] - adjusted p-values
        - alpha_star: float - adjusted alpha level used
    """
    if len(p_values) == 0:
        return [], [], alpha
    
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Stage 1: Apply Benjamini-Hochberg to estimate number of true nulls
    reject_stage1, pvals_corrected_stage1, _, _ = multipletests(
        p_values, alpha=alpha, method='fdr_bh'
    )
    
    # Estimate proportion of true null hypotheses
    R = np.sum(reject_stage1)  # Number of rejections
    if R == 0:
        # No rejections in stage 1, assume all are true nulls
        m0 = n
    else:
        # Estimate m0 (number of true nulls)
        m0 = n - R
    
    # Stage 2: Apply BH procedure with adjusted alpha
    # Adjusted alpha incorporates the estimate of m0
    alpha_star = alpha * n / max(m0, 1)
    alpha_star = min(alpha_star, 1.0)  # Ensure alpha_star <= 1
    
    # Apply Benjamini-Hochberg with adjusted alpha
    reject_stage2, pvals_corrected_stage2, _, _ = multipletests(
        p_values, alpha=alpha_star, method='fdr_bh'
    )
    
    return reject_stage2.tolist(), pvals_corrected_stage2.tolist(), alpha_star


def bradley_terry_analysis(
    df: pd.DataFrame,
    target_benchmarker: int,
    test_type: str,
    compare_against: List[int],
    confidence_level: float = 0.95,
    fdr_level: float = 0.05,
    use_fdr_correction: bool = True,
    n_bootstrap: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Perform Bradley-Terry analysis with two-stage FDR correction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where columns are tests/experiments, rows are ranking positions,
        values are benchmarker IDs that achieved that rank
    target_benchmarker : int
        The benchmarker ID to analyze
    test_type : str
        Type of test: 'superiority' or 'inferiority'
        - 'superiority': Test if target is better than compare_against
        - 'inferiority': Test if target is worse than compare_against
    compare_against : List[int]
        List of benchmarker IDs to compare the target against
    confidence_level : float
        Confidence level for intervals (default 0.95)
    fdr_level : float
        False discovery rate level for multiple testing (default 0.05)
    use_fdr_correction : bool
        Whether to apply two-stage FDR correction (default True)
    n_bootstrap : int
        Number of bootstrap samples for additional inference
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    Dict containing:
        - 'model': Fitted model object
        - 'abilities': Estimated ability parameters for all benchmarkers
        - 'test_results': Results of hypothesis tests with FDR correction
        - 'conclusions': Text conclusions
        - 'test_type': The type of test performed
    """
    
    # Validate test_type
    test_type = test_type.lower()
    if test_type not in ['superiority', 'inferiority']:
        raise ValueError("test_type must be 'superiority' or 'inferiority'")
    
    # Get all unique benchmarkers
    all_benchmarkers = set()
    for col in df.columns:
        all_benchmarkers.update(df[col].dropna().astype(int).unique())
    all_benchmarkers = sorted(list(all_benchmarkers))
    
    if verbose:
        print(f"Found {len(all_benchmarkers)} benchmarkers: {all_benchmarkers}")
        print(f"Analyzing benchmarker {target_benchmarker}")
        print(f"Test type: {test_type.upper()}")
        print(f"Comparing against: {compare_against}")
        if use_fdr_correction:
            print(f"Using two-stage FDR correction at level {fdr_level}")
        print("\n" + "="*60)
    
    # Convert to pairwise comparisons
    pairwise_df = rankings_to_pairwise_comparisons(df)
    
    if verbose:
        print(f"Created {len(pairwise_df)} pairwise comparisons from rankings")
    
    # Fit Bradley-Terry model
    model_result, benchmarker_to_idx = fit_bradley_terry_model(pairwise_df, all_benchmarkers)
    
    # Extract abilities (including reference category with ability = 0)
    n_benchmarkers = len(all_benchmarkers)
    abilities = np.zeros(n_benchmarkers)
    se_abilities = np.zeros(n_benchmarkers)
    
    # Fill in estimated abilities
    for i in range(n_benchmarkers - 1):
        abilities[i] = model_result.params[i]
        se_abilities[i] = model_result.bse[i]
    
    # Reference category has ability 0 and SE calculated differently
    # For the reference category, we can estimate SE through the covariance matrix
    
    # Create abilities DataFrame
    abilities_df = pd.DataFrame({
        'benchmarker_id': all_benchmarkers,
        'ability': abilities,
        'se': se_abilities,
        'lower_ci': abilities - stats.norm.ppf((1 + confidence_level)/2) * se_abilities,
        'upper_ci': abilities + stats.norm.ppf((1 + confidence_level)/2) * se_abilities
    })
    
    if verbose:
        print("\nBradley-Terry Ability Estimates:")
        print("(Higher ability = better performance)")
        print("-"*40)
        print(abilities_df.to_string(index=False))
    
    # Perform hypothesis tests
    target_idx = benchmarker_to_idx[target_benchmarker]
    target_ability = abilities[target_idx]
    target_se = se_abilities[target_idx]
    
    results = {
        'model': model_result,
        'abilities': abilities_df,
        'pairwise_comparisons': pairwise_df,
        'benchmarker_to_idx': benchmarker_to_idx,
        'fdr_level': fdr_level,
        'use_fdr_correction': use_fdr_correction,
        'test_type': test_type,
        'compare_against': compare_against
    }
    
    # Perform the appropriate test based on test_type
    if test_type == 'superiority':
        # Test superiority
        superior_tests = {}
        superior_p_values = []
        superior_ids = []
        
        for comp_id in compare_against:  # FIXED: Use compare_against
            if comp_id in benchmarker_to_idx:
                comp_idx = benchmarker_to_idx[comp_id]
                comp_ability = abilities[comp_idx]
                comp_se = se_abilities[comp_idx]
                
                # Calculate difference and its SE
                diff = target_ability - comp_ability
                se_diff = np.sqrt(target_se**2 + comp_se**2)
                
                # Z-test for difference
                z_stat = diff / se_diff if se_diff > 0 else 0
                p_value = 1 - stats.norm.cdf(z_stat)  # One-tailed test for superiority
                
                # Probability that target > comp (using normal approximation)
                prob_superior = stats.norm.cdf(diff / se_diff) if se_diff > 0 else 0.5
                
                superior_tests[comp_id] = {
                    'difference': diff,
                    'se_diff': se_diff,
                    'z_statistic': z_stat,
                    'p_value_raw': p_value,
                    'prob_superior': prob_superior
                }
                
                superior_p_values.append(p_value)
                superior_ids.append(comp_id)
        
        # Apply two-stage FDR correction
        if use_fdr_correction and len(superior_p_values) > 0:
            reject_superior, adjusted_p_superior, alpha_star_superior = two_stage_fdr_correction(
                superior_p_values, alpha=fdr_level
            )
            for i, comp_id in enumerate(superior_ids):
                superior_tests[comp_id]['p_value_adjusted'] = adjusted_p_superior[i]
                superior_tests[comp_id]['reject_null'] = reject_superior[i]
                superior_tests[comp_id]['significant'] = reject_superior[i]
        else:
            for comp_id in superior_ids:
                superior_tests[comp_id]['p_value_adjusted'] = superior_tests[comp_id]['p_value_raw']
                superior_tests[comp_id]['reject_null'] = superior_tests[comp_id]['p_value_raw'] < (1 - confidence_level)
                superior_tests[comp_id]['significant'] = superior_tests[comp_id]['p_value_raw'] < (1 - confidence_level)
        
        results['test_results'] = superior_tests
        inferior_tests = {}  # Empty for superiority test
        
        if verbose and superior_tests:
            print("\n" + "="*60)
            print(f"SUPERIORITY TEST: Is Benchmarker {target_benchmarker} SUPERIOR to {compare_against}?")
            print("-"*60)
            if use_fdr_correction:
                print(f"Two-stage FDR correction applied (FDR level = {fdr_level})")
                if len(superior_p_values) > 0:
                    print(f"Adjusted alpha level: {alpha_star_superior:.4f}")
            print("-"*60)
            
            for comp_id in compare_against:
                if comp_id in superior_tests:
                    test = superior_tests[comp_id]
                    print(f"\nBenchmarker {target_benchmarker} vs {comp_id}:")
                    print(f"  Ability difference: {test['difference']:.3f} ± {test['se_diff']:.3f}")
                    print(f"  P(B{target_benchmarker} > B{comp_id}): {test['prob_superior']:.3f}")
                    print(f"  P-value (raw): {test['p_value_raw']:.4f}")
                    if use_fdr_correction:
                        print(f"  P-value (FDR-adjusted): {test['p_value_adjusted']:.4f}")
                        print(f"  Reject null (FDR): {test['reject_null']}")
                    else:
                        print(f"  Significant at {confidence_level:.0%} level: {test['significant']}")
    
    elif test_type == 'inferiority':
        # Test inferiority
        inferior_tests = {}
        inferior_p_values = []
        inferior_ids = []
        
        for comp_id in compare_against:  # FIXED: Use compare_against
            if comp_id in benchmarker_to_idx:
                comp_idx = benchmarker_to_idx[comp_id]
                comp_ability = abilities[comp_idx]
                comp_se = se_abilities[comp_idx]
                
                # Calculate difference and its SE
                diff = target_ability - comp_ability
                se_diff = np.sqrt(target_se**2 + comp_se**2)
                
                # Z-test for difference
                z_stat = diff / se_diff if se_diff > 0 else 0
                p_value = stats.norm.cdf(z_stat)  # One-tailed test for inferiority
                
                # Probability that target < comp
                prob_inferior = stats.norm.cdf(-diff / se_diff) if se_diff > 0 else 0.5
                
                inferior_tests[comp_id] = {
                    'difference': diff,
                    'se_diff': se_diff,
                    'z_statistic': z_stat,
                    'p_value_raw': p_value,
                    'prob_inferior': prob_inferior
                }
                
                inferior_p_values.append(p_value)
                inferior_ids.append(comp_id)
        
        # Apply two-stage FDR correction
        if use_fdr_correction and len(inferior_p_values) > 0:
            reject_inferior, adjusted_p_inferior, alpha_star_inferior = two_stage_fdr_correction(
                inferior_p_values, alpha=fdr_level
            )
            for i, comp_id in enumerate(inferior_ids):
                inferior_tests[comp_id]['p_value_adjusted'] = adjusted_p_inferior[i]
                inferior_tests[comp_id]['reject_null'] = reject_inferior[i]
                inferior_tests[comp_id]['significant'] = reject_inferior[i]
        else:
            for comp_id in inferior_ids:
                inferior_tests[comp_id]['p_value_adjusted'] = inferior_tests[comp_id]['p_value_raw']
                inferior_tests[comp_id]['reject_null'] = inferior_tests[comp_id]['p_value_raw'] < (1 - confidence_level)
                inferior_tests[comp_id]['significant'] = inferior_tests[comp_id]['p_value_raw'] < (1 - confidence_level)
        
        results['test_results'] = inferior_tests
        superior_tests = {}  # Empty for inferiority test
        
        if verbose and inferior_tests:
            print("\n" + "="*60)
            print(f"INFERIORITY TEST: Is Benchmarker {target_benchmarker} INFERIOR to {compare_against}?")
            print("-"*60)
            if use_fdr_correction:
                print(f"Two-stage FDR correction applied (FDR level = {fdr_level})")
                if len(inferior_p_values) > 0:
                    print(f"Adjusted alpha level: {alpha_star_inferior:.4f}")
            print("-"*60)
            
            for comp_id in compare_against:
                if comp_id in inferior_tests:
                    test = inferior_tests[comp_id]
                    print(f"\nBenchmarker {target_benchmarker} vs {comp_id}:")
                    print(f"  Ability difference: {test['difference']:.3f} ± {test['se_diff']:.3f}")
                    print(f"  P(B{target_benchmarker} < B{comp_id}): {test['prob_inferior']:.3f}")
                    print(f"  P-value (raw): {test['p_value_raw']:.4f}")
                    if use_fdr_correction:
                        print(f"  P-value (FDR-adjusted): {test['p_value_adjusted']:.4f}")
                        print(f"  Reject null (FDR): {test['reject_null']}")
                    else:
                        print(f"  Significant at {confidence_level:.0%} level: {test['significant']}")
    
    # Store both test results (one will be empty)
    results['superior_tests'] = superior_tests
    results['inferior_tests'] = inferior_tests
    
    # Generate conclusions based on FDR-corrected results
    conclusions = []
    
    if test_type == 'superiority':
        # Check superiority (using FDR-corrected results)
        all_superior_sig = all(test['significant'] for test in superior_tests.values()) if superior_tests else False
        any_superior_sig = any(test['significant'] for test in superior_tests.values()) if superior_tests else False
        
        if all_superior_sig and superior_tests:
            conclusions.append(f"✓ Significant evidence (FDR-controlled at {fdr_level}) that Benchmarker {target_benchmarker} is SUPERIOR to all of {compare_against}")
        elif any_superior_sig and superior_tests:
            sig_ones = [bid for bid, test in superior_tests.items() if test['significant']]
            conclusions.append(f"✓ Significant evidence (FDR-controlled) that Benchmarker {target_benchmarker} is superior to {sig_ones}")
        else:
            conclusions.append(f"✗ No significant evidence for superiority over {compare_against} after FDR correction")
    
    elif test_type == 'inferiority':
        # Check inferiority (using FDR-corrected results)
        all_inferior_sig = all(test['significant'] for test in inferior_tests.values()) if inferior_tests else False
        any_inferior_sig = any(test['significant'] for test in inferior_tests.values()) if inferior_tests else False
        
        if all_inferior_sig and inferior_tests:
            conclusions.append(f"✓ Significant evidence (FDR-controlled at {fdr_level}) that Benchmarker {target_benchmarker} is INFERIOR to all of {compare_against}")
        elif any_inferior_sig and inferior_tests:
            sig_ones = [bid for bid, test in inferior_tests.items() if test['significant']]
            conclusions.append(f"✓ Significant evidence (FDR-controlled) that Benchmarker {target_benchmarker} is inferior to {sig_ones}")
        else:
            conclusions.append(f"✗ No significant evidence for inferiority to {compare_against} after FDR correction")
    
    results['conclusions'] = conclusions
    
    if verbose:
        print("\n" + "="*60)
        print("CONCLUSIONS (with FDR correction)")
        print("="*60)
        for conclusion in conclusions:
            print(conclusion)
        
        print("\n" + "="*60)
        print("MODEL DIAGNOSTICS")
        print("="*60)
        print(f"AIC: {model_result.aic:.2f}")
        print(f"BIC: {model_result.bic:.2f}")
        print(f"Log-Likelihood: {model_result.llf:.2f}")
        print(f"Pseudo R-squared: {1 - model_result.deviance / model_result.null_deviance:.4f}")
    
    return results


def plot_bradley_terry_results(results: Dict, target_benchmarker: int,
                              figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create visualizations for Bradley-Terry analysis results with FDR correction.
    
    Parameters:
    -----------
    results : Dict
        Results from bradley_terry_analysis
    target_benchmarker : int
        The analyzed benchmarker
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    matplotlib Figure object
    """
    abilities_df = results['abilities']
    superior_tests = results.get('superior_tests', {})
    inferior_tests = results.get('inferior_tests', {})
    use_fdr = results.get('use_fdr_correction', False)
    test_type = results.get('test_type', 'unknown')
    compare_against = results.get('compare_against', [])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Ability estimates with confidence intervals
    ax = axes[0, 0]
    
    # Sort by ability for better visualization
    abilities_sorted = abilities_df.sort_values('ability', ascending=True)
    
    # Highlight target benchmarker
    colors = ['lightblue' if bid == target_benchmarker else 'gray' 
              for bid in abilities_sorted['benchmarker_id']]
    
    # Create horizontal bar plot with error bars
    y_pos = np.arange(len(abilities_sorted))
    ax.barh(y_pos, abilities_sorted['ability'], xerr=abilities_sorted['se']*1.96,
            color=colors, alpha=0.7, capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'B{bid}' for bid in abilities_sorted['benchmarker_id']])
    ax.set_xlabel('Ability Estimate (±95% CI)')
    ax.set_title('Bradley-Terry Ability Estimates')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Test results (superiority or inferiority)
    ax = axes[0, 1]
    
    if test_type == 'superiority' and superior_tests:
        benchmarkers = list(superior_tests.keys())
        probs = [test['prob_superior'] for test in superior_tests.values()]
        
        if use_fdr:
            significant = [test.get('reject_null', False) for test in superior_tests.values()]
            p_values_display = [test['p_value_adjusted'] for test in superior_tests.values()]
            title_suffix = " (FDR-corrected)"
        else:
            significant = [test['significant'] for test in superior_tests.values()]
            p_values_display = [test['p_value_raw'] for test in superior_tests.values()]
            title_suffix = ""
        
        colors = ['green' if sig else 'red' for sig in significant]
        
        bars = ax.bar(range(len(benchmarkers)), probs, color=colors, alpha=0.7)
        ax.set_xticks(range(len(benchmarkers)))
        ax.set_xticklabels([f'B{bid}' for bid in benchmarkers])
        ax.set_ylabel('P(Target > Benchmarker)')
        ax.set_title(f'Superiority Tests{title_suffix}')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='95% threshold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, p_val, sig) in enumerate(zip(bars, p_values_display, significant)):
            symbol = "**" if sig else "ns"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{symbol}\np={p_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    elif test_type == 'inferiority' and inferior_tests:
        benchmarkers = list(inferior_tests.keys())
        probs = [test['prob_inferior'] for test in inferior_tests.values()]
        
        if use_fdr:
            significant = [test.get('reject_null', False) for test in inferior_tests.values()]
            p_values_display = [test['p_value_adjusted'] for test in inferior_tests.values()]
            title_suffix = " (FDR-corrected)"
        else:
            significant = [test['significant'] for test in inferior_tests.values()]
            p_values_display = [test['p_value_raw'] for test in inferior_tests.values()]
            title_suffix = ""
        
        colors = ['green' if sig else 'red' for sig in significant]
        
        bars = ax.bar(range(len(benchmarkers)), probs, color=colors, alpha=0.7)
        ax.set_xticks(range(len(benchmarkers)))
        ax.set_xticklabels([f'B{bid}' for bid in benchmarkers])
        ax.set_ylabel('P(Target < Benchmarker)')
        ax.set_title(f'Inferiority Tests{title_suffix}')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='95% threshold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, p_val, sig) in enumerate(zip(bars, p_values_display, significant)):
            symbol = "**" if sig else "ns"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{symbol}\np={p_val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No test results to display', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Plot 3: Empty or comparison plot
    ax = axes[1, 0]
    ax.text(0.5, 0.5, f'Test Type: {test_type.upper()}\nTarget: B{target_benchmarker}\nCompare Against: {compare_against}',
            ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Plot 4: P-value comparison (raw vs adjusted)
    ax = axes[1, 1]
    
    all_tests = superior_tests if test_type == 'superiority' else inferior_tests
    
    if all_tests and use_fdr:
        all_tests_labels = []
        raw_p_values = []
        adjusted_p_values = []
        
        for bid, test in all_tests.items():
            symbol = '>' if test_type == 'superiority' else '<'
            all_tests_labels.append(f'B{target_benchmarker}{symbol}B{bid}')
            raw_p_values.append(test['p_value_raw'])
            adjusted_p_values.append(test['p_value_adjusted'])
        
        x_pos = np.arange(len(all_tests_labels))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, raw_p_values, width, label='Raw p-values', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, adjusted_p_values, width, label='FDR-adjusted p-values', alpha=0.7)
        
        ax.set_xlabel('Comparison')
        ax.set_ylabel('P-value')
        ax.set_title('P-value Comparison: Raw vs FDR-adjusted')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_tests_labels, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=results.get('fdr_level', 0.05), color='red', linestyle='--', 
                  label=f'FDR level ({results.get("fdr_level", 0.05)})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        if raw_p_values and adjusted_p_values:
            ax.set_ylim(0, max(max(raw_p_values), max(adjusted_p_values)) * 1.1)
    elif all_tests and not use_fdr:
        all_tests_labels = []
        raw_p_values = []
        
        for bid, test in all_tests.items():
            symbol = '>' if test_type == 'superiority' else '<'
            all_tests_labels.append(f'B{target_benchmarker}{symbol}B{bid}')
            raw_p_values.append(test['p_value_raw'])
        
        bars = ax.bar(range(len(all_tests_labels)), raw_p_values, alpha=0.7)
        ax.set_xlabel('Comparison')
        ax.set_ylabel('P-value')
        ax.set_title('P-values (No Multiple Testing Correction)')
        ax.set_xticks(range(len(all_tests_labels)))
        ax.set_xticklabels(all_tests_labels, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No p-values to display', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def bootstrap_bradley_terry(df: pd.DataFrame, target_benchmarker: int,
                           test_type: str, compare_against: List[int],
                           n_bootstrap: int = 1000, confidence_level: float = 0.95,
                           fdr_level: float = 0.05, use_fdr_correction: bool = True,
                           verbose: bool = True) -> Dict:
    """
    Perform bootstrap inference for Bradley-Terry model with FDR correction.
    
    This provides an alternative way to get confidence intervals and p-values
    by resampling the tests (columns) with replacement.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Rankings data
    target_benchmarker : int
        Target benchmarker to analyze
    test_type : str
        'superiority' or 'inferiority'
    compare_against : List[int]
        Benchmarkers to compare against
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for intervals
    fdr_level : float
        FDR level for multiple testing
    use_fdr_correction : bool
        Whether to apply FDR correction
    verbose : bool
        Print progress
    
    Returns:
    --------
    Dict with bootstrap results
    """
    if verbose:
        print("\nPerforming bootstrap analysis...")
        print(f"Number of bootstrap samples: {n_bootstrap}")
    
    # Store bootstrap results
    abilities_boot = []
    
    for i in range(n_bootstrap):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Bootstrap sample {i + 1}/{n_bootstrap}")
        
        # Resample tests (columns) with replacement
        n_tests = len(df.columns)
        boot_indices = np.random.choice(n_tests, n_tests, replace=True)
        boot_df = df.iloc[:, boot_indices]
        boot_df.columns = df.columns  # Keep original column names
        
        # Fit model on bootstrap sample
        try:
            boot_results = bradley_terry_analysis(
                boot_df, target_benchmarker, 
                test_type, compare_against,
                confidence_level=confidence_level,
                fdr_level=fdr_level,
                use_fdr_correction=use_fdr_correction,
                verbose=False
            )
            abilities_boot.append(boot_results['abilities']['ability'].values)
        except:
            # Skip if model fitting fails
            continue
    
    abilities_boot = np.array(abilities_boot)
    
    # Calculate bootstrap statistics
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    bootstrap_results = {
        'abilities_mean': np.mean(abilities_boot, axis=0),
        'abilities_std': np.std(abilities_boot, axis=0),
        'abilities_lower_ci': np.percentile(abilities_boot, lower_percentile, axis=0),
        'abilities_upper_ci': np.percentile(abilities_boot, upper_percentile, axis=0),
        'n_successful_boots': len(abilities_boot)
    }
    
    if verbose:
        print(f"\nBootstrap completed: {len(abilities_boot)}/{n_bootstrap} successful")
        print("\nBootstrap confidence intervals:")
        all_benchmarkers = sorted(list(set(
            int(x) for col in df.columns 
            for x in df[col].dropna()
        )))
        for i, bid in enumerate(all_benchmarkers):
            print(f"  Benchmarker {bid}: {bootstrap_results['abilities_mean'][i]:.3f} "
                  f"[{bootstrap_results['abilities_lower_ci'][i]:.3f}, "
                  f"{bootstrap_results['abilities_upper_ci'][i]:.3f}]")
    
    return bootstrap_results


# Example usage
if __name__ == "__main__":
    # Create example data
    data = {
        'Mouse DRG': [3, 2, 3, 4, 5, 6, 7],
        'Mouse Skin': [2, 2, 1, 4, 5, 6, 7],
        'Human T Diff': [3, 3, 5, 4, 1, 6, 7],
        'Mouse Bonemarrow': [3, 2, 3, 1, 5, 6, 7],
        'Mouse Spleen': [3, 2, 1, 4, 5, 6, 7],
        'Mouse Blood': [5, 3, 1, 4, 5, 6, 7],
        'Mouse Brain': [3, 3, 2, 4, 1, 6, 7],
        'Mouse CNS': [2, 2, 4, 1, 5, 6, 7],
        'Human Liver': [2.0, 1.0, 5.0, 3.0, np.nan, 6.0, 7.0]  # Example with NaN
    }
    
    df = pd.DataFrame(data)
    print("Input DataFrame:")
    print(df)
    print("\n" + "="*60)
    
    # Test 1: Superiority analysis
    print("\nTEST 1: Is Benchmarker 5 SUPERIOR to Benchmarkers 6 and 7?")
    print("="*60)
    results_superiority = bradley_terry_analysis(
        df=df,
        target_benchmarker=5,
        test_type='superiority',
        compare_against=[6, 7],
        confidence_level=0.95,
        fdr_level=0.05,
        use_fdr_correction=True,
        verbose=True
    )
    
    # Test 2: Inferiority analysis
    print("\n" + "="*60)
    print("\nTEST 2: Is Benchmarker 5 INFERIOR to Benchmarkers 1, 2, 3, and 4?")
    print("="*60)
    results_inferiority = bradley_terry_analysis(
        df=df,
        target_benchmarker=5,
        test_type='inferiority',
        compare_against=[1, 2, 3, 4],
        confidence_level=0.95,
        fdr_level=0.05,
        use_fdr_correction=True,
        verbose=True
    )
    
    # Create visualizations for superiority test
    print("\nCreating visualizations for superiority test...")
    fig1 = plot_bradley_terry_results(
        results_superiority, 
        target_benchmarker=5
    )
    plt.suptitle('Superiority Analysis: Benchmarker 5 vs [6, 7]', y=1.02)
    plt.show()
    
    # Create visualizations for inferiority test
    print("\nCreating visualizations for inferiority test...")
    fig2 = plot_bradley_terry_results(
        results_inferiority, 
        target_benchmarker=5
    )
    plt.suptitle('Inferiority Analysis: Benchmarker 5 vs [1, 2, 3, 4]', y=1.02)
    plt.show()
    
    # Optional: Compare with no FDR correction
    print("\n" + "="*60)
    print("COMPARISON: Superiority test WITHOUT FDR correction")
    print("="*60)
    results_no_fdr = bradley_terry_analysis(
        df=df,
        target_benchmarker=5,
        test_type='superiority',
        compare_against=[6, 7],
        confidence_level=0.95,
        use_fdr_correction=False,
        verbose=True
    )