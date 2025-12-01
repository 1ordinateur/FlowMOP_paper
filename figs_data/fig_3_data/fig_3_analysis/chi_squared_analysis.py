import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2
from scipy.stats import binom_test, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def get_benchmarker_name(benchmarker_id):
    """
    Map benchmarker ID to human-readable name.
    
    Parameters:
    -----------
    benchmarker_id : str or int
        The benchmarker ID
    
    Returns:
    --------
    str : The human-readable name
    """
    id_str = str(benchmarker_id)
    name_map = {
        '1': 'Expert 1',
        '2': 'Expert 2',
        '3': 'Expert 3',
        '4': 'Expert 4',
        '5': 'FlowMOP',
        '6': 'FlowCut',
        '7': 'PeacoQC'
    }
    return name_map.get(id_str, f'Method {id_str}')


def analyze_benchmarker_performance(df: pd.DataFrame,
                                   your_method: str = '7',
                                   control_methods: Optional[List[str]] = None,
                                   novel_methods: Optional[List[str]] = None,
                                   analysis_name: str = "",
                                   adjust_pairwise: bool = True,
                                   adjustment_method: str = 'two_stage_fdr',
                                   verbose: bool = True) -> Dict:
    """
    Analyze benchmarker performance using chi-squared tests with multiple comparison adjustment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with:
        - Rows: individual samples/tests
        - Columns: benchmarker IDs
        - Values: 1 for pass, 0 for fail
    your_method : str
        Column name for your method (default '7')
    control_methods : Optional[List[str]]
        List of control method column names (default ['1', '2', '3', '4'])
    novel_methods : Optional[List[str]]
        List of novel method column names (default ['5', '6'])
    analysis_name : str
        Name of the analysis (e.g., "Debris", "Time", "Doublets")
    adjust_pairwise : bool
        Whether to adjust p-values for multiple comparisons (default True)
    adjustment_method : str
        Method for p-value adjustment: 'two_stage_fdr' (recommended), 'bonferroni', 'holm', 
        'fdr_bh' (Benjamini-Hochberg), 'fdr_by' (Benjamini-Yekutieli), or 'none' 
        (default 'two_stage_fdr')
    verbose : bool
        Whether to print detailed results
    
    Returns:
    --------
    dict : Dictionary containing all test results and statistics
    """
    
    # Set defaults if not provided
    if control_methods is None:
        control_methods = ['1', '2', '3', '4']
    if novel_methods is None:
        novel_methods = ['5', '6']
    
    all_methods = control_methods + novel_methods + [your_method]
    
    # Filter to only existing columns
    existing_methods = [m for m in all_methods if m in df.columns]
    existing_controls = [m for m in control_methods if m in df.columns]
    existing_novels = [m for m in novel_methods if m in df.columns]
    
    results = {
        'analysis_name': analysis_name,
        'total_samples': len(df),
        'methods_analyzed': existing_methods,
        'adjustment_method': adjustment_method if adjust_pairwise else 'none'
    }
    
    if verbose:
        print("=" * 70)
        print(f"CHI-SQUARED ANALYSIS: {analysis_name if analysis_name else 'All Data'}")
        print("=" * 70)
        print(f"Total samples: {len(df)}")
        if adjust_pairwise:
            print(f"Multiple comparison adjustment: {adjustment_method}")
        print()
    
    # Calculate pass/fail statistics for each method
    method_stats = {}
    contingency_data = []
    
    for method in existing_methods:
        passes = df[method].sum()  # Sum of 1s (passes)
        failures = len(df) - passes  # Count of 0s (failures)
        pass_rate = passes / len(df)
        
        method_stats[method] = {
            'passes': int(passes),
            'failures': int(failures),
            'total': len(df),
            'pass_rate': pass_rate
        }
        contingency_data.append([passes, failures])
    
    # 1. OVERALL CHI-SQUARED TEST
    if verbose:
        print("1. OVERALL COMPARISON (Chi-squared test)")
        print("-" * 50)
    
    # Create contingency table
    contingency_table = np.array(contingency_data)
    
    # Perform chi-squared test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    results['overall_chi_squared'] = {
        'statistic': chi2_stat,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < 0.05
    }
    
    if verbose:
        print(f"Chi-squared statistic: {chi2_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print("✓ Significant differences exist between methods (p < 0.05)")
        else:
            print("✗ No significant differences between methods (p ≥ 0.05)")
        
        print("\nPass Rates by Benchmarker:")
        print(f"{'Benchmarker':<15} {'Pass Rate':<12} {'Passes/Total':<15}")
        print("-" * 45)
        
        for method in existing_methods:
            stats = method_stats[method]
            label = get_benchmarker_name(method)
            if method in existing_controls:
                label += " (C)"  # Control
            elif method in existing_novels:
                label += " (N)"  # Novel
            elif method == your_method:
                label += " (Y)"  # Your method
            
            print(f"{label:<15} {stats['pass_rate']:>6.1%}      {stats['passes']}/{stats['total']}")
    
    # 2. NON-INFERIORITY TEST vs Controls
    if your_method in method_stats and existing_controls:
        if verbose:
            print("\n2. NON-INFERIORITY TEST (Your Method vs Controls)")
            print("-" * 50)
        
        control_pass_rates = [method_stats[m]['pass_rate'] for m in existing_controls]
        your_pass_rate = method_stats[your_method]['pass_rate']
        
        mean_control = np.mean(control_pass_rates)
        sd_control = np.std(control_pass_rates, ddof=1) if len(control_pass_rates) > 1 else 0
        
        # Non-inferiority margin: mean - 1 SD
        lower_bound = mean_control - sd_control
        
        results['non_inferiority'] = {
            'control_mean': mean_control,
            'control_sd': sd_control,
            'threshold': lower_bound,
            'your_pass_rate': your_pass_rate,
            'is_non_inferior': your_pass_rate >= lower_bound
        }
        
        if verbose:
            print(f"Control methods mean pass rate: {mean_control:.1%}")
            print(f"Control methods SD: {sd_control:.1%}")
            print(f"Non-inferiority threshold (mean - 1SD): {lower_bound:.1%}")
            print(f"Your method pass rate: {your_pass_rate:.1%}")
            print()
            
            if your_pass_rate >= lower_bound:
                print(f"✓ Your method is NON-INFERIOR to controls")
                print(f"  (Pass rate {your_pass_rate:.1%} ≥ threshold {lower_bound:.1%})")
            else:
                print(f"✗ Your method is INFERIOR to controls")
                print(f"  (Pass rate {your_pass_rate:.1%} < threshold {lower_bound:.1%})")
    
    # 3. PAIRWISE COMPARISONS (ALL vs ALL except CONTROL vs CONTROL)
    results['pairwise_tests'] = {}
    
    if verbose:
        print("\n3. PAIRWISE COMPARISONS (All vs All, excluding Control vs Control)")
        print("-" * 50)
        if adjust_pairwise and adjustment_method != 'none':
            if adjustment_method == 'two_stage_fdr':
                print("Note: Using two-stage FDR procedure:")
                print("  Stage 1: Test for any difference (FDR-controlled)")
                print("  Stage 2: Test for superiority only if Stage 1 significant (FDR-controlled)")
            else:
                print(f"Note: P-values adjusted using {adjustment_method} method")
            print()
    
    # Collect all p-values for adjustment
    superiority_tests = []  # For superiority tests
    all_chi2_tests = []  # For all comparisons
    
    # Perform all pairwise comparisons (except control vs control)
    for i, method1 in enumerate(existing_methods):
        for j, method2 in enumerate(existing_methods):
            if i >= j:  # Skip diagonal and lower triangle (avoid duplicates)
                continue
                
            # Skip control vs control comparisons
            if method1 in existing_controls and method2 in existing_controls:
                if verbose:
                    print(f"Skipping Control vs Control: {get_benchmarker_name(method1)} vs {get_benchmarker_name(method2)}")
                continue
            
            # Get statistics for both methods
            method1_passes = method_stats[method1]['passes']
            method1_failures = method_stats[method1]['failures']
            method2_passes = method_stats[method2]['passes']
            method2_failures = method_stats[method2]['failures']
            
            # Create 2x2 contingency table
            cont_2x2 = np.array([
                [method1_passes, method1_failures],
                [method2_passes, method2_failures]
            ])
            
            # Chi-squared test
            chi2_stat_pair, p_value_chi2, _, _ = chi2_contingency(cont_2x2)
            
            # Determine test type and direction
            # If method1 is your method and method2 is novel, test for superiority
            # If both are novel or one is your method, also test for differences
            test_superiority = False
            alternative = 'two-sided'
            comparison_label = f"{method1}_vs_{method2}"
            
            # Determine which method should be tested as superior
            if method1 == your_method and method2 in existing_novels:
                test_superiority = True
                alternative = 'larger'
                superior_method = method1
                inferior_method = method2
            elif method2 == your_method and method1 in existing_novels:
                test_superiority = True
                alternative = 'larger' 
                superior_method = method2
                inferior_method = method1
                # Swap for correct ordering in z-test
                method1_passes, method2_passes = method2_passes, method1_passes
                method1_failures, method2_failures = method2_failures, method1_failures
            elif method1 in existing_novels and method2 in existing_novels:
                # Novel vs novel - could test both directions or just difference
                test_superiority = False
            
            # Perform appropriate statistical test
            if test_superiority:
                # One-sided test for superiority using z-test
                counts = [method1_passes, method2_passes]
                nobs = [method_stats[method1]['total'], method_stats[method2]['total']]
                
                try:
                    z_stat, p_value_superiority = proportions_ztest(counts, nobs, alternative=alternative)
                except:
                    z_stat, p_value_superiority = np.nan, np.nan
                
                # Fisher's exact test for small samples
                if min(method1_passes, method1_failures, method2_passes, method2_failures) < 5:
                    odds_ratio, p_value_fisher = fisher_exact(cont_2x2, alternative='greater' if alternative == 'larger' else 'two-sided')
                else:
                    odds_ratio, p_value_fisher = np.nan, np.nan
                
                superiority_tests.append({
                    'method1': superior_method if 'superior_method' in locals() else method1,
                    'method2': inferior_method if 'inferior_method' in locals() else method2,
                    'p_value': p_value_superiority if not np.isnan(p_value_superiority) else 1.0
                })
            else:
                z_stat = np.nan
                p_value_superiority = np.nan
                p_value_fisher = np.nan
                odds_ratio = np.nan
            
            # Determine types for display
            if method1 == your_method or method2 == your_method:
                comparison_type = 'your_method'
            elif (method1 in existing_novels and method2 in existing_controls) or \
                 (method2 in existing_novels and method1 in existing_controls):
                comparison_type = 'novel_vs_control'
            elif method1 in existing_novels and method2 in existing_novels:
                comparison_type = 'novel_vs_novel'
            else:
                comparison_type = 'other'
            
            all_chi2_tests.append({
                'method1': method1,
                'method2': method2,
                'comparison_type': comparison_type,
                'chi2_stat': chi2_stat_pair,
                'chi2_p': p_value_chi2,
                'z_stat': z_stat,
                'sup_p': p_value_superiority,
                'fisher_p': p_value_fisher,
                'odds_ratio': odds_ratio,
                'test_superiority': test_superiority
            })
    
    # Apply multiple comparison adjustment
    if adjust_pairwise and adjustment_method != 'none':
        
        if adjustment_method == 'two_stage_fdr':
            # Two-stage FDR procedure
            # Stage 1: Test for any difference using chi-squared tests with FDR control
            chi2_p_values = [t['chi2_p'] for t in all_chi2_tests]
            reject_chi2, adjusted_chi2_p, _, _ = multipletests(chi2_p_values, method='fdr_bh', alpha=0.05)
            
            for i, test in enumerate(all_chi2_tests):
                test['adjusted_chi2_p'] = adjusted_chi2_p[i]
                test['stage1_significant'] = reject_chi2[i]
            
            # Stage 2: Only test superiority for tests that showed significant difference in Stage 1
            # This controls the FDR within the subset of discoveries from Stage 1
            eligible_superiority_tests = []
            for test in superiority_tests:
                # Find corresponding chi2 test
                chi2_test = next((t for t in all_chi2_tests if 
                                t['method1'] == test['method1'] and t['method2'] == test['method2']), None)
                if chi2_test and chi2_test.get('stage1_significant', False):
                    eligible_superiority_tests.append(test)
                else:
                    # Not eligible for Stage 2 testing
                    test['adjusted_p'] = 1.0  # Set to 1 as it wasn't tested in Stage 2
            
            # Apply FDR to eligible superiority tests
            if eligible_superiority_tests:
                eligible_p_values = [t['p_value'] for t in eligible_superiority_tests]
                _, adjusted_sup_p, _, _ = multipletests(eligible_p_values, method='fdr_bh', alpha=0.05)
                for i, test in enumerate(eligible_superiority_tests):
                    test['adjusted_p'] = adjusted_sup_p[i]
            
            # Fill in non-eligible tests
            for test in superiority_tests:
                if 'adjusted_p' not in test:
                    test['adjusted_p'] = test['p_value']  # Keep original if not in Stage 2
                    
        else:
            # Standard adjustment methods
            # Adjust superiority p-values
            if superiority_tests:
                sup_p_values = [t['p_value'] for t in superiority_tests]
                
                # Use statsmodels for adjustment
                if adjustment_method in ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']:
                    _, adjusted_sup_p, _, _ = multipletests(sup_p_values, method=adjustment_method, alpha=0.05)
                    for i, test in enumerate(superiority_tests):
                        test['adjusted_p'] = adjusted_sup_p[i]
                else:
                    for test in superiority_tests:
                        test['adjusted_p'] = test['p_value']
            
            # Adjust chi-squared p-values for all comparisons
            chi2_p_values = [t['chi2_p'] for t in all_chi2_tests]
            if adjustment_method in ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']:
                _, adjusted_chi2_p, _, _ = multipletests(chi2_p_values, method=adjustment_method, alpha=0.05)
                for i, test in enumerate(all_chi2_tests):
                    test['adjusted_chi2_p'] = adjusted_chi2_p[i]
            else:
                for test in all_chi2_tests:
                    test['adjusted_chi2_p'] = test['chi2_p']
    else:
        # No adjustment
        for test in superiority_tests:
            test['adjusted_p'] = test['p_value']
        for test in all_chi2_tests:
            test['adjusted_chi2_p'] = test['chi2_p']
    
    # Create superiority lookup
    sup_lookup = {}
    for test in superiority_tests:
        key = (test['method1'], test['method2'])
        sup_lookup[key] = test
    
    # Store results and display
    for test in all_chi2_tests:
        method1 = test['method1']
        method2 = test['method2']
        
        # Determine significance
        key = (method1, method2)
        if test['test_superiority'] and key in sup_lookup:
            is_superior = sup_lookup[key]['adjusted_p'] < 0.05
            sup_p_orig = sup_lookup[key]['p_value']
            sup_p_adj = sup_lookup[key]['adjusted_p']
        else:
            is_superior = False
            sup_p_orig = test['sup_p']
            sup_p_adj = test['sup_p']
        
        sig_different = test['adjusted_chi2_p'] < 0.05
        
        # Store results
        results['pairwise_tests'][f'{method1}_vs_{method2}'] = {
            'method1': method1,
            'method1_passes': method_stats[method1]['passes'],
            'method1_pass_rate': method_stats[method1]['pass_rate'],
            'method2': method2,
            'method2_passes': method_stats[method2]['passes'],
            'method2_pass_rate': method_stats[method2]['pass_rate'],
            'chi2_statistic': test['chi2_stat'],
            'chi2_p_value': test['chi2_p'],
            'chi2_p_adjusted': test['adjusted_chi2_p'],
            'z_statistic': test['z_stat'],
            'superiority_p_value': sup_p_orig,
            'superiority_p_adjusted': sup_p_adj,
            'fisher_p_value': test['fisher_p'],
            'odds_ratio': test['odds_ratio'],
            'is_superior': is_superior,
            'significantly_different': sig_different,
            'comparison_type': test['comparison_type']
        }
        
        # Display results
        if verbose:
            # Determine label for display
            label1 = get_benchmarker_name(method1)
            label2 = get_benchmarker_name(method2)
            
            if method1 in existing_controls:
                label1 += " (Control)"
            elif method1 in existing_novels:
                label1 += " (Novel)"
            if method1 == your_method:
                label1 += " [YOUR METHOD]"
                
            if method2 in existing_controls:
                label2 += " (Control)"
            elif method2 in existing_novels:
                label2 += " (Novel)"
            if method2 == your_method:
                label2 += " [YOUR METHOD]"
            
            print(f"\n{label1} vs {label2}:")
            
            stats1 = method_stats[method1]
            stats2 = method_stats[method2]
            print(f"  {label1}: {stats1['passes']}/{len(df)} passes ({stats1['pass_rate']:.1%})")
            print(f"  {label2}: {stats2['passes']}/{len(df)} passes ({stats2['pass_rate']:.1%})")
            print(f"  Difference: {(stats1['pass_rate'] - stats2['pass_rate']):.1%}")
            
            # Chi-squared test results
            if adjust_pairwise and adjustment_method != 'none':
                print(f"  Chi-squared: {test['chi2_stat']:.4f}")
                print(f"    Original p-value: {test['chi2_p']:.4f}")
                print(f"    Adjusted p-value: {test['adjusted_chi2_p']:.4f}")
            else:
                print(f"  Chi-squared: {test['chi2_stat']:.4f} (p={test['chi2_p']:.4f})")
            
            # Superiority test results if applicable
            if test['test_superiority']:
                # Check if method was filtered out in Stage 1 (for two-stage FDR)
                if adjustment_method == 'two_stage_fdr' and not test.get('stage1_significant', True):
                    print(f"  [Stage 1: No significant difference detected]")
                    print(f"  [Stage 2: Superiority test not performed]")
                elif not np.isnan(test['z_stat']):
                    print(f"  Z-statistic (superiority): {test['z_stat']:.4f}")
                    if adjust_pairwise and adjustment_method != 'none':
                        if adjustment_method == 'two_stage_fdr':
                            print(f"    Stage 2 p-value (original): {sup_p_orig:.4f}")
                            print(f"    Stage 2 p-value (adjusted): {sup_p_adj:.4f}")
                        else:
                            print(f"    Original p-value: {sup_p_orig:.4f}")
                            print(f"    Adjusted p-value: {sup_p_adj:.4f}")
                    else:
                        print(f"    P-value (one-sided): {sup_p_orig:.4f}")
                
                if not np.isnan(test['fisher_p']):
                    print(f"  Fisher's exact p-value: {test['fisher_p']:.4f}")
                
                if is_superior:
                    print(f"  ✓ {label1} is SUPERIOR to {label2}")
                else:
                    print(f"  ✗ {label1} is NOT significantly superior to {label2}")
            
            # Results for non-superiority tests
            else:
                if sig_different:
                    if stats1['pass_rate'] > stats2['pass_rate']:
                        print(f"  ✓ {label1} significantly better than {label2}")
                    else:
                        print(f"  ✗ {label1} significantly worse than {label2}")
                else:
                    print(f"  ≈ No significant difference between methods")
    
    # 4. SUMMARY
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Overall test
        if results['overall_chi_squared']['significant']:
            print("✓ SIGNIFICANT: Methods show statistically significant differences")
        else:
            print("✗ NOT SIGNIFICANT: No statistical differences between methods")
        
        # Non-inferiority
        if 'non_inferiority' in results:
            if results['non_inferiority']['is_non_inferior']:
                print("✓ NON-INFERIOR: Your method performs as well as controls")
            else:
                print("✗ INFERIOR: Your method underperforms compared to controls")
        
        # Superiority (using adjusted p-values)
        if 'pairwise_tests' in results and your_method in existing_methods:
            superior_to = []
            inferior_to = []
            equal_to = []
            
            for key, test_result in results['pairwise_tests'].items():
                # Check if your method is involved in this comparison
                if test_result['method1'] == your_method:
                    other_method = test_result['method2']
                    if test_result.get('is_superior', False):
                        superior_to.append(get_benchmarker_name(other_method))
                    elif test_result['significantly_different']:
                        if test_result['method1_pass_rate'] < test_result['method2_pass_rate']:
                            inferior_to.append(get_benchmarker_name(other_method))
                    else:
                        equal_to.append(get_benchmarker_name(other_method))
                elif test_result['method2'] == your_method:
                    other_method = test_result['method1']
                    if test_result['significantly_different']:
                        if test_result['method2_pass_rate'] < test_result['method1_pass_rate']:
                            inferior_to.append(get_benchmarker_name(other_method))
                        elif test_result['method2_pass_rate'] > test_result['method1_pass_rate']:
                            superior_to.append(get_benchmarker_name(other_method))
                    else:
                        equal_to.append(get_benchmarker_name(other_method))
            
            if superior_to:
                print(f"✓ SUPERIOR to: {', '.join(superior_to)}")
            if inferior_to:
                print(f"✗ INFERIOR to: {', '.join(inferior_to)}")
            if equal_to:
                print(f"≈ EQUIVALENT to: {', '.join(equal_to)}")
            
            if not superior_to and not inferior_to and not equal_to:
                print("No comparisons with your method found")
            elif adjust_pairwise:
                print(f"  (After {adjustment_method} adjustment)")
        
        if your_method in method_stats:
            print(f"\nYour method achieved {method_stats[your_method]['pass_rate']:.1%} pass rate")
            print(f"across {len(df)} samples")
    
    # Store method statistics in results
    results['method_stats'] = method_stats
    
    return results


def plot_pass_rates(df: pd.DataFrame,
                   your_method: str = '7',
                   control_methods: Optional[List[str]] = None,
                   novel_methods: Optional[List[str]] = None,
                   analysis_name: str = "",
                   figsize: Tuple[float, float] = (8, 6),
                   save_path: Optional[str] = None,
                   show_plot: bool = True) -> plt.Figure:
    """
    Plot pass rates as a clean bar graph with customized colors for novel methods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with pass/fail data (1/0) for each method
    your_method : str
        Column name for your method (default '7' for FlowMOP)
    control_methods : Optional[List[str]]
        List of control method column names
    novel_methods : Optional[List[str]]
        List of novel method column names
    analysis_name : str
        Title for the plot
    figsize : Tuple[float, float]
        Figure size (width, height)
    save_path : Optional[str]
        Path to save the figure
    show_plot : bool
        Whether to display the plot
    
    Returns:
    --------
    plt.Figure : The matplotlib figure object
    """
    
    # Set defaults if not provided
    if control_methods is None:
        control_methods = ['1', '2', '3', '4']
    if novel_methods is None:
        novel_methods = ['5', '6']
    
    # Define color scheme
    # Control methods: shades of gray
    control_colors = ['#808080', '#696969', '#A9A9A9', '#C0C0C0']
    
    # Novel methods colors - CORRECTED MAPPING
    # Based on actual IDs: 5=FlowMOP, 6=FlowCut, 7=PeacoQC
    novel_colors = {
        '5': '#4682B4',  # FlowMOP - steel blue
        '6': '#D2691E',  # FlowCut - brown/sienna
        '7': '#228B22'   # PeacoQC - forest green
    }
    
    # Prepare data - reorder to show FlowMOP, PeacoQC, FlowCut
    # First get existing controls and novels
    existing_controls = [m for m in control_methods if m in df.columns]
    existing_novels = [m for m in novel_methods + [your_method] if m in df.columns]
    
    # Reorder novels: FlowMOP (5), PeacoQC (7), FlowCut (6)
    novel_order = []
    if '5' in existing_novels:
        novel_order.append('5')  # FlowMOP
    if '7' in existing_novels:
        novel_order.append('7')  # PeacoQC  
    if '6' in existing_novels:
        novel_order.append('6')  # FlowCut
    # Add any other novels that might exist
    for m in existing_novels:
        if m not in novel_order:
            novel_order.append(m)
    
    # Combine: controls first, then reordered novels
    existing_methods = existing_controls + novel_order
    
    # Calculate pass rates
    pass_rates = []
    method_labels = []
    colors = []
    
    for method in existing_methods:
        pass_rate = df[method].mean() * 100  # Convert to percentage
        pass_rates.append(pass_rate)
        
        # Set labels using the helper function
        label = get_benchmarker_name(method)
        method_labels.append(label)
        
        # Set colors based on the actual label name, not the method ID
        if 'Expert' in label:
            # Use gray for Expert methods
            colors.append('#808080')
        elif label == 'FlowMOP':
            colors.append('#4682B4')  # Steel blue
        elif label == 'FlowCut':
            colors.append('#D2691E')  # Brown/sienna
        elif label == 'PeacoQC':
            colors.append('#228B22')  # Forest green
        else:
            colors.append('#808080')  # Default gray
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    
    # Create bar plot
    x_pos = np.arange(len(method_labels))
    bars = ax.bar(x_pos, pass_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Benchmarker', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    
    if analysis_name:
        ax.set_title(f'{analysis_name} - Pass Rates by Benchmarker', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Pass Rates by Benchmarker', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=0, ha='center')
    
    # Set y-axis
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    
    # Add value labels on top of bars
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend to distinguish control vs novel methods
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', edgecolor='black', label='Expert Methods'),
        Patch(facecolor='#4682B4', edgecolor='black', label='FlowMOP'),
        Patch(facecolor='#D2691E', edgecolor='black', label='FlowCut'),
        Patch(facecolor='#228B22', edgecolor='black', label='PeacoQC')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig